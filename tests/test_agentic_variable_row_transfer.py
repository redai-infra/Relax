# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import copy
import random
from types import SimpleNamespace

import pytest

from relax.agentic.pipeline.transfer import (
    AGENTIC_ROW_IDENTITY_KEY,
    AGENTIC_ROW_IDENTITY_TAG_KEY,
    AGENTIC_ROW_IDENTITY_TAGS_FIELD,
    AGENTIC_VARIABLE_ROW_PADDING_KEY,
    TransferDomain,
    _variable_row_padding_sample,
)
from relax.utils.types import Sample


MAX_ROWS_ENV = "RELAX_AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE"


@pytest.fixture(autouse=True)
def _enable_variable_rows(monkeypatch):
    monkeypatch.setenv(MAX_ROWS_ENV, "50")


class _FakeRolloutBatch(dict):
    def numel(self):
        return len(self["total_lengths"])


class _RecordingDataClient:
    def __init__(self, *, delay_first: bool = False):
        self.calls: list[dict] = []
        self.events: list[str] = []
        self.delay_first = delay_first

    async def async_put(self, *, data, partition_id, custom_meta, is_last):
        call_number = len(self.calls)
        self.events.append(f"start-{call_number}")
        if self.delay_first and call_number == 0:
            await asyncio.sleep(0.01)
        self.calls.append(
            {
                "data": data,
                "partition_id": partition_id,
                "custom_meta": custom_meta,
                "is_last": is_last,
            }
        )
        self.events.append(f"end-{call_number}")


def _args(**overrides):
    values = {
        "rollout_batch_size": 2,
        "over_sampling_batch_size": 2,
        "n_samples_per_prompt": 1,
        "colocate": False,
        "global_batch_size": 4,
        "num_iters_per_train_update": 4,
        "group_rm": True,
        "agentic_custom_advantage_path": "examples.graphgpo.advantage.compute",
        "use_dynamic_batch_size": True,
        "fully_async": False,
        "hybrid": False,
        "reward_key": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_variable_row_transfer_is_explicitly_opt_in(monkeypatch):
    monkeypatch.delenv(MAX_ROWS_ENV)

    transfer = TransferDomain(args=_args(), data_system_client=None)

    assert transfer._variable_row_mode is False


def test_variable_row_transfer_rejects_export_above_partition_bound(monkeypatch):
    monkeypatch.setenv(MAX_ROWS_ENV, "1")
    transfer = TransferDomain(args=_args(), data_system_client=None)

    with pytest.raises(RuntimeError, match="exceeded the configured partition bound"):
        transfer._prepare_variable_row_groups(
            groups=[_group(0, 3)],
            partition_rollout_id=0,
            is_last=False,
        )

    assert transfer._partition_actual_rows == {}


def _group(group_index: int, rows: int) -> list[Sample]:
    group: list[Sample] = []
    trajectory_id = f"trajectory-{group_index}"
    for turn_index in range(rows):
        sample = Sample(
            group_index=group_index,
            index=group_index,
            session_id=trajectory_id,
            tokens=[group_index + 1, turn_index + 11],
            response_length=1,
            reward=1.0,
            custom_advantage=1.0,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
            weight_versions=["7"],
            metadata={
                "row_id": f"row-{group_index}-{turn_index}",
                "rollout_group_id": str(group_index),
                "policy_version": "7",
                "task_id": f"task-{group_index}",
                "trajectory_id": trajectory_id,
                "turn_id": f"turn_{turn_index:03d}",
                "turn_index": turn_index,
                "terminal": turn_index == rows - 1,
                "truncated": False,
            },
        )
        setattr(sample, "_agentic_export_name", f"turn_{turn_index:03d}")
        group.append(sample)
    return group


def _multi_trajectory_group(
    *,
    group_index: int,
    task_id: str,
    trajectory_lengths: list[int],
    policy_version: str = "11",
) -> list[Sample]:
    group: list[Sample] = []
    for slot_index, trajectory_length in enumerate(trajectory_lengths):
        trajectory_id = f"trajectory-{group_index}-{slot_index}"
        sample_index = group_index * 100 + slot_index
        for turn_index in range(trajectory_length):
            token_id = group_index * 10_000 + slot_index * 100 + turn_index
            sample = Sample(
                group_index=group_index,
                index=sample_index,
                session_id=trajectory_id,
                tokens=[31, token_id],
                response_length=1,
                reward=1.0,
                custom_advantage=1.0,
                loss_mask=[1],
                rollout_log_probs=[-0.25],
                weight_versions=[policy_version],
                metadata={
                    "row_id": f"row-{group_index}-{slot_index}-{turn_index}",
                    "rollout_group_id": str(group_index),
                    "policy_version": policy_version,
                    "task_id": task_id,
                    "trajectory_id": trajectory_id,
                    "turn_id": f"turn_{turn_index:03d}",
                    "turn_index": turn_index,
                    "terminal": turn_index == trajectory_length - 1,
                    "truncated": False,
                },
            )
            setattr(sample, "_agentic_export_name", f"turn_{turn_index:03d}")
            group.append(sample)
    random.Random(group_index).shuffle(group)
    return group


def _install_fake_converter(monkeypatch, converted_samples: list[list[Sample]]) -> None:
    def _convert(_args, samples):
        converted_samples.append(list(samples))
        return _FakeRolloutBatch(
            tokens=[list(sample.tokens) for sample in samples],
            total_lengths=[len(sample.tokens) for sample in samples],
            response_lengths=[sample.response_length for sample in samples],
            loss_masks=[
                [0] * sample.response_length if sample.remove_sample else list(sample.loss_mask) for sample in samples
            ],
            rollout_log_probs=[list(sample.rollout_log_probs or []) for sample in samples],
        )

    monkeypatch.setattr("relax.utils.utils.convert_samples_to_train_data", _convert)


async def _run_partition(
    transfer: TransferDomain,
    *,
    rollout_id: int,
    previous_partition_quota: int,
    current_partition_quota: int,
    groups: list[list[Sample]],
) -> None:
    transfer.rebind_step(rollout_id=rollout_id)
    transfer.configure_transfer_quota(
        previous_partition_quota=previous_partition_quota,
        current_partition_quota=current_partition_quota,
    )
    transfer.enqueue_ready_groups(groups)
    await transfer.drain_ready_group_payloads()
    await transfer.wait_for_pending_transfers()


async def _run_partition_in_batches(
    transfer: TransferDomain,
    *,
    rollout_id: int,
    current_partition_quota: int,
    batches: list[list[list[Sample]]],
) -> None:
    transfer.rebind_step(rollout_id=rollout_id)
    transfer.configure_transfer_quota(
        previous_partition_quota=0,
        current_partition_quota=current_partition_quota,
    )
    for groups in batches:
        transfer.enqueue_ready_groups(groups)
        await transfer.drain_ready_group_payloads()
        await asyncio.sleep(0)
    await transfer.wait_for_pending_transfers()


def test_variable_row_transfer_five_rows_adds_three_final_padding_rows_and_serializes_puts(monkeypatch):
    converted_samples: list[list[Sample]] = []
    _install_fake_converter(monkeypatch, converted_samples)
    client = _RecordingDataClient(delay_first=True)
    transfer = TransferDomain(args=_args(), data_system_client=client)

    asyncio.run(
        _run_partition_in_batches(
            transfer,
            rollout_id=7,
            current_partition_quota=2,
            batches=[[_group(0, 2)], [_group(1, 3)]],
        )
    )

    assert [len(samples) for samples in converted_samples] == [2, 6]
    assert [call["partition_id"] for call in client.calls] == ["train_7", "train_7"]
    assert [call["is_last"] for call in client.calls] == [False, True]
    assert client.events == ["start-0", "end-0", "start-1", "end-1"]
    assert [meta[AGENTIC_VARIABLE_ROW_PADDING_KEY] for call in client.calls for meta in call["custom_meta"]] == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]

    final_samples = converted_samples[-1]
    assert [sample.remove_sample for sample in final_samples] == [False, False, False, True, True, True]
    for padding_sample in final_samples[-3:]:
        assert padding_sample.custom_advantage == 0.0
        assert padding_sample.reward == 0.0
        assert padding_sample.metadata[AGENTIC_VARIABLE_ROW_PADDING_KEY] is True
        assert padding_sample.metadata["row_id"].startswith("agentic-padding-v1:7:")
    physical_row_ids = [
        meta[AGENTIC_ROW_IDENTITY_KEY]["row_id"] for call in client.calls for meta in call["custom_meta"]
    ]
    assert len(physical_row_ids) == len(set(physical_row_ids))
    assert all(
        int(data_tag) == meta[AGENTIC_ROW_IDENTITY_TAG_KEY]
        for call in client.calls
        for data_tag, meta in zip(
            call["data"][AGENTIC_ROW_IDENTITY_TAGS_FIELD],
            call["custom_meta"],
            strict=True,
        )
    )
    timing = transfer.transfer_timing_snapshot()
    assert len(timing) == 2
    assert all(record["ok"] is True for record in timing)
    assert all(
        record[key] >= 0
        for record in timing
        for key in ("reorder_ms", "identity_validation_ms", "serialization_ms", "queue_transfer_ms")
    )
    assert transfer._partition_actual_rows == {}
    assert transfer._partition_identity_state == {}


def test_variable_row_transfer_exact_eight_rows_adds_no_padding(monkeypatch):
    converted_samples: list[list[Sample]] = []
    _install_fake_converter(monkeypatch, converted_samples)
    client = _RecordingDataClient()
    transfer = TransferDomain(args=_args(), data_system_client=client)

    asyncio.run(
        _run_partition_in_batches(
            transfer,
            rollout_id=3,
            current_partition_quota=2,
            batches=[[_group(0, 4)], [_group(1, 4)]],
        )
    )

    assert [len(samples) for samples in converted_samples] == [4, 4]
    assert [call["is_last"] for call in client.calls] == [False, True]
    assert all(
        meta[AGENTIC_VARIABLE_ROW_PADDING_KEY] is False for call in client.calls for meta in call["custom_meta"]
    )


def test_variable_row_transfer_preserves_previous_partition_row_count_across_rebind(monkeypatch):
    converted_samples: list[list[Sample]] = []
    _install_fake_converter(monkeypatch, converted_samples)
    client = _RecordingDataClient()
    transfer = TransferDomain(args=_args(), data_system_client=client)

    async def _scenario():
        await _run_partition(
            transfer,
            rollout_id=10,
            previous_partition_quota=0,
            current_partition_quota=2,
            groups=[_group(0, 2)],
        )
        assert transfer._partition_actual_rows == {10: 2}
        await _run_partition(
            transfer,
            rollout_id=11,
            previous_partition_quota=1,
            current_partition_quota=0,
            groups=[_group(1, 3)],
        )

    asyncio.run(_scenario())

    assert [call["partition_id"] for call in client.calls] == ["train_10", "train_10"]
    assert [call["is_last"] for call in client.calls] == [False, True]
    assert [len(samples) for samples in converted_samples] == [2, 6]
    assert sum(meta[AGENTIC_VARIABLE_ROW_PADDING_KEY] for call in client.calls for meta in call["custom_meta"]) == 3
    assert transfer._partition_actual_rows == {}


def test_two_tasks_eight_shuffled_unequal_trajectories_round_trip_identity_and_action_rows(monkeypatch):
    converted_samples: list[list[Sample]] = []
    _install_fake_converter(monkeypatch, converted_samples)
    client = _RecordingDataClient()
    args = _args(
        rollout_batch_size=2,
        over_sampling_batch_size=2,
        n_samples_per_prompt=8,
        colocate=True,
        global_batch_size=16,
        num_iters_per_train_update=1,
    )
    transfer = TransferDomain(args=args, data_system_client=client)
    task_a = _multi_trajectory_group(
        group_index=10,
        task_id="task-a",
        trajectory_lengths=[1, 3, 2, 4, 1, 2, 3, 5],
    )
    task_b = _multi_trajectory_group(
        group_index=20,
        task_id="task-b",
        trajectory_lengths=[2, 1, 4, 3, 2, 5, 1, 3],
    )
    expected_tokens = {sample.metadata["row_id"]: list(sample.tokens) for sample in [*task_a, *task_b]}

    asyncio.run(
        _run_partition(
            transfer,
            rollout_id=13,
            previous_partition_quota=0,
            current_partition_quota=2,
            groups=[task_b, task_a],
        )
    )

    assert len(client.calls) == 1
    call = client.calls[0]
    shuffled_round_trip = list(
        zip(
            call["custom_meta"],
            call["data"]["tokens"],
            call["data"]["loss_masks"],
            call["data"][AGENTIC_ROW_IDENTITY_TAGS_FIELD],
            strict=True,
        )
    )
    random.Random(123).shuffle(shuffled_round_trip)
    real_identities = []
    for custom_meta, tokens, loss_mask, row_tag in shuffled_round_trip:
        identity = custom_meta[AGENTIC_ROW_IDENTITY_KEY]
        assert int(row_tag) == custom_meta[AGENTIC_ROW_IDENTITY_TAG_KEY]
        if identity["padding"]:
            assert loss_mask == [0]
            assert identity["rollout_group_id"] is None
            continue
        real_identities.append(identity)
        assert tokens == expected_tokens[identity["row_id"]]
        assert loss_mask == [1]
        assert identity["action_token_count"] == 1
    assert {identity["task_id"] for identity in real_identities} == {"task-a", "task-b"}
    assert {identity["rollout_group_id"] for identity in real_identities} == {"10", "20"}
    assert {identity["policy_version"] for identity in real_identities} == {"11"}
    assert len({identity["trajectory_id"] for identity in real_identities}) == 16
    assert len({identity["row_id"] for identity in real_identities}) == len(expected_tokens)


def test_variable_row_identity_rejects_duplicate_retry_incomplete_group_and_policy_mix():
    transfer = TransferDomain(args=_args(), data_system_client=None)
    first_group = _group(0, 2)
    transfer._prepare_variable_row_groups(
        groups=[first_group],
        partition_rollout_id=5,
        is_last=False,
    )
    with pytest.raises(RuntimeError, match="duplicate row_id|more than once"):
        transfer._prepare_variable_row_groups(
            groups=[copy.deepcopy(first_group)],
            partition_rollout_id=5,
            is_last=False,
        )

    different_policy = _group(1, 1)
    for sample in different_policy:
        sample.weight_versions = ["8"]
        sample.metadata["policy_version"] = "8"
    with pytest.raises(RuntimeError, match="mixes policy versions"):
        transfer._prepare_variable_row_groups(
            groups=[different_policy],
            partition_rollout_id=5,
            is_last=True,
        )

    incomplete_transfer = TransferDomain(
        args=_args(n_samples_per_prompt=2),
        data_system_client=None,
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        incomplete_transfer._prepare_variable_row_groups(
            groups=[_group(9, 2)],
            partition_rollout_id=9,
            is_last=False,
        )


def test_variable_row_identity_rejects_duplicate_row_inside_group_and_stale_restart_state():
    duplicate_group = _group(0, 2)
    duplicate_group[1].metadata["row_id"] = duplicate_group[0].metadata["row_id"]
    transfer = TransferDomain(args=_args(), data_system_client=None)
    with pytest.raises(RuntimeError, match="duplicate row_id"):
        transfer._prepare_variable_row_groups(
            groups=[duplicate_group],
            partition_rollout_id=0,
            is_last=False,
        )

    transfer._partition_identity_state[1] = {
        "row_ids": {"residual-row"},
        "group_manifests": {},
        "policy_version": "7",
    }
    with pytest.raises(RuntimeError, match="stale or future"):
        transfer.rebind_step(rollout_id=3)


def test_default_transfer_mode_does_not_add_padding(monkeypatch):
    converted_samples: list[list[Sample]] = []
    _install_fake_converter(monkeypatch, converted_samples)
    client = _RecordingDataClient()
    transfer = TransferDomain(
        args=_args(
            rollout_batch_size=1,
            over_sampling_batch_size=1,
            group_rm=False,
        ),
        data_system_client=client,
    )

    asyncio.run(
        _run_partition(
            transfer,
            rollout_id=4,
            previous_partition_quota=0,
            current_partition_quota=1,
            groups=[_group(0, 5)],
        )
    )

    assert [len(samples) for samples in converted_samples] == [5]
    assert client.calls[0]["is_last"] is True
    assert client.calls[0]["custom_meta"] == [{"total_lengths": 2}] * 5
    assert transfer._partition_actual_rows == {}
    assert transfer.transfer_timing_snapshot() == []


def test_variable_row_transfer_rejects_fully_async_mode():
    with pytest.raises(ValueError, match="synchronous training only"):
        TransferDomain(
            args=_args(fully_async=True, hybrid=False),
            data_system_client=object(),
        )


def test_variable_row_transfer_rejects_hybrid_mode():
    with pytest.raises(ValueError, match="synchronous training only"):
        TransferDomain(
            args=_args(fully_async=True, hybrid=True),
            data_system_client=None,
        )


def test_discard_pending_transfers_clears_variable_partition_accounting():
    transfer = TransferDomain(args=_args(), data_system_client=None)
    transfer._partition_actual_rows[9] = 3
    transfer._partition_identity_state[9] = {
        "row_ids": {"row"},
        "group_manifests": {},
        "policy_version": "7",
    }

    asyncio.run(transfer.discard_pending_transfers())

    assert transfer._partition_actual_rows == {}
    assert transfer._partition_identity_state == {}


def test_padding_sample_converts_to_zero_loss_mask():
    from relax.utils.utils import convert_samples_to_train_data

    template = _group(0, 1)[0]
    template.metadata["raw_reward"] = 9.0
    padding_sample = _variable_row_padding_sample(args=_args(), template=template)
    convert_args = SimpleNamespace(
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path="examples.graphgpo.advantage.compute",
        reward_key=None,
        multimodal_keys=None,
        use_opd=False,
        debug_train_only=True,
    )

    batch = convert_samples_to_train_data(convert_args, [padding_sample])

    assert batch["loss_masks"] == [[0]]
    assert batch["rewards"] == [0.0]
    assert batch["raw_reward"] == [0.0]
    assert template.remove_sample is False
    assert template.metadata["raw_reward"] == 9.0
