# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import subprocess
import sys
import textwrap

import pytest


HAS_INFERENCE_DEPS = all(importlib.util.find_spec(name) is not None for name in ("ray", "sglang", "sglang_router"))


@pytest.mark.skipif(not HAS_INFERENCE_DEPS, reason="Missing ray/sglang dependencies")
def test_completed_surplus_group_is_not_generated_or_finalized_twice() -> None:
    """Run the real group lifecycle without importing the DCS backend.

    ``sglang_rollout`` reaches the DCS package through an unrelated engine
    import. Isolating that boundary in a subprocess keeps this CPU test below
    the no-GPU container memory limit without contaminating pytest's module
    state or replacing the rollout function under test.
    """
    script = textwrap.dedent(
        """
        import asyncio
        import copy
        import sys
        import types
        from types import SimpleNamespace
        from unittest.mock import patch

        checkpoint_service = types.ModuleType("relax.distributed.checkpoint_service")
        checkpoint_service.__path__ = []
        checkpoint_client = types.ModuleType("relax.distributed.checkpoint_service.client")
        checkpoint_client.__path__ = []
        checkpoint_engine = types.ModuleType("relax.distributed.checkpoint_service.client.engine")

        def unused_create_client(*_args, **_kwargs):
            raise AssertionError("group-RM unit test unexpectedly constructed a DCS client")

        checkpoint_engine.create_client = unused_create_client
        sys.modules[checkpoint_service.__name__] = checkpoint_service
        sys.modules[checkpoint_client.__name__] = checkpoint_client
        sys.modules[checkpoint_engine.__name__] = checkpoint_engine

        from relax.engine.rollout import sglang_rollout
        from relax.utils.types import Sample


        def completed_sample():
            return SimpleNamespace(
                status=Sample.Status.COMPLETED,
                response="done",
                response_length=1,
                reward=0.5,
                loss_mask=[1],
                session_id=None,
                metadata={},
            )


        async def exercise_surplus_requeue():
            group = [completed_sample(), completed_sample()]
            counters = {"generation": 0, "reward": 0, "opd": 0}

            class OpdManager:
                async def prefill(self, _samples, _encoder):
                    counters["opd"] += 1

            state = SimpleNamespace(aborted=False, opd_manager=OpdManager())

            async def fake_generate_and_rm(_args, sample, _sampling_params, **_kwargs):
                counters["generation"] += 1
                return sample

            async def fake_group_rm(_args, samples):
                counters["reward"] += 1
                return [float(counters["reward"])] * len(samples)

            args = SimpleNamespace(
                enable_cross_version_kv_continuation=True,
                group_rm=True,
                sglang_enable_deterministic_inference=False,
            )
            with (
                patch.object(sglang_rollout, "GenerateState", return_value=state),
                patch.object(sglang_rollout, "generate_and_rm", side_effect=fake_generate_and_rm),
                patch.object(sglang_rollout, "batched_async_rm", side_effect=fake_group_rm),
            ):
                first_result = await sglang_rollout.generate_and_rm_group(args, group, sampling_params={})
                # Ray serializes the surplus group into and out of the remote
                # data buffer before the next physical rollout.
                requeued_group = copy.deepcopy(first_result)
                second_result = await sglang_rollout.generate_and_rm_group(args, requeued_group, sampling_params={})

            assert second_result == requeued_group
            assert counters == {"generation": 0, "reward": 1, "opd": 1}
            assert [sample.reward for sample in group] == [1.0, 1.0]
            assert all(sample.metadata["_cross_version_kv_group_rm_finalized"] for sample in group)
            assert all(sample.metadata["_cross_version_kv_group_opd_finalized"] for sample in group)

            # RM and OPD have independent terminal markers. If OPD fails after
            # RM succeeds, retry only OPD; a later surplus requeue repeats
            # neither side effect.
            retry_group = [completed_sample(), completed_sample()]
            retry_counters = {"generation": 0, "reward": 0, "opd": 0}

            async def retry_generate(_args, sample, _sampling_params, **_kwargs):
                retry_counters["generation"] += 1
                return sample

            async def retry_group_rm(_args, samples):
                retry_counters["reward"] += 1
                return [2.0] * len(samples)

            class FlakyOpdManager:
                async def prefill(self, _samples, _encoder):
                    retry_counters["opd"] += 1
                    if retry_counters["opd"] == 1:
                        raise RuntimeError("teacher response lost")

            retry_state = SimpleNamespace(aborted=False, opd_manager=FlakyOpdManager())
            with (
                patch.object(sglang_rollout, "GenerateState", return_value=retry_state),
                patch.object(sglang_rollout, "generate_and_rm", side_effect=retry_generate),
                patch.object(sglang_rollout, "batched_async_rm", side_effect=retry_group_rm),
            ):
                try:
                    await sglang_rollout.generate_and_rm_group(args, retry_group, sampling_params={})
                except RuntimeError as exc:
                    assert str(exc) == "teacher response lost"
                else:
                    raise AssertionError("first OPD prefill unexpectedly succeeded")

                assert retry_counters == {"generation": 0, "reward": 1, "opd": 1}
                assert all(sample.metadata["_cross_version_kv_group_rm_finalized"] for sample in retry_group)
                assert all(
                    "_cross_version_kv_group_opd_finalized" not in sample.metadata for sample in retry_group
                )

                retried_group = await sglang_rollout.generate_and_rm_group(args, retry_group, sampling_params={})
                final_group = await sglang_rollout.generate_and_rm_group(
                    args, copy.deepcopy(retried_group), sampling_params={}
                )

            assert retry_counters == {"generation": 0, "reward": 1, "opd": 2}
            assert all(sample.metadata["_cross_version_kv_group_opd_finalized"] for sample in final_group)


        asyncio.run(exercise_surplus_requeue())
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)
