from dataclasses import dataclass
from typing import Any

from relax.utils.misc import load_function
from relax.utils.types import Sample


@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None
    # Exact post-conversion row count transferred to the training partition.
    # This is transport metadata for data-dependent 1:N converters; keeping it
    # separate prevents orchestration from depending on monitoring key names.
    train_row_count: int | None = None


@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    if not evaluation and isinstance(output, RolloutFnTrainOutput):
        # One-way adapter for custom rollout functions written against the
        # initial expanded-batch contract. Keep the legacy metric observable,
        # but normalize its control value into the typed transport field here.
        legacy_row_count = (output.metrics or {}).get("rollout/train_batch_row_count")
        if output.train_row_count is None:
            output.train_row_count = legacy_row_count
        elif legacy_row_count is not None and output.train_row_count != legacy_row_count:
            raise ValueError("Rollout train_row_count conflicts with the legacy rollout/train_batch_row_count metric.")

    # Apply --rollout-sample-filter-path (train only). The filter sets
    # sample.remove_sample=True in-place; downstream (relax/utils/utils.py:126)
    # zeros loss_mask for those samples so they don't contribute gradient, while
    # keeping reward in GRPO group-normalization. Reloaded every call
    # (ReloadScope.IMMEDIATE per reload_utils.py:180) to support hot-swap.
    if not evaluation and isinstance(output, RolloutFnTrainOutput):
        train_args = args[0] if args else kwargs.get("args")
        filter_path = getattr(train_args, "rollout_sample_filter_path", None)
        if filter_path:
            load_function(filter_path)(train_args, output.samples)

    return output
