# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Format-aware reward router.

Resolves the reward type for a single sample by inspecting its metadata
and applying a configurable fallback chain.  Supports mixed-task batches
where different samples may require different reward functions.
"""

from relax.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Metadata keys inspected in priority order when resolving the reward type.
_DEFAULT_METADATA_KEYS = ("rm_type", "task_type", "label_type")


def resolve_rm_type(
    sample,
    default_rm_type: str | None = None,
    metadata_keys: tuple[str, ...] = _DEFAULT_METADATA_KEYS,
) -> str | None:
    """Resolve the reward type for *sample*.

    Resolution order (first non-empty value wins):

    1. ``sample.metadata["rm_type"]``
    2. ``sample.metadata["task_type"]``
    3. ``sample.metadata["label_type"]``
    4. *default_rm_type* (typically the CLI ``--rm-type`` value)

    When multiple metadata keys disagree a warning is logged and the
    highest-priority value is used.  When no type can be determined a
    warning is logged and ``None`` is returned.
    """
    metadata = getattr(sample, "metadata", None) or {}
    if not isinstance(metadata, dict):
        metadata = {}

    candidates: list[str] = []
    seen: set[str] = set()

    for key in metadata_keys:
        value = metadata.get(key)
        if value and isinstance(value, str):
            value = value.strip()
            if value and value not in seen:
                candidates.append(value)
                seen.add(value)

    # Append the CLI default (lowest priority).
    if default_rm_type:
        default_rm_type = default_rm_type.strip()
        if default_rm_type and default_rm_type not in seen:
            candidates.append(default_rm_type)

    if not candidates:
        logger.warning(
            "RewardRouter: could not resolve reward type – "
            "metadata=%s, default_rm_type=%r. Falling back to None (zero reward).",
            {k: metadata.get(k) for k in metadata_keys if k in metadata},
            default_rm_type,
        )
        return None

    if len(candidates) > 1:
        logger.warning(
            "RewardRouter: conflicting reward type hints %s – "
            "using highest-priority value %r.",
            candidates,
            candidates[0],
        )

    return candidates[0]
