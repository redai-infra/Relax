# Copyright (c) 2026 Relax Authors. All Rights Reserved.


def get_memory_preflight_error(
    *,
    rank: int,
    target: str,
    free_bytes: int,
    total_bytes: int,
    target_bytes: int,
    bucket_bytes: int,
    margin_bytes: int,
) -> str | None:
    required_bytes = target_bytes + bucket_bytes + margin_bytes
    if free_bytes >= required_bytes:
        return None
    return (
        f"rank={rank} target={target}, "
        f"free={free_bytes / 1024**3:.2f} GiB, "
        f"target_weights={target_bytes / 1024**3:.2f} GiB, "
        f"conversion_bucket={bucket_bytes / 1024**3:.2f} GiB, "
        f"margin={margin_bytes / 1024**3:.2f} GiB, "
        f"total_gpu={total_bytes / 1024**3:.2f} GiB"
    )


def validate_export_response(response: dict, names: list[str], expected_version: int | str) -> tuple[list, str]:
    if not response.get("success", False):
        raise RuntimeError(f"SGLang weight export failed: {response.get('message', 'unknown error')}")

    actual_version = response.get("weight_version")
    if str(actual_version) != str(expected_version):
        raise RuntimeError(f"SGLang weight version mismatch: expected {expected_version}, got {actual_version}")

    metadata = response.get("metadata")
    if not isinstance(metadata, list) or len(metadata) != len(names):
        raise RuntimeError(
            f"SGLang export metadata count mismatch: expected {len(names)}, "
            f"got {0 if metadata is None else len(metadata)}"
        )
    if [item.get("name") for item in metadata] != names:
        raise RuntimeError("SGLang export metadata names do not match the requested Bridge weights")

    serialized = response.get("serialized_named_tensors")
    if not serialized:
        raise RuntimeError("SGLang export returned no serialized CUDA tensors")
    return metadata, serialized
