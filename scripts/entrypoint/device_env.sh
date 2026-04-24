#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Shared GPU environment helpers for both CUDA and ROCm deployments.

relax_gpu_platform() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "cuda"
        return 0
    fi

    if command -v rocm-smi >/dev/null 2>&1; then
        echo "rocm"
        return 0
    fi

    echo "unknown"
}


relax_primary_visible_devices_env() {
    local platform
    platform="$(relax_gpu_platform)"

    local var_name
    if [ "${platform}" = "rocm" ]; then
        # On the ROCm PyTorch image used for MI355 validation, torch only
        # enumerates devices correctly when CUDA_VISIBLE_DEVICES is used.
        for var_name in CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; do
            if [ -n "${!var_name:-}" ]; then
                echo "${var_name}"
                return 0
            fi
        done
        echo "CUDA_VISIBLE_DEVICES"
        return 0
    fi

    for var_name in CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; do
        if [ -n "${!var_name:-}" ]; then
            echo "${var_name}"
            return 0
        fi
    done
    echo "CUDA_VISIBLE_DEVICES"
}


relax_visible_devices() {
    local var_name
    var_name="$(relax_primary_visible_devices_env)"
    printf '%s\n' "${!var_name:-}"
}


relax_export_visible_devices() {
    local gpu_ids="$1"
    local platform
    platform="$(relax_gpu_platform)"

    if [ "${platform}" = "rocm" ]; then
        export CUDA_VISIBLE_DEVICES="${gpu_ids}"
        unset ROCR_VISIBLE_DEVICES
        unset HIP_VISIBLE_DEVICES
        return 0
    fi

    export CUDA_VISIBLE_DEVICES="${gpu_ids}"
    unset ROCR_VISIBLE_DEVICES
    unset HIP_VISIBLE_DEVICES
}


relax_gpu_count_from_env_or_default() {
    local default_count="${1:-8}"
    local visible
    visible="$(relax_visible_devices)"

    if [ -n "${visible}" ]; then
        echo "${visible}" | tr ',' '\n' | grep -c '[0-9]'
    else
        echo "${default_count}"
    fi
}


relax_detect_fast_interconnect() {
    local links

    if command -v nvidia-smi >/dev/null 2>&1; then
        links="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)"
        [ "${links:-0}" -gt 0 ] && echo 1 || echo 0
        return 0
    fi

    if command -v rocm-smi >/dev/null 2>&1; then
        links="$(rocm-smi --showtopotype --csv 2>/dev/null | grep -Eci 'XGMI|MGMI')"
        [ "${links:-0}" -gt 0 ] && echo 1 || echo 0
        return 0
    fi

    echo 0
}


relax_select_top_gpus_by_free_mem() {
    local count="${1:-1}"

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | sort -t, -k2 -rn \
            | head -n "${count}" \
            | cut -d, -f1 \
            | paste -sd ','
        return 0
    fi

    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showmeminfo vram --csv 2>/dev/null \
            | awk -F, '$1 ~ /^card[0-9]+$/ {gsub("card", "", $1); free = $2 - $3; print $1 "," free}' \
            | sort -t, -k2 -gr \
            | head -n "${count}" \
            | cut -d, -f1 \
            | paste -sd ','
        return 0
    fi

    return 1
}
