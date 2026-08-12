#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B 8xP800 colocate (sync) training script for DAPO math dataset.
#
# Usage:
#   bash scripts/training/sft/run-qwen35-9B-8xklx.sh

set -ex
set -o pipefail

gpus_num=8
export WORLD_SIZE=$(( gpus_num / 8 ))

now=$(date "+%Y-%m-%d-%H:%M:%S")

export WORKDIR="${WORKDIR:-/workspace}"
export MODEL_DIR="${MODEL_DIR:-/workspace}"
export DATA_DIR="${DATA_DIR:-/workspace}"
export EXP_DIR="${EXP_DIR:-/workspace}"
export WANDB_API_KEY="${WANDB_API_KEY:=YOUR-KEY}"
export WEB_PROXY="${WEB_PROXY:-}"

export XMLIR_USE_HYDRA_LINEAR=1
export XMLIR_ENABLE_FAST_FC=1

export RELAX_SKIP_TORCH_MEMORY_SAVER=1
export XMLIR_MEMCPY_RETRY_SYNC=true
export CUDA_ENABLE_P2P_NO_UVA=0
export CUDA_FAKE_UVA_ENABLE=1
export CUDA_ERROR_LEVEL=0
export XPU_SUPPORT_IPC_EVENT=1
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-"eth0"}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-"eth0"}
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-"eth1,eth2,eth3,eth4"}

exp_target="${EXP_TARGET:-}"

ROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/llm_video_feed_sample_di_v2/sft/train.parquet}"
PROJECT_NAME="${PROJECT_NAME:="XHS_Relax"}"
EXP_NAME="Qwen3.5-9B-MM-SFT-${gpus_num}xP800${exp_target}"

unset http_proxy
unset https_proxy

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-klx.sh"
fi
source "${SCRIPT_DIR}/../../models/qwen35-9B.sh"

SYSTEM_PROMPT="$(cat <<'RELAX_SYS_EOF'
角色：你是一名资深广告效果预测专家，专注于短视频信息流广告领域。

任务：根据用户特征和广告特征，判断该用户在刷视频流时是否会对这条视频广告点赞。

规则：只能输出 yes 或 no，不得输出任何其他内容。
RELAX_SYS_EOF
)"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-9B
   --ref-load ${MODEL_DIR}/Qwen3.5-9B
   --megatron-to-hf-mode bridge
   --load ${EXP_DIR}/sft/checkpoint/Qwen3-9B_mcore_${gpus_num}xklx/
   --save ${EXP_DIR}/sft/checkpoint/Qwen3-9B_mcore_${gpus_num}xklx/
   --save-interval 500
   --max-actor-ckpt-to-keep 1
   --num-epoch 1
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data ${ROMPT_DATA}
   --input-key ${INPUT_KEY:-input}
   --label-key ${LABEL_KEY:-label}
   --global-batch-size 512
   --use-dynamic-batch-size
   # --max-tokens-per-gpu 20480
   --max-tokens-per-gpu 10240
   --balance-data
   --system-prompt "${SYSTEM_PROMPT}"
   --sft-prefetch-buffer-size 512
   --sft-prefetch-num-workers 8
   --use-distributed-optimizer
   --overlap-grad-reduce
   --overlap-param-gather
   --cross-entropy-fusion-impl te
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --calculate-per-token-loss

   # --recompute-granularity full
   # --recompute-method block
   # --recompute-num-layers 8

   --no-rope-fusion
   --colocate
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-05
   --lr-decay-style cosine
   --min-lr 1e-06
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0

   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
   # --use-precision-aware-optimizer
)

EVAL_ARGS=(
   --eval-size 0.01
   --eval-interval 200
)

PREDICT_ARGS=()

WANDB_ARGS=(
   --tb-experiment-name ${EXP_NAME}-${now}
   --no-use-wandb
   --wandb-project ${PROJECT_NAME}
   --wandb-group ${EXP_NAME}-${now}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
   --no-use-metrics-service
   --no-use-tensorboard
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-health-check
)

# infer cpu threads num
NUM_GPUS_TOTAL="${NUM_GPUS_TOTAL:-8}"
if [ -z "${CPU_THREADS_PER_ACTOR:-}" ]; then
    _cores_per_socket=$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket:/ {gsub(/ /,"",$2); print $2; exit}')
    _sockets=$(lscpu 2>/dev/null | awk -F: '/^Socket\(s\):/ {gsub(/ /,"",$2); print $2; exit}')
    if [ -n "${_cores_per_socket}" ] && [ -n "${_sockets}" ] && [ "${_sockets}" -gt 0 ]; then
        _total_phys=$((_cores_per_socket * _sockets))
        CPU_THREADS_PER_ACTOR=$((_total_phys / NUM_GPUS_TOTAL))
        # clamp to [4, 64], avoid bad values
        [ "${CPU_THREADS_PER_ACTOR}" -lt 4 ] && CPU_THREADS_PER_ACTOR=4
        [ "${CPU_THREADS_PER_ACTOR}" -gt 64 ] && CPU_THREADS_PER_ACTOR=64
    else
        CPU_THREADS_PER_ACTOR=24
    fi
fi
echo "[cpu-threads] NUM_GPUS_TOTAL=${NUM_GPUS_TOTAL} CPU_THREADS_PER_ACTOR=${CPU_THREADS_PER_ACTOR}"
export CPU_THREADS_PER_ACTOR

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${WORKDIR}/TransferQueue:${WORKDIR}/Megatron-LM/:${SCRIPT_DIR}:${WORKDIR}/Megatron-Bridge/src/:$PYTHONPATH\",
    \"LD_LIBRARY_PATH\":\"${CONDA_PREFIX}/xcudart/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/xtorch_ops:${CONDA_PREFIX}/lib/python3.10/site-packages/torch_xmlir/:${CONDA_PREFIX}/lib/python3.10/site-packages/torch_xmlir/xre/so\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"TOKENIZERS_PARALLELISM\": \"true\",
    \"NCCL_CUMEM_ENABLE\": \"0\",
    \"NCCL_SOCKET_IFNAME\": \"eth0\",
    \"GLOO_SOCKET_IFNAME\": \"eth0\",
    \"NCCL_IB_HCA\": \"mlx5\",
    \"NCCL_IB_GID_INDEX\": \"3\",
    \"CUDA_DEVICE_ORDER\": \"OAM_ID\",
    \"CUDA_ENABLE_P2P_NO_UVA\": \"0\",
    \"CUDA_FAKE_UVA_ENABLE\": \"1\",
    \"CUDART_DUMMY_REGISTER\": \"1\",
    \"XPU_FORCE_USERMODE_LAUNCH\": \"1\",
    \"XMLIR_DIST_SINGLETON_STREAM\": \"true\",
    \"CUDA_VISIBLE_DEVICES\": \"0,1,2,3,4,5,6,7\",
    \"XPU_VISIBLE_DEVICES\": \"0,1,2,3,4,5,6,7\",
    \"XMLIR_FA_GEMM_TYPE\": \"float\",
    \"XBLAS_FC_HBM_VERSION\": \"40\",
    \"XMLIR_ENABLE_FAST_FC\": \"1\",
    \"XMLIR_USE_HYDRA_LINEAR\": \"1\",
    \"XTE_DISABLE_MOE_DW_FUSION\": \"0\",
    \"XTE_RECOMPUTE_LN_OUT_TOTAL\": \"1\",
    \"XMLIR_PARALLEL_SAVE_MEMORY\": \"false\",
    \"XMLIR_DISABLE_CUDA_ALLOCATOR\": \"false\",
    \"XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL\": \"0\",
    \"XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL\": \"False\",
    \"XMLIR_DUMP_FALLBACK_OP_LIST_BOOL\": \"true\",
    \"XMLIR_DIST_ASYNC_ISEND_IRECV\": \"false\",
    \"XMLIR_BATCH_PARALLEL\": \"false\",
    \"XPU_FORCE_SHARED_DEVICE_CONTEXT\": \"1\",
    \"BKCL_RDMA_PROXY_DISABLE\": \"1\",
    \"BKCL_USE_AR\": \"1\",
    \"BKCL_RING_OPT\": \"1\",
    \"BKCL_FLAT_RING\": \"1\",
    \"BKCL_CCIX_RING\": \"1\",
    \"BKCL_TREE_THRESHOLD\": \"1048576\",
    \"BKCL_CCIX_BUFFER_GM\": \"1\",
    \"BKCL_FORCE_L3_RDMA\": \"0\",
    \"BKCL_RING_BUFFER_GM\": \"1\",
    \"BKCL_ENABLE_XDR\": \"1\",
    \"BKCL_RDMA_FORCE_TREE\": \"1\",
    \"BKCL_XLINK_D2D\": \"0\",
    \"BKCL_XLINK_ETH\": \"0\",
    \"BKCL_XLINK_C2C\": \"1\",
    \"BKCL_TRANS_UNSUPPORTED_DATATYPE\": \"1\",
    \"BKCL_KL3_TURBO_MODE\": \"1\",
    \"BKCL_RING_BUFFER_SIZE\": \"2097152\",
    \"ALLREDUCE_ASYNC\": \"false\",
    \"ALLGATHER_ASYNC\": \"false\",
    \"ALLREDUCE_FUSION\": \"0\",
    \"BKCL_TIMEOUT\": \"400000\",
    \"CUDA_DISABLE_PRINTF\": \"1\",
    \"BKCL_RDMA_VERBS\": \"1\",
    \"BKCL_RDMA_NICS\": \"${BKCL_RDMA_NICS}\",
    \"NVTE_DEBUG\": \"1\",
    \"NVTE_DEBUG_LEVEL\": \"1\",
    \"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES\": \"1\",
    \"TORCH_XCCL_DEFAUTL_PG_TIMEOUT_MILSEC\": \"7200000\",
    \"CUDA_ERROR_LEVEL\": \"0\",
    \"HYDRA_FULL_ERROR\": \"1\",
    \"XMLIR_ENABLE_NEW_PG\": \"1\",
    \"TORCH_XCCL_HEARTBEAT_TIMEOUT_SEC\": \"1800\",
    \"TORCH_XCCL_ENABLE_TIMING\": \"1\",
    \"TORCH_FR_BUFFER_SIZE\": \"2000\",
    \"TORCH_XCCL_TRACE_BUFFER_SIZE\": \"2000\",
    \"VERL_LOGGING_LEVEL\": \"DEBUG\",
    \"BKCL_ALL_TO_ALL_OPT\": \"1\",
    \"SGLANG_IS_FLASHINFER_AVAILABLE\": \"false\",
    \"USE_MOE_FC_V3\": \"1\",
    \"FLA_USE_NAIVE\": \"1\",
    \"FORCE_DISABLE_FLA\": \"1\",
    \"DISABLE_CAST_CACHE\": \"1\",
    \"FORCE_NN_LINEAR\": \"0\",
    \"XMLIR_USE_HYDRA_LINEAR\": \"1\",
    \"SGL_CPU_QUANTIZATION\": \"1\",
    \"XPU_ENABLE_CTX_LAZY_INIT\": \"1\",
    \"XPU_SUPPORT_IPC_EVENT\": \"1\",
    \"TRITON_SKIP_AUTOTUNE\": \"1\",
    \"XMLIR_FORCE_USE_XPU_GRAPH\": \"1\",
    \"XSGL_USE_TORCH_CAUSAL_CONV\": \"1\",
    \"XSGL_FUSE_SPLIT_NORM_ROPE_NEOX\": \"1\",
    \"XPU_FLASH_ATTENTION_DECODER_USE_BALANCE\": \"1\",
    \"CUDA_ENABLE_P2P_NO_UVA\": \"0\",
    \"CUDA_FAKE_UVA_ENABLE\": \"1\",
    \"XSGL_TRANSPOSE_SSM_STATE\": \"1\",
    \"XSGL_TRANSPOSE_CONV_STATE\": \"1\",
    \"USE_FUSED_GATED_DELTA_RULE\": \"1\",
    \"RAY_OVERRIDE_JOB_RUNTIME_ENV\":\"1\",
    \"XMLIR_D_XPU_L3_SIZE\": \"0\",
    \"XMLIR_MEMCPY_RETRY_SYNC\": \"true\",
    \"DEBUG_DUMP_TOKENS\": \"0\",
    \"RELAX_SKIP_TORCH_MEMORY_SAVER\":\"1\",
    \"XMLIR_MATMUL_FAST_MODE\": \"1\",
    \"XMLIR_ENABLE_FAST_FC\": \"1\",
    \"HYDRAX_USE_PROTEUS\": \"0\",
    \"http_proxy\": \"${WEB_PROXY}\",
    \"https_proxy\": \"${WEB_PROXY}\",
  }
}"

mkdir -p log

RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://127.0.0.1:8265}"

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_JOB_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, '"${gpus_num}"']}' \
   --colocate \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${PREDICT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee log/qwen35-9B-mm-sft-${gpus_num}xklx-${now}.log
