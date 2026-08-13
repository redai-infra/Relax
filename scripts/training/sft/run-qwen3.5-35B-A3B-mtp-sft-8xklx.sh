#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-35B-A3B MTP SFT on OpenMathReasoning-mini, 8xklx, ray-submit launch.
#
# Usage:
#   bash scripts/training/sft/run-qwen3.5-35B-A3B-mtp-sft-8xklx.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "Current time: $now"

export WANDB_API_KEY="${WANDB_API_KEY:=YOUR-KEY}"
export WORKDIR="${WORKDIR:-/workspace}"
export MODEL_DIR="${MODEL_DIR:-/workspace}"
export DATA_DIR="${DATA_DIR:-/workspace}"
export PROJECT_NAME=qwen3.5-35B-sft
export PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet}"
LOAD_DIR="${LOAD_DIR:-${MODEL_DIR}/Qwen3.5-35B-A3B/}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/checkpoints/qwen3.5-35B-A3B-mtp-sft}"
EXP_NAME="${EXP_NAME:-qwen3.5-35b-a3b-mtp-sft-8xklx}"

export MEGATRON=${WORKDIR}/Megatron-LM
 
export XMLIR_USE_HYDRA_LINEAR=1
export XMLIR_ENABLE_FAST_FC=1
export XTE_DISABLE_MOE_DW_FUSION=0
 
export RELAX_SKIP_TORCH_MEMORY_SAVER=1
export XMLIR_MEMCPY_RETRY_SYNC=true
export CUDA_ENABLE_P2P_NO_UVA=0
export CUDA_FAKE_UVA_ENABLE=1
export CUDA_ERROR_LEVEL=0
export XPU_SUPPORT_IPC_EVENT=1
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-"eth0"}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-"eth0"}
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-"bond0,bond1,bond2,bond3,bond4,bond5,bond6,bond7"}

unset http_proxy
unset https_proxy

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-klx.sh"
fi
source "${SCRIPT_DIR}/../../models/qwen35-35B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-35B-A3B/
   --ref-load ${MODEL_DIR}/Qwen3.5-35B-A3B/
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache

   --load ${LOAD_DIR}
   --save ${SAVE_DIR}/${EXP_NAME}
   --save-interval ${SAVE_INTERVAL:-100}
   --max-actor-ckpt-to-keep 1
   --num-epoch ${NUM_EPOCH:-1}
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key ${INPUT_KEY:-problem}
   --label-key ${LABEL_KEY:-generated_solution}
   --global-batch-size ${GLOBAL_BATCH_SIZE:-128}
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-18432}
   --balance-data
   # --cross-entropy-fusion-impl te
   # --cross-entropy-loss-fusion
)

MTP_ARGS=(
   --mtp-num-layers ${MTP_NUM_LAYERS:-1}
   --enable-mtp-training
   --mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR:-0.2}
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE:-4}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE:-1}
   --context-parallel-size ${CP_SIZE:-1}
   --expert-model-parallel-size ${EP_SIZE:-4}
   --expert-tensor-parallel-size ${ETP_SIZE:-1}

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --calculate-per-token-loss

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   --moe-grouped-gemm true
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR:-1e-5}
   --lr-decay-style cosine
   --min-lr ${MIN_LR:-1e-6}
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   --no-rope-fusion
   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0.0
)

WANDB_ARGS=(
   --tb-experiment-name qwen3.5-35B-klx-${now}
   --use-wandb
   --wandb-project ${PROJECT_NAME}
   --wandb-group qwen3.5-35B-SFT-8xklx
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
   --no-use-metrics-service
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-health-check
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${WORKDIR}/TransferQueue:${WORKDIR}/Megatron-LM/:${SCRIPT_DIR}:${WORKDIR}/Megatron-Bridge/src/:$PYTHONPATH\",
    \"LD_LIBRARY_PATH\":\"${CONDA_PREFIX}/xcudart/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/xtorch_ops:${CONDA_PREFIX}/lib/python3.10/site-packages/torch_xmlir/:${CONDA_PREFIX}/lib/python3.10/site-packages/torch_xmlir/xre/so\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"OPENBLAS_NUM_THREADS\": \"${CPU_THREADS_PER_ACTOR}\",
    \"OMP_NUM_THREADS\": \"${CPU_THREADS_PER_ACTOR}\",
    \"MKL_NUM_THREADS\": \"${CPU_THREADS_PER_ACTOR}\",
    \"NUMEXPR_NUM_THREADS\": \"${CPU_THREADS_PER_ACTOR}\",
    \"TOKENIZERS_PARALLELISM\": \"true\",
    \"NCCL_CUMEM_ENABLE\": \"0\",
    \"NCCL_SOCKET_IFNAME\": \"eth0\",
    \"NCCL_IB_HCA\": \"mlx5\",
    \"NCCL_IB_GID_INDEX\": \"3\",
    \"CUDA_DEVICE_ORDER\": \"OAM_ID\",
    \"CUDA_ENABLE_P2P_NO_UVA\": \"0\",
    \"CUDA_FAKE_UVA_ENABLE\": \"1\",
    \"CUDART_DUMMY_REGISTER\": \"1\",
    \"XPU_FORCE_USERMODE_LAUNCH\": \"1\",
    \"CUDA_VISIBLE_DEVICES\": \"0,1,2,3,4,5,6,7\",
    \"XPU_VISIBLE_DEVICES\": \"0,1,2,3,4,5,6,7\",
    \"XMLIR_FA_GEMM_TYPE\": \"float\",
    \"XBLAS_FC_HBM_VERSION\": \"40\",
    \"XMLIR_USE_HYDRA_LINEAR\": \"${XMLIR_USE_HYDRA_LINEAR}\",
    \"XTE_DISABLE_MOE_DW_FUSION\": \"${XTE_DISABLE_MOE_DW_FUSION}\",
    \"XMLIR_ENABLE_FAST_FC\": \"${XMLIR_ENABLE_FAST_FC}\",
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
    \"XMLIR_DIST_SINGLETON_STREAM\": \"1\",
    \"SGL_CPU_QUANTIZATION\": \"0\",
    \"XSGL_ENABLE_MEM_SAVER\": \"0\",
    \"XPU_ENABLE_CTX_LAZY_INIT\": \"1\",
    \"XPU_SUPPORT_IPC_EVENT\": \"1\",
    \"XSGL_USE_TORCH_CAUSAL_CONV\": \"1\",
    \"TRACE_WEIGHT_PATHS\": \"0\",
    \"TRITON_SKIP_AUTOTUNE\": \"1\",
    \"FLA_USE_NAIVE\": \"1\",
    \"FORCE_DISABLE_FLA\": \"1\",
    \"DUMP_CONVERTED_WEIGHTS_DIR\": \"\",
    \"DISABLE_CAST_CACHE\": \"1\",
    \"FORCE_NN_LINEAR\": \"0\",
    \"USE_FUSED_GATED_DELTA_RULE\": \"1\",
    \"XSGL_TRANSPOSE_SSM_STATE\": \"1\",
    \"XSGL_TRANSPOSE_CONV_STATE\": \"1\",
    \"XSGL_FUSE_SPLIT_NORM_ROPE_NEOX\": \"1\",
    \"XSGL_MOE_UNSTABLE_TOPK\": \"1\",
    \"XPU_FLASH_ATTENTION_DECODER_USE_BALANCE\": \"1\",
    \"XMLIR_FORCE_USE_XPU_GRAPH\": \"1\",
    \"RAY_OVERRIDE_JOB_RUNTIME_ENV\":\"1\",
    \"XMLIR_D_XPU_L3_SIZE\": \"0\",
    \"RELAX_SKIP_TORCH_MEMORY_SAVER\": \"1\",
    \"XMLIR_MATMUL_FAST_MODE\": \"1\",
    \"XMLIR_MEMCPY_RETRY_SYNC\": \"${XMLIR_MEMCPY_RETRY_SYNC}\",
    \"HYDRAX_USE_PROTEUS\": \"0\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\",
    \"TP_SOCKET_IFNAME\": \"${TP_SOCKET_IFNAME}\",
    \"NVTE_DEBUG\": \"1\",
    \"NVTE_DEBUG_LEVEL\": \"1\",
    \"HEALTH_GENERATE_TOPK\": \"-1\"
  }
}"

mkdir -p log

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HOST_IP:-127.0.0.1}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, 8]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${MTP_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3.5-35b-a3b-mtp-sft-8xp800-${now}.log
