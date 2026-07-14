#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B 8xP800 colocate (sync) training script for DAPO math dataset.
#
# Usage:
#   bash scripts/training/text/run-qwen35-9B-8xklx.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

export WORKDIR="${WORKDIR:-/workspace}"
export MODEL_DIR="${MODEL_DIR:-/workspace}"
export DATA_DIR="${DATA_DIR:-/workspace}"
export PROJECT_NAME=Relax-Qwen3.5-9B-P800
export WANDB_API_KEY="${WANDB_API_KEY:=YOUR-KEY}"

export MEGATRON=${WORKDIR}/Megatron-LM

export CUDA_ENABLE_P2P_NO_UVA=0
export CUDA_FAKE_UVA_ENABLE=1
export CUDA_ERROR_LEVEL=0
export XMLIR_MEMCPY_RETRY_SYNC=true

export XPU_SUPPORT_IPC_EVENT=1
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-"eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4"}

unset http_proxy
unset https_proxy

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-klx.sh"
fi
source "${SCRIPT_DIR}/../../models/qwen35-9B.sh"

NUM_ROLLOUT="${NUM_ROLLOUT:=1000}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-9B
   --ref-load ${MODEL_DIR}/Qwen3.5-9B
   --megatron-to-hf-mode bridge

   --load ${EXP_DIR}/Qwen3-9B_mcore_8xgpu/
   --save ${EXP_DIR}/Qwen3-9B_mcore_8xgpu/
   --save-interval 50
   --max-actor-ckpt-to-keep 1
)

PROMPT_SET=${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
EVAL_DATA=${DATA_DIR}/aime-2024/aime-2024.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 512
   --balance-data
   --use-fault-tolerance
)

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 20
   --eval-prompt-data aime ${EVAL_DATA}
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method block
   --recompute-num-layers 16

   --use-dynamic-batch-size
   --max-tokens-per-gpu 24576
   # --micro-batch-size 1 # avoid OOM

   --no-rope-fusion
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)

   --sglang-disable-custom-all-reduce
   --sglang-page-size 64
   --sglang-attention-backend kunlun
   --sglang-mm-attention-backend fa3
   --sglang-disable-radix-cache
   --sglang-max-running-requests 256
   # --sglang-disable-cuda-graph
   --sglang-reasoning-parser qwen3
   --sglang-tool-call-parser qwen3_coder
)

WANDB_ARGS=(
   --tb-experiment-name qwen3.5-p800-9B-${now}
   --use-wandb
   --wandb-project ${PROJECT_NAME}
   --wandb-group qwen3.5-p800-9B-${now}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
   --no-use-metrics-service
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${WORKDIR}/TransferQueue:${WORKDIR}/Megatron-LM/:${SCRIPT_DIR}:${WORKDIR}/Megatron-Bridge/src/:$PYTHONPATH\",
    \"LD_LIBRARY_PATH\":\"${CONDA_PREFIX}/xcudart/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/xtorch_ops:${CONDA_PREFIX}/lib/python3.10/site-packages/torch_xmlir/:${CONDA_PREFIX}/lib/python3.10/site-packages/torch_xmlir/xre/so\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"OPENBLAS_NUM_THREADS\": \"64\",
    \"OMP_NUM_THREADS\": \"64\",
    \"TOKENIZERS_PARALLELISM\": \"true\",
    \"NCCL_CUMEM_ENABLE\": \"0\",
    \"NCCL_SOCKET_IFNAME\": \"eth0\",
    \"NCCL_IB_HCA\": \"mlx5\",
    \"NCCL_IB_GID_INDEX\": \"3\",
    \"CUDA_DEVICE_ORDER\": \"OAM_ID\",
    \"CUDART_DUMMY_REGISTER\": \"1\",
    \"XPU_FORCE_USERMODE_LAUNCH\": \"1\",
    \"CUDA_VISIBLE_DEVICES\": \"0,1,2,3,4,5,6,7\",
    \"XPU_VISIBLE_DEVICES\": \"0,1,2,3,4,5,6,7\",
    \"XMLIR_FA_GEMM_TYPE\": \"float\",
    \"XBLAS_FC_HBM_VERSION\": \"40\",
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
    \"HYDRAX_USE_PROTEUS\": \"0\"
  }
}"

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
   --colocate \
   --max-staleness 0 \
   --num-data-storage-units 1 \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee log/qwen35-9B-GRPO-gpu8-${now}.log
