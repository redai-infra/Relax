#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-35B-A3B 16xXPU (2-node × 8 昆仑 XPU) fully sync training script
# for DAPO math dataset.
#
# 继承自:
#   - run-qwen3-4b-xpu.sh   （昆仑 XPU 运行时环境 / BKCL / XCCL / SGLang Kunlun 后端）
#   - run-qwen35-35B-A3B-16xgpu.sh （35B 切分、ckpt、cpu-offload 优化器、partial rollout）
#
# Usage:
#   bash scripts/training/text/run-qwen35-35B-A3B-16xP800.sh
#
# Prereq：两机 Ray 集群已拉起（10.0.0.25 head + 10.0.0.44 worker，各 8 XPU）

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

# ── XPU 运行时开关（host 侧也要设；worker 进程通过 RUNTIME_ENV_JSON 再覆盖一次）
export CUDA_ENABLE_P2P_NO_UVA=0
export CUDA_FAKE_UVA_ENABLE=1
export CUDA_ERROR_LEVEL=0
export XPU_SUPPORT_IPC_EVENT=1

# ── 路径与项目名
export WORKDIR=/workspace/
export CONDA_PREFIX=/root/miniconda/envs/python310_torch29_cuda
export MODEL_DIR=/workspace                              # 模型权重 / 训练样本的共同父目录
export PROJECT_NAME=${PROJECT_NAME:=Relax-Qwen3.5-35B-A3B-P800}
export MODEL_CONFIG_DIR=${MODEL_DIR}/Rrelax/scripts/models
export MEGATRON=${MODEL_DIR}/Megatron-LM
export NUM_GPUS=16
export WANDB_API_KEY=${WANDB_API_KEY:=YOUR-KEY}

unset http_proxy
unset https_proxy

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# 外部 entrypoint 没注入时才自己 source local.sh（双机场景一般由 ray-job.sh 注入）
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "/workspace/Relax/scripts/entrypoint/local-qwen35-35B-A3B-16xP800.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

EXP_DIR="${MODEL_DIR:=${SCRIPT_DIR}/../../../../exps}"
NUM_ROLLOUT="${NUM_ROLLOUT:=1000}"

CKPT_ARGS=(
   --hf-checkpoint ${EXP_DIR}/Qwen3.5-35B-A3B/
   --ref-load ${EXP_DIR}/Qwen3.5-35B-A3B/
   --megatron-to-hf-mode bridge

   # 首次运行 mcore ckpt 不存在，走 bridge 从 HF 现场转；保存路径保持一致以便后续加载
   --load ${EXP_DIR}/Qwen3.5-35B-A3B_mcore_16xgpu/
   --save ${EXP_DIR}/Qwen3.5-35B-A3B_mcore_16xgpu/
   --save-interval 100
   --max-actor-ckpt-to-keep 1
)

PROMPT_SET=${EXP_DIR}/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 128
   --use-fault-tolerance
   # --max-global-restart 1
   --balance-data
)

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 20
   --eval-prompt-data aime ${EXP_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

# 16 卡双机切分：TP=4, PP=2, EP=8, ETP=1, CP=1 → 刚好占满
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
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

   # 35B 显存紧，必须保留
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   # NOTE(wuhuan): to avoid algorithm performance degradation
   --no-rope-fusion
   --moe-router-load-balancing-type "none"
   --moe-aux-loss-coeff 0.0
)

# 昆仑 XPU 专用 SGLang 参数 + 35B 推理 TP=2
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-disable-custom-all-reduce
   --sglang-page-size 64
   --sglang-attention-backend kunlun
   --sglang-disable-radix-cache
   --sglang-max-running-requests 32
   --sglang-disable-cuda-graph
   # Kunlun XPU: torch_memory_saver 不可用，改用 scheduler_update_weights_mixin 中的
   # manual offload fallback (model.to('cpu')) 释放 SGLang 显存供训练使用
   # --no-offload-rollout
)

WANDB_ARGS=(
   # --use-clearml
   # --use-metrics-service
   # --tb-project-name  ${PROJECT_NAME}
   # --tb-experiment-name qwen35-35B-A3B-16xpu-sync-${now}
   --use-wandb
   --wandb-mode ${WANDB_MODE}
   --wandb-project relax—qwen3.5-35B
   --wandb-group qwen3.5-p800-35B
   --wandb-key ${WANDB_API_KEY}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   # --disable-bf16-reduced-precision-matmul
)

PARTIAL_ROLLOUT_ARGS=(
   # --partial-rollout
   # --over-sampling-batch-size 48
   # --mask-offpolicy-in-partial-rollout
   # --partial-rollout-max-aborted-count 3
)

# 昆仑 XPU 运行时环境（与 run-qwen3-4b-xpu.sh 对齐）
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${WORKDIR}/Megatron-Bridge/src:${WORKDIR}/TransferQueue:${WORKDIR}/Megatron-LM/:${WORKDIR}/Relax\",
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
    \"CUDA_ENABLE_P2P_NO_UVA\": \"0\",
    \"CUDA_FAKE_UVA_ENABLE\": \"1\",
    \"CUDART_DUMMY_REGISTER\": \"1\",
    \"XPU_FORCE_USERMODE_LAUNCH\": \"1\",
    \"XMLIR_DIST_SINGLETON_STREAM\": \"true\",
    \"CUDA_VISIBLE_DEVICES\": \"0,1,2,3,4,5,6,7\",
    \"XPU_VISIBLE_DEVICES\": \"0,1,2,3,4,5,6,7\",
    \"XMLIR_FA_GEMM_TYPE\": \"float\",
    \"XBLAS_FC_HBM_VERSION\": \"40\",
    \"XMLIR_ENABLE_FAST_FC\": \"0\",
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
    \"BKCL_TREE_THRESHOLD\": \"1\",
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
    \"BKCL_RDMA_NICS\": \"mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7\",
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
    \"XMLIR_USE_HYDRA_LINEAR\": \"0\",
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
    \"FORCE_NN_LINEAR\": \"1\",
    \"USE_FUSED_GATED_DELTA_RULE\": \"1\",
    \"XSGL_TRANSPOSE_SSM_STATE\": \"1\",
    \"XSGL_TRANSPOSE_CONV_STATE\": \"1\",
    \"XSGL_FUSE_SPLIT_NORM_ROPE_NEOX\": \"1\",
    \"XSGL_MOE_UNSTABLE_TOPK\": \"1\",
    \"XPU_FLASH_ATTENTION_DECODER_USE_BALANCE\": \"1\",
    \"XMLIR_FORCE_USE_XPU_GRAPH\": \"1\",
    \"RAY_OVERRIDE_JOB_RUNTIME_ENV\":\"1\",
    \"XMLIR_MEMCPY_RETRY_SYNC\": \"true\",
    \"RELAX_SKIP_TORCH_MEMORY_SAVER\":\"1\"
  }
}"

# ── 自动拉起 Ray 集群（如果尚未运行）
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! ray status &>/dev/null; then
  echo "=== Ray cluster not running, starting via local.sh ==="
  export HEAD_IP=${HEAD_IP:-10.10.1.100}
  export WORKER_NODES=${WORKER_NODES:-10.10.0.52}
  source "${_SCRIPT_DIR}/local-qwen35-35B-A3B-16xP800.sh"
fi

# Ray dashboard：我们总是在 head 容器内提交，因此直连 127.0.0.1 即可；
# 如你从 worker 提交，请 export HOST_IP=10.0.0.25 再运行。
HOST_IP=${HOST_IP:-127.0.0.1}

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HOST_IP}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 16], "rollout": [1, 16]}' \
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
   "${PARTIAL_ROLLOUT_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen35-35B-A3B-GRPO-xpu16-sync-${now}.log