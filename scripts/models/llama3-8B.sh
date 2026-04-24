# Copyright (c) 2026 Relax Authors. All Rights Reserved.

MODEL_ARGS=(
   --swiglu
   --num-layers 32
   --hidden-size 4096
   --ffn-hidden-size 14336
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-5
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-500000}"
   --vocab-size 128256
   --kv-channels 128
   --untie-embeddings-and-output-weights
   --seq-length "${MODEL_ARGS_SEQ_LENGTH:-8192}"
)
