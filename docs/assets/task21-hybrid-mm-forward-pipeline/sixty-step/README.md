# Task 21 60-step P+R training curves

This directory records the 60-optimizer-step B and P+R training curves. Both
runs use the same fresh process, seed, workload, and five-GPU allocation.

| Run | ProcessorPool | Train-forward reuse | Chunk forward | Steps | Validation |
| --- | ---: | ---: | ---: | ---: | --- |
| B60 | 0 | 0 | 0 | 60 | passed |
| P+R60 | 8 | 1 | 0 | 60 | passed |

The analyzer's P+R attribution is `MM_PROCESSOR_POOL_SIZE=8`,
`HYBRID_REUSE_TRAIN_LOGPROBS=1`, and `HYBRID_PIPELINE_FORWARD=0`. The run was
recorded as `R60` by the queue launcher because the queue label names the reuse
ablation; the manifest controls are the canonical attribution.

Both runs have `training_exit_status=0`, `validation_exit_status=0`, and
`exit_status=0`, with 60 finite points for each of `train/grad_norm`,
`train/loss`, and `rollout/raw_reward`. The warmup-excluded summary is retained
in `sixty_step_summary.json`; `sixty_step_training_curves.csv` and the three
PNG files provide the complete curves.

Benchmark commit: `e2be8cd158609cc2dfae72b7ba92df72cacb3091`.
Campaign: `CURVE60-BR-e2be8cd-20260821-1120-GPU01247`.
