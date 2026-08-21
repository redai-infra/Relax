# Task 21 60-step training benchmark

This directory contains the complete 60-step B60 and P+R60 benchmark. Both
runs use seed `20260820`, benchmark commit
`e2be8cd158609cc2dfae72b7ba92df72cacb3091`, the Qwen3-VL-8B-Instruct /
OpenR1-Multimodal workload, and the five-GPU allocation `0,1,2,4,7`.

| Run | ProcessorPool | Train-forward log-prob reuse | Pipeline forward | Pipeline overlap | Steps | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B60 | 0 | 0 | 0 | 0 | 60 | passed |
| P+R60 | 8 | 1 | 0 | 0 | 60 | passed |

The P+R60 attribution is `MM_PROCESSOR_POOL_SIZE=8`,
`HYBRID_REUSE_TRAIN_LOGPROBS=1`, `HYBRID_PIPELINE_FORWARD=0`, and
`HYBRID_PIPELINE_OVERLAP=0`.

| Run | Step token throughput | Mean step time | Response-token throughput | Raw reward mean | Grad-norm max |
| --- | ---: | ---: | ---: | ---: | ---: |
| B60 | 964.677824 token/s | 203.618466 s | 484.372473 token/s | 0.44359375 | 4.8916488 |
| P+R60 | 1475.172210 token/s | 131.714886 s | 734.883529 token/s | 0.438046875 | 3.0623481 |

P+R60 is `1.529x` B60 for step-token throughput and reduces mean step time by
`35.31%`. Steps `0-9` are warmup; the analyzer uses steps `10-59` for its 50
steady-state measurements. The CSV preserves every scalar for steps `0-59`
without smoothing or interpolation.

Both runs have `training_exit_status=0`, `validation_exit_status=0`, and
`exit_status=0`. All 60 values for `train/grad_norm`, `train/loss`, and
`rollout/raw_reward` are finite. The analyzer validation is `passed` with
`measurement_scope=steady_state`.

Run IDs:

- B60:
  `CURVE60-BR-e2be8cd-20260821-1120-GPU01247-B60-GPU01247-seed20260820-attempt1`
- P+R60:
  `CURVE60-BR-e2be8cd-20260821-1120-GPU01247-R60-GPU01247-seed20260820-attempt1`
