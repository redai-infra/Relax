# Task 21: paired 60-step steady-state campaign

This directory contains the superseding B/P/P+R/P+S campaign. Each condition
uses paired seeds `20260820` and `20260821`, 60 optimizer updates in a fresh
process, warmup steps `0-9`, and the declared steady window `10-59`.

| Condition | ProcessorPool | Train-forward reuse | Chunk forward | Included runs | Validation |
| --- | ---: | ---: | ---: | ---: | --- |
| B | 0 | 0 | 0 | 2 | passed |
| P | 8 | 0 | 0 | 2 | passed |
| P+R | 8 | 1 | 0 | 2 | passed |
| P+S | 8 | 0 | 1 | 2 | passed |

Benchmark commit: `e2be8cd158609cc2dfae72b7ba92df72cacb3091`.
Campaigns: `CURVE60-BR-e2be8cd-20260821-1120-GPU01247` and
`CURVE60-FULL2-e2be8cd-20260821-192422-GPU01247`.

## Result

| Condition | Steady token throughput mean (range) | Mean step time | Paired geometric-mean result |
| --- | ---: | ---: | ---: |
| B | 1,103.25 (964.68-1,241.83) token/s | 179.62 s | reference |
| P | 1,157.92 (1,131.16-1,184.69) token/s | 170.52 s | +5.77% vs B |
| P+R | 1,522.04 (1,475.17-1,568.91) token/s | 129.21 s | **+39.00% vs B**, **+31.42% vs P** |
| P+S | 1,229.04 (1,136.58-1,321.50) token/s | 162.82 s | +11.97% vs B, +5.87% vs P |

P+R improves throughput in both seeds: `+52.92%` and `+26.34%` vs B,
and `+24.52%` and `+38.70%` vs P. It is therefore the sustained-throughput
path supported by this campaign. P is not directionally consistent vs B, and
P+S is not directionally consistent vs P, so their positive geometric means
are retained as ablation results rather than promoted to the main claim.

All eight formal runs have `training_exit_status=0`,
`validation_exit_status=0`, and `exit_status=0`, with 60 finite points for each
of `train/grad_norm`, `train/loss`, and `rollout/raw_reward`. One S60 attempt was
terminated after an external GPU lease interruption (`exit_status=137`); it is
excluded, and its successful automatic retry is the formal run.

## Artifacts

- `sixty_step_training_curves.csv`: every plotted scalar point and source run.
- `task21_60step_{grad_norm,loss,reward}.png`: two-seed means and ranges; no
  smoothing or interpolation.
- `sixty_step_summary.json`: compact controls, run metrics, comparisons, and
  exclusion record.
- `sixty_step_campaign_report.json`: full strict-analyzer output.
- `*_paired_runs.csv`: reviewable per-seed comparisons.

The full raw logs, manifests, TensorBoard events, timeline JSONL, and NVML data
remain under `/data01/LWX/relax-task21/`.
