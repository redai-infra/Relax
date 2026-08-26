| Algorithm | Reward (all) | Reward (last 20) | Accuracy | Exact loss-mask tokens | Grad norm (mean / max) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRPO | 0.831250 | 0.862500 | 0.915625 | 3930695 | 0.446638 / 8.789782 |
| Dr.GRPO | 0.857500 | 0.912500 | 0.928750 | 3275766 | 0.103210 / 2.017861 |

`Exact loss-mask tokens` is reconstructed from the unmodified per-rank train dumps, not from response length.
