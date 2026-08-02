# Task 22 sync-intent clean A/B

## Goal

Measure the sync-intent admission policy without Task 22 calibration
instrumentation. Both arms use the same clean-policy commit and command line.
The only behavioral switch is:

```text
OFF  RELAX_SYNC_INTENT_POLICY=0
ON   RELAX_SYNC_INTENT_POLICY=1
```

OFF calls the upstream `generate_rollout_async` implementation directly. ON
calls the separate sync-intent rollout loop.

## Fixed workload

```text
model                              Qwen3-4B
dataset                            dapo-math-17k
topology                           2 Actor GPUs (TP=2) + 2 single-GPU rollout engines
num rollout                        20
headline window                    Actor steps 2-19
rollout batch                      8 prompt groups
samples per prompt                 8
global batch                       64 samples
max response                       8192 tokens
rollout seed                       42
train seed                         1234
max staleness                      2
partial rollout                    enabled
partial max aborted count          2
mask partial off-policy prefix     enabled
over-sampling window               16 groups
weight publication                 every Actor step
```

## Fixed execution configuration

```text
Actor max tokens/GPU               9216
Actor attention                    FlashAttention
Rollout attention                  Triton
Rollout sampling                   PyTorch
SGLang static memory fraction      0.8
SGLang CUDA graph max batch        64
piecewise CUDA graph               disabled
priority scheduler                 enabled in both arms
priority preemption                disabled
default priority                   0
old-debt priority                  1 in ON only
```

The priority scheduler is enabled in both arms so its scheduler overhead is
common. OFF does not attach a request priority payload because upstream samples
do not carry the ON-only `work_origin` metadata.

## Policy parameters

These values are present in both contracts but only consumed by ON:

```text
candidate window                   16 groups
intent TTL                         600 seconds
quiesce margin                     max(2 seconds, 1.25 * recent median group latency)
Actor train ETA                    EWMA(0.7 previous, 0.3 current)
```

The policy does not read a dataset name, reward, response length, step
threshold, or slow/fast step list.

## Instrumentation exclusions

Neither arm enables:

- Task 22 calibration timeline or timestamp logs;
- permit lifecycle JSONL;
- one-second scheduler heartbeat;
- high-frequency `nvidia-smi` sampling;
- `--sglang-show-time-cost`;
- ClearML;
- timeline dumps.

Only standard framework logs and the normal `perf <step>` metrics are retained.
The runner persists its log, lifecycle, and canonical contract after the job
exits.

## Execution order

Use the same instance and clean commit:

1. ON, 20 steps;
2. OFF, 20 steps.

Running ON first is conservative: any persistent host or compile cache benefit
for the second arm favors OFF. Each runner restarts Ray/Serve through the normal
local entrypoint. Steps 0-1 are excluded from the headline window.

Before analysis, run:

```bash
python scripts/task22/compare_sync_intent_clean_contracts.py \
  <off>/run_contract.json \
  <on>/run_contract.json
```

The comparison must report `PASS`.

## Metrics and decision

Primary:

```text
samples/s = 18 * 64 / sum(step_time[2:20])
```

Also report:

- response-token proxy per pipeline second;
- Actor train tokens and weighted Actor tokens/s;
- train wait and gate wait;
- response length, raw reward, TIS clip fraction, mismatch KL, and grad norm;
- step P50/P95 and paired slow-plus-fast cycle statistics.

GO requires:

1. ON aggregate samples/s exceeds OFF by at least 5%;
2. the gain is not explained by shorter response or fewer Actor train tokens;
3. Actor weighted tokens/s does not regress by more than 2%;
4. no missing samples, OOM, NaN/Inf, hang, or fatal error;
5. quality guardrails show no obvious short-run regression.
