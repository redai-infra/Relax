# GraphGPO on ALFWorld

This recipe reproduces GraphGPO on ALFWorld text-only with
Qwen2.5-1.5B-Instruct. It keeps the Relax rollout and Megatron training path:
each environment turn is one named explicit export, and the group-level hook
returns one scalar advantage for each exported turn.

The recipe supports three methods through the same data and launch path:

- `METHOD=grpo`: episode-level GRPO advantage only.
- `METHOD=gigpo`: GiGPO same-state step advantage plus episode advantage.
- `METHOD=graphgpo`: graph edge advantage plus episode advantage.

The formal reproduction uses the default
`EPISODE_WEIGHTING=trajectory_once`: each episode return participates in the
group statistics exactly once, then the normalized value is broadcast to that
trajectory's turns. This preserves native GRPO weighting when trajectories
have different lengths. `EPISODE_WEIGHTING=reference_cross_steps` is retained
only as an explicit historical-reference ablation; it must not be used for the
main GRPO, GiGPO, or GraphGPO comparison.

## Frozen algorithm contract

Each custom-advantage invocation contains one current rollout group: eight
trajectories from exactly one task. Episode return is
`10 * success - 0.1 * invalid_action_count`. The default episode advantage
standardizes those eight trajectory returns with sample standard deviation
(`ddof=1`) and `epsilon=1e-6`, once per trajectory, before broadcasting the
result to every real turn in that trajectory.

The graph has one occurrence edge per real turn. Repeated and parallel edges
remain repeated when same-source graph returns are standardized, while the
shortest-path topology uses the minimum observed cost for each source/target
pair. The final real action of a successful trajectory points to the canonical
`__GRAPHGPO_SUCCESS__` sink. Self-loops and cycles are allowed. Nodes that
cannot reach success receive `max_finite_distance + 1`; an all-fail graph has
zero graph advantage, and singleton or zero-variance groups also return zero.

The accepted Task 37/reference-commit reward is
`10 * omega ** d(next_state)` (unit costs, `omega=0.1`). Thus same-source edges
whose next-state distances are `0`, `1`, and `2` have the fixed oracle
`[10, 1, 0.1]`. This intentionally differs from paper Eq. 4,
`10 * omega ** (d(next_state) + cost)`, whose unit-cost values are smaller by a
factor of `omega`. Both conventions remain explicit in the graph kernel, but
training uses the Task/reference convention.

An ALFWorld state key is SHA256 over `reference_anchor_v1`: the raw observation
plus the reference-visible tracker context and the complete sorted admissible
command list (including `help`). Tracker input is restricted to exactly
`location`, `holding`, `history_items`, and `item_location`. Top-level and
nested mappings are canonicalized with the fixed whitelist order and sorted
nested keys, finite JSON values, UTF-8 text, and no insignificant whitespace
before rendering and hashing.

Graph diagnostics are disabled by default, including their timers. To record
them, set `GRAPHGPO_DIAGNOSTICS_JSONL` to an output path. Each task-local graph
then appends one JSONL record with `perf_counter_ns` durations for graph build,
reverse Dijkstra, and graph-advantage calculation; node and edge-occurrence
counts; exact `(source, action, target)` duplicate rate; cross-trajectory
shared-source rate; singleton-source rate; nonzero-advantage rate; all-fail
status; unreachable-node rate; and a node-distance histogram. The diagnostics
operate only on the pure-Python graph objects and do not synchronize GPU work.

```bash
GRAPHGPO_DIAGNOSTICS_JSONL=/path/to/run/graph-diagnostics.jsonl \
  METHOD=graphgpo ... \
  bash examples/graphgpo/run_alfworld_qwen2_5_1_5b.sh
```

After the run, retain the raw JSONL and create a non-overwriting summary with
per-stage totals plus median and nearest-rank p95 values:

```bash
python3 -m examples.graphgpo.diagnostics_summary \
  --input /path/to/run/graph-diagnostics.jsonl \
  --output /path/to/run/graph-diagnostics.summary.json
```

Every turn's metadata must carry unique `row_id` plus `rollout_group_id` and
`policy_version`. The custom adapter rejects mixed group/version inputs before
using the compact local `(trajectory_id, turn_index)` graph key. The latter
therefore remains compatible with existing graph math and output lookup while
the full identity tuple is preserved and checked at the transport boundary.

The implementation follows the frozen Task 37 proposal. It does not claim the
paper's reported success rate until the registered multi-seed experiment has
finished.

## Frozen inputs

- Relax commit: `8a54679e971087566bfda939a5f75649e07fb861`
- GraphGPO reference commit:
  `20bd331bdbc9026a5668e11362178e10ab7400c8`
- Model:
  `Qwen/Qwen2.5-1.5B-Instruct@775b11afaf83e0dc75bd5abaf90133e47b3ec082`
- ALFWorld: `0.4.2`, text-only
- Expected base image for the external execution protocol:
  `ghcr.io/redai-infra/relaxrl@sha256:3fa8ce578acda6c829b83016bde42c38fa892681e4f36ca330f545616fe578e2`

Install ALFWorld in an isolated environment and download its text data:

```bash
uv venv --python 3.10 /path/to/alfworld-0.4.2
uv pip install --python /path/to/alfworld-0.4.2/bin/python \
  -r examples/graphgpo/requirements-alfworld.txt
export ALFWORLD_DATA=/path/to/alfworld-data
/path/to/alfworld-0.4.2/bin/alfworld-download \
  --data-dir "${ALFWORLD_DATA}"
uv pip freeze --python /path/to/alfworld-0.4.2/bin/python \
  > /path/to/locks/alfworld-0.4.2-python.freeze.txt
```

Do not start GPU training until importing ALFWorld and a real text-only
`reset`/`step` smoke both pass.

If ALFWorld is installed in a separate interpreter, set
`ALFWORLD_PYTHON=/path/to/alfworld-0.4.2/bin/python`. That interpreter also
uses the pinned OpenAI Python SDK and `httpx` in
`requirements-alfworld.txt`, because the managed process talks to Relax's
local chat-completions endpoint. The training process itself still uses the
image's normal `python3`.

## Prepare manifests and prompt rows

The preparation command does not import ALFWorld. It filters the downloaded
games using ALFWorld 0.4.2's text-environment rules, hashes every selected game,
trajectory, PDDL, and grammar file, and writes one seed row per task. Each row
stores the complete `ALFWORLD_DATA`-relative `game.tw-pddl` path as `task_id`.
The environment adapter sorts all discovered game files, restricts the
single-task environment to that exact file, and verifies the ID returned by
`reset`. A missing, outside-root, or mismatched game fails the trajectory.

```bash
export ALFWORLD_DATA=/path/to/alfworld-data
python3 -m examples.graphgpo.prepare_alfworld \
  --data-root "${ALFWORLD_DATA}" \
  --output-dir /path/to/task37-data-lock
```

The default outputs are:

```text
train.manifest.json
train.prompts.jsonl
eval_in_distribution.manifest.json
eval_in_distribution.prompts.jsonl
prepare.lock.json
```

Every prompt row also freezes the request-level sampling contract consumed by
the managed agent: train uses `temperature=1.0`, validation uses
`temperature=0.4`, and both use `top_p=1.0`, `max_tokens=512`. These values
are recorded in `prepare.lock.json`; the OpenAI-compatible request does not
rely on an inference-server default.

Training keeps all eligible train games. Validation takes the first 128
eligible `valid_seen` games in normalized path order. Override either choice
explicitly:

```bash
python3 -m examples.graphgpo.prepare_alfworld \
  --data-root "${ALFWORLD_DATA}" \
  --output-dir /path/to/task37-data-lock-custom \
  --limit train=512 \
  --limit eval_in_distribution=128
```

Prepared files are content-addressed and idempotent. The command refuses to
replace an existing file with different bytes.

Before a dry run or training launch, the recipe rechecks the lock, manifest,
and prompt hashes; split names; task counts and order; `MAX_STEPS`; and the
model revision. It also rehashes every ALFWorld game, trajectory, and shared
asset referenced by the selected manifests. A mismatch stops before Ray or a
GPU job is contacted.

The model snapshot has an independent JSON lock with schema
`task37-huggingface-model-lock-v1`. It pins the repository revision and the
size and SHA256 of all nine snapshot files. Set `MODEL_LOCK` to that file; the
preflight rehashes the local checkpoint before launch.

The environment config keeps `num_train_games` and `num_eval_games` at `-1`.
This is intentional: ALFWorld must not truncate its unsorted discovery list
before the adapter applies normalized sorting and the manifest task binding.

## Dependency-free checks

The graph kernel, action parser, state adapter, manifest builder, managed-agent
contract, and packaging tests run without ALFWorld or a GPU:

```bash
python -m pytest tests/graphgpo -q
```

The ALFWorld import remains inside the real environment factory, so importing
`examples.graphgpo` and running fake-environment tests does not require the
optional dependency.

Agentic group-RM evaluation still invokes a batched reward function. The
launcher therefore wires
`examples.graphgpo.reward.reward_func`, which validates and forwards the
finite environment reward already attached to every explicit turn row. It
does not rescore model text.

## Inspect the launch without using a GPU

Set paths to existing local fixtures and use `DRY_RUN=1`. The script validates
the frozen inputs and prints the fully expanded Relax command without
contacting Ray:

```bash
DRY_RUN=1 \
METHOD=graphgpo \
ALFWORLD_DATA=/path/to/alfworld-data \
HF_CHECKPOINT=/path/to/Qwen2.5-1.5B-Instruct-snapshot \
SAVE_DIR=/path/to/task37-output \
DATA_ARTIFACT_DIR=/path/to/task37-data-lock \
DEPENDENCY_LOCK=/path/to/locks/alfworld-0.4.2-python.freeze.txt \
MODEL_LOCK=/path/to/locks/qwen2.5-1.5b-775b11af-v1.lock.json \
ALFWORLD_PYTHON=/path/to/alfworld-0.4.2/bin/python \
bash examples/graphgpo/run_alfworld_qwen2_5_1_5b.sh
```

The default environment config is
`configs/alfworld_qwen2_5_1_5b.yaml`. It uses only the text environment and
expands all dataset paths from `ALFWORLD_DATA`.

Evaluation is enabled by default with `ENABLE_EVAL=1`, preserving the formal
128-task evaluation path. For a rollout/training plumbing smoke only, set
`ENABLE_EVAL=0`; the launcher then omits the evaluation artifacts from
preflight and installs no evaluation arguments. The selected value is written
to the run lock. Formal runs must retain the default.

## Train

After the real environment smoke and the dry-run command pass:

```bash
METHOD=graphgpo \
SEED=0 \
NUM_GPUS=2 \
OUTER_EPOCHS=150 \
TASK_GROUPS=16 \
GROUP_SIZE=8 \
GLOBAL_BATCH_SIZE=128 \
ALFWORLD_DATA=/path/to/alfworld-data \
HF_CHECKPOINT=/path/to/Qwen2.5-1.5B-Instruct-snapshot \
SAVE_DIR=/path/to/task37-output \
DATA_ARTIFACT_DIR=/path/to/task37-data-lock \
DEPENDENCY_LOCK=/path/to/locks/alfworld-0.4.2-python.freeze.txt \
MODEL_LOCK=/path/to/locks/qwen2.5-1.5b-775b11af-v1.lock.json \
ALFWORLD_PYTHON=/path/to/alfworld-0.4.2/bin/python \
bash examples/graphgpo/run_alfworld_qwen2_5_1_5b.sh
```

Change only `METHOD` and `SEED` for the registered GRPO/GiGPO/GraphGPO
comparison. The script fixes group size, sampling, batch, optimizer, KL, and
model settings across methods. Keep the default `trajectory_once` weighting
unchanged across the three methods. SGLang static memory fraction defaults to
the reference recipe's `0.50` for every method; any explicit
`SGLANG_MEM_FRACTION_STATIC` override is validated in `(0, 1]` and recorded in
the run lock. The script also records a local run lock and the
expanded command under `SAVE_DIR/runs/`. These raw files can contain local
paths and must be scrubbed before they are copied into public report material.
The run lock records the image digest expected by the outer execution
protocol, but the recipe script cannot prove which image launched it. The
outer container/Ray job runner must actually select that digest and retain its
own runtime inspection evidence.

For the two registered ablations:

```bash
# Episode-only parity.
METHOD=graphgpo BETA=0 BETA_EPISODE=1 ...

# Graph-only.
METHOD=graphgpo BETA=1 BETA_EPISODE=0 ...
```

`RELAX_AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE` is set to `MAX_STEPS` by the
launcher. Variable turn counts therefore use the explicit variable-row path;
the final physical batch is padded only with zero-loss rows.

## Output contract

Each trajectory writes `turn_000`, `turn_001`, and so on as JSONL records.
Every record contains only the current user prompt and assistant response.
Metadata carries the task, trajectory, turn, state transition, validity,
terminal state, and episode return needed by
`examples.graphgpo.custom_advantage.compute_custom_advantage`.

Reports must aggregate success and return after deduplicating by
`trajectory_id`; averaging per-turn rows would length-weight long episodes.
The launcher installs the custom eval hook that records episode count,
success rate, mean episode return, and truncation rate with exactly one vote
per trajectory.
