# Config Doctor and Dry-run

`relax.entrypoints.doctor` validates a Relax training configuration before launch. It does not start Ray, Ray Serve, SGLang, GPU workers, or the training loop. It only parses arguments, runs diagnostic rules, derives the role topology, previews resource demand, and prints the expected launch command.

## Basic Usage

Put training arguments after `--`:

```bash
python -m relax.entrypoints.doctor -- \
  --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 8 \
  --global-batch-size 32 \
  --num-rollout 1 \
  --colocate
```

The report contains four core sections:

- `Final merged config`: the parsed and normalized training configuration.
- `Role topology`: algorithm key, candidate roles, required roles, planned roles, per-role resource plan, and placement-group relation.
- `resource_summary`: GPU demand derived from colocate / fully-async / hybrid rules.
- `Expected launch command`: the training entrypoint command that doctor previews.

Use JSON output in CI:

```bash
python -m relax.entrypoints.doctor --format json -- \
  --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 8 \
  --global-batch-size 32 \
  --num-rollout 1 \
  --colocate
```

Valid configurations exit with code `0`. Any `error` diagnostic returns a non-zero exit code. `--strict-warnings` promotes warnings to CI failures.

## Skipping Remote HF Config Access

For purely local static checks in an environment without HuggingFace access, use:

```bash
python -m relax.entrypoints.doctor --doctor-skip-hf-validate -- <training args>
```

This appends `--skip-hf-validate` to the training arguments and avoids fetching remote HF configs during parsing.

## Validation Fallback and Targeted Diagnostics

Doctor first runs the same complete argument parsing and validation as the training entrypoint. If validation fails, it parses again without validation to construct only the merged argument Namespace. This fallback does not fetch remote HF configuration, derive resource fields, run backend validation, or check the TransferQueue version. Mode conflicts and missing roles can therefore retain their targeted rule ids and fixes instead of collapsing into a generic `CONFIG_PARSE_ERROR`.

If argparse cannot parse the arguments themselves, the fallback cannot construct a Namespace and the report retains `CONFIG_PARSE_ERROR`. Both paths only read configuration; neither calls `ray.init()` nor creates training services.

## Topology and Resource Semantics

`candidate_roles` are roles the algorithm registry may create, `required_roles` must have resources for the current configuration, and `roles` are the roles actually planned after applying `--resource`. Fully-async mode requires `reference` only when `--use-kl-loss` or a non-zero `--kl-coef` is set. It does not require `actor_fwd` when the effective configuration is true-on-policy.

In synchronous colocate mode, actor, rollout, and an eligible critic share a placement group, so GPU demand is the maximum among the shared roles. Hybrid mode carries both fully-async and colocate execution flags, but actor and rollout use separate placement groups, so their GPU counts are added.

## Dataset Path Checks

`--prompt-data` accepts one file, a directory, a file list, or the `@[start:end]` slice syntax. Doctor resolves the generalized path before checking each physical file or directory. For example, `[a.jsonl,b.jsonl]@[0:100]` checks `a.jsonl` and `b.jsonl` separately instead of treating the full expression as a filename.

## Sensitive Value Redaction

Text and JSON reports redact sensitive values from `--agent-env`, API keys, tokens, passwords, credentials, private keys, and notification URLs. Redaction covers raw arguments, the expected command, the final merged configuration, parse errors, and diagnostic details. Sensitive values are rendered as `<redacted>`.

## Covered Error Classes

Doctor emits diagnostics by rule id. Current rules cover:

- `CONFIG_RESOURCE_REQUIRED`: missing `--resource`.
- `CONFIG_RESOURCE_SHAPE`: malformed resource entry or `num_serves != 1`.
- `CONFIG_ALGORITHM_SUPPORTED`: unregistered algorithm key.
- `CONFIG_REQUIRED_ROLES`: required roles are absent from `--resource`.
- `CONFIG_MODE_CONFLICT`: direct `--fully-async` plus `--colocate` combination.
- `CONFIG_DEBUG_MODE_CONFLICT`: both debug-only modes are enabled.
- `CONFIG_PPO_TOPOLOGY`: PPO missing critic, unsupported async mode, or invalid staleness.
- `CONFIG_SFT_REQUIREMENTS`: SFT missing data source, dynamic batching, or predict dependencies.
- `CONFIG_BATCH_SIZE`: missing or inconsistent batch-size fields.
- `CONFIG_ROLLOUT_COUNT`: missing rollout bound.
- `CONFIG_OVERSAMPLING`: over-sampling batch smaller than rollout batch.
- `CONFIG_DYNAMIC_BATCH`: dynamic batching without token budget.
- `CONFIG_CONTEXT_LENGTH`: context length exceeds per-GPU token budget.
- `CONFIG_SGLANG_PARALLEL`: SGLang PP/DP parameter conflicts.
- `CONFIG_EVAL`: eval config missing or incompatible with SFT/RL mode.
- `CONFIG_SAVE`: save interval without save path.
- `CONFIG_GENRM_COLOCATE`: invalid GenRM colocate GPU split.
- `CONFIG_PATHS`: local paths do not exist.
- `CONFIG_LORA`: LoRA merge / adapter mode conflict.
- `CONFIG_QKV_FORMAT`: `bshd` conflicts with dynamic batching or non-Megatron backend.
- `CONFIG_ROTATE_CKPT`: checkpoint rotation missing required options.

## Extending Rules

Add new rules in `relax/utils/doctor/rules.py` with `@diagnostic_rule(rule_id, title)`. A rule reads `DoctorContext` and returns `DiagnosticResult` objects. It must not start external processes, call Ray, or allocate GPU resources.

When adding an algorithm or backend, update the pure topology mapping in `relax/utils/doctor/topology.py` and add an error sample to `tests/doctor/fixtures/error_cases.json`.
