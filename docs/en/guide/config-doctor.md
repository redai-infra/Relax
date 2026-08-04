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

The report contains these core fields:

- `config_state`: `validated`, `partial`, or `unavailable`, distinguishing complete configurations from fallback results.
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

Doctor first runs the same complete argument parsing and validation as the training entrypoint. If validation fails, it parses again without validation and marks the resulting Namespace as `partial`. This fallback does not fetch remote HF configuration, derive resource fields, run backend validation, or check the TransferQueue version.

A partial configuration runs only rules explicitly declared partial-safe and does not produce a role topology or GPU estimate. The report always retains the original `CONFIG_PARSE_ERROR`, even when targeted rules also match, so the complete validation failure is not hidden. Any rule exception is converted to a structured `DOCTOR_RULE_EXECUTION_ERROR` instead of escaping as a traceback.

Doctor collects the options actually registered by the Relax, Megatron, SGLang, and teacher parsers. An option not registered by any parser produces `CONFIG_UNKNOWN_ARGUMENT`, so a misspelled flag such as `--does-not-exist` is no longer silently ignored.

## Topology and Resource Semantics

`candidate_roles` are roles the algorithm registry may create, `required_roles` must have resources for the current configuration, and `roles` are the roles actually planned after applying `--resource`. Fully-async mode requires `reference` only when `--use-kl-loss` or a non-zero `--kl-coef` is set. It does not require `actor_fwd` when the effective configuration is true-on-policy.

Role selection, optional roles, managed teachers, placement groups, and GPU totals are derived by `relax/core/service_plan.py`; Controller and Doctor consume the same result. In synchronous colocate mode, actor, rollout, and an eligible critic share a placement group, so GPU demand is the maximum among shared roles. Hybrid actor and rollout services use separate placement groups, so their GPU counts are added.

`advantages` and `sft` are CPU roles that may use `[1, 0]`. Model roles such as actor, rollout, critic, reference, actor_fwd, genrm, and managed teacher require a positive GPU count.

## Dataset Path Checks

`--prompt-data` accepts one file, a directory, a file list, or the `@[start:end]` slice syntax. Doctor resolves the generalized path before checking each physical file or directory. For example, `[a.jsonl,b.jsonl]@[0:100]` checks `a.jsonl` and `b.jsonl` separately instead of treating the full expression as a filename.

## Sensitive Value Redaction

Text and JSON reports redact `--agent-env`, sensitive fields nested in `--train-env-vars`, `--wandb-key`, API keys, tokens, passwords, credentials, private keys, and notification URLs. Redaction covers raw arguments, the expected command, the final merged configuration, parse errors, and diagnostic details. Sensitive values are rendered as `<redacted>`.

## Covered Error Classes

Doctor emits diagnostics by rule id. Current rules cover:

- `CONFIG_RESOURCE_REQUIRED`: missing `--resource`.
- `CONFIG_RESOURCE_SHAPE`: malformed resource entry, `num_serves != 1`, or a model role with zero GPUs.
- `CONFIG_UNKNOWN_ARGUMENT`: an option is not registered by any runtime parser.
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

When adding an algorithm, role, or backend, update the shared plan in `relax/core/service_plan.py` and add plan coverage used by both Controller and Doctor. Error samples live in `tests/doctor/fixtures/error_cases.json`.
