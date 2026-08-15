# GraphGPO reproducibility evidence

`examples.graphgpo.reproducibility` generates a content-addressed evidence
bundle without treating planned runs or operator-entered versions as verified
results.

First retain the complete machine-readable test report and the complete
pre-commit output. For example:

```bash
python -m pytest tests/graphgpo \
  --junitxml=/path/to/evidence/pytest-graphgpo.xml \
  2>&1 | tee /path/to/evidence/pytest-graphgpo.log
pre-commit run --all-files \
  2>&1 | tee /path/to/evidence/pre-commit.log
```

Then generate the bundle. Artifact and report labels are public identifiers;
local source paths are used only to read and hash files and are not serialized.

```bash
python -m examples.graphgpo.reproducibility \
  --output-dir /path/to/evidence/bundle \
  --artifact train_manifest=/path/to/train.manifest.json \
  --artifact eval_manifest=/path/to/eval_in_distribution.manifest.json \
  --artifact expanded_command=/path/to/expanded_command.sh \
  --artifact run_lock=/path/to/run_lock.env \
  --junit pytest_graphgpo=/path/to/evidence/pytest-graphgpo.xml \
  --log pytest_graphgpo_full=/path/to/evidence/pytest-graphgpo.log \
  --log pre_commit_full=/path/to/evidence/pre-commit.log \
  --version relax_baseline_commit=<full-40-character-commit> \
  --version candidate_commit=<full-40-character-commit> \
  --version graphgpo_reference_commit=<full-40-character-commit> \
  --version container_image_digest=<image>@sha256:<64-hex-digest> \
  --version model_revision=<immutable-model-revision> \
  --version alfworld_version=<installed-version>
```

Omit a `--version` argument when it is not known. In particular, do not use a
worktree label or the baseline commit as `candidate_commit`; leave it absent
until an immutable candidate commit exists. Missing values are emitted as
`null` with `verification: not_claimed`. Supplied values are marked
`operator_supplied` and `not_verified_by_generator` because a declaration by
itself is not execution evidence.

The generated files are:

- `seed_mapping.json`: the registered GRPO, GiGPO, GraphGPO, and graph-only
  conditions for seeds 0, 1, and 2. Every entry remains `planned`; this file
  never claims a run completed.
- `test_report.json`: test counts parsed from JUnit XML and SHA256 records for
  full text logs. Outcomes are never inferred from unstructured text.
- `reproducibility.manifest.json`: SHA256 and byte size for supplied artifacts
  and generated reports, declared revision fields, and versions observed in
  the interpreter that generated the bundle.

Runtime package observations describe the evidence-generator interpreter,
not a remote training image. Container-side package, CUDA, driver, hardware,
and image inspection records should therefore be supplied as hashed input
artifacts rather than inferred from this command.
