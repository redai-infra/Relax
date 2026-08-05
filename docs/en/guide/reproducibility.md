# Reproducibility Bundle

Relax writes a small, sanitized experiment manifest for every training launch. The manifest captures the command,
resolved arguments, code revisions, software and hardware, input identifiers, training topology, and Ray/Slurm runtime
shape needed to inspect or rerun an experiment. Collection is best effort: a missing tool, timeout, or unwritable output
directory emits a warning and never blocks training.

## Where the manifest is written

`relax.entrypoints.train` records provenance before Ray starts, then updates the same file with the initialized Ray
topology. Rank 0 is the only writer. The default path is:

```text
<tensorboard-dir>/manifest_<run-id>.json
```

If `tensorboard_dir` is unset, Relax uses the launch working directory. Names contain a timestamp and random suffix, so
existing runs are not overwritten. You can also generate a standalone manifest:

```bash
python -m relax.entrypoints.reproducibility generate --output manifest.json
```

## Check, compare, and rerun

Run commands from the same Relax checkout directory used for training.

```bash
# Report code, package, hardware, configuration, and runtime differences.
python -m relax.entrypoints.reproducibility check manifest.json

# Compare two recorded runs and print a Markdown table.
python -m relax.entrypoints.reproducibility diff old.json new.json

# Print, but do not execute, a reconstruction script for review.
python -m relax.entrypoints.reproducibility rerun manifest.json --dry-run

# Validate the current environment, then execute only the recorded argv.
python -m relax.entrypoints.reproducibility rerun manifest.json --confirm
```

`check` returns `0` for an exact match or a compatible warning and `1` for an incompatible difference. A warning is
used when the original Ray cluster is not observable from the checking process or only a compatible patch version
changed. `rerun --confirm` refuses an incompatible environment and never executes a command containing a redacted
argument. The dry-run script may include checkout and package reconstruction commands; review it before using it as a
shell script. The confirmed mode does not install packages or change Git state.

## Schema and compatibility

The authoritative v1 JSON Schema is
[`docs/schema/experiment-manifest-v1.schema.json`](../../schema/experiment-manifest-v1.schema.json). Every generated
manifest has `schema_version: "1.0"`. Readers accept later `1.x` documents, preserve unknown fields, normalize legacy
`cli_args` into `command.argv`, and reject another major version. New optional fields may be added in a minor release;
removing or changing an existing field requires a new major version.

The major sections are:

| Section | Contents |
| --- | --- |
| `command` | Sanitized argv and the normalized launch working directory (`.`) |
| `code` | Relax/Megatron commit, branch, and dirty state |
| `config` | Resolved arguments, selected environment, runtime env, and config hashes |
| `environment` | Python, CUDA/NCCL, and key package versions |
| `hardware` | CPU, memory, NUMA, GPU model/count/memory, and driver |
| `runtime` | Local, single-node Ray, or multi-node Ray mode; node roles/resources; Slurm and parallel topology |
| `inputs` | Sanitized model/tokenizer/dataset identifiers plus bounded size/hash metadata |
| `training` | Algorithm, batch sizes, and TP/PP/CP/EP/DP topology |

Collectors are independent and optional. A timed-out collector can therefore leave a section absent while the
manifest remains a valid best-effort record.

## Privacy and safety boundaries

Relax recursively removes secret-bearing keys, authorization headers, URL passwords, private IP addresses, internal
hostnames, home-directory owners, and infrastructure endpoint fields. Environment variables are allowlisted by prefix
before collection, and only a smaller allowlist can be restored for a rerun. Node identity and Ray node-address
resources are not persisted.

Small input files (at most 1 MiB) may receive a SHA-256 digest; model weights and datasets are never copied into the
bundle. A redacted command is intentionally not runnable. Treat the manifest as shareable operational metadata, but
review it before publishing because application-specific free-form text cannot be proven secret-free in general.

## Minimal CPU example

The following commands exercise manual generation, environment checking, and dry-run inspection without starting
training:

```bash
python -m relax.entrypoints.reproducibility generate --output manifest.json
python -m relax.entrypoints.reproducibility check manifest.json
python -m relax.entrypoints.reproducibility rerun manifest.json --dry-run
```

For a real rerun, use the manifest automatically emitted by the original training command and invoke `--confirm` only
after `check` and the dry-run output are understood.
