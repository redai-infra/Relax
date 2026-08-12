# Reproducibility Bundle

Relax writes a versioned experiment manifest for every training run. The manifest records enough information to review the code, resolved configuration, software environment, hardware, Ray topology, model and data inputs, and the original command without storing credentials or internal addresses by default.

## Output

The default path is:

```text
relax_runs/<experiment-name>/<unique-run-id>/experiment-manifest.json
```

Each default run ID contains a UTC timestamp, process ID, and random suffix, so repeated runs do not overwrite earlier manifests. Set `RELAX_MANIFEST_PATH` to choose a specific file. Manifest creation is best effort: an I/O or collection error emits a warning and never blocks training.

## Schema v1

`schema_version` is currently `1.0`. Readers accept integer v1 and later v1 minor versions, ignore unknown fields, and supply empty defaults for optional sections. A different major version is rejected.

| Section | Contents |
| --- | --- |
| `code` | Repository name, commit, branch, and dirty state |
| `command` | Argument vector and portable working directory |
| `config` | Resolved arguments and Ray runtime environment |
| `environment` | Python, platform, package, CUDA, and container image versions |
| `hardware` | CPU, memory, GPU model, driver, and memory |
| `runtime` | Local, single-node Ray, or multi-node Ray mode; aggregate resources; per-node resource summaries; role and parallel topology |
| `inputs` | Model, checkpoint, and dataset identifiers with cheap local file metadata |

Collection avoids recursive directory scans, model hashing, network calls, and `pip freeze`. The manifest includes `collection_duration_ms` so startup overhead can be measured.

## Redaction

Secret-like keys (for example `token`, `password`, `api_key`, and `authorization`) and address-like keys are replaced with `<redacted>`. The same policy is applied recursively to resolved arguments, runtime environment values, and command-line flags. Ray node IDs, hostnames, IP addresses, and node resource labels are never written.

Only a small allowlist of environment information is collected. A command containing a redacted argument cannot be replayed; provide credentials through the runtime environment instead.

## Check and replay

Compare the current code, Python/package versions, platform, CPU, and GPUs with a manifest:

```bash
python -m relax.entrypoints.reproducibility check path/to/experiment-manifest.json
```

The command exits with `0` when the environment matches, `1` when drift is found, and `2` for an invalid manifest. Every difference includes a suggested fix, which makes the command suitable for CI.

Preview the recorded command:

```bash
python -m relax.entrypoints.reproducibility rerun path/to/experiment-manifest.json
```

Execute it in the current directory after the environment check succeeds:

```bash
python -m relax.entrypoints.reproducibility rerun path/to/experiment-manifest.json --execute
```

Use `--allow-drift` only when the reported differences are intentional. Replay passes the recorded argument vector directly to the process and never invokes a shell.
