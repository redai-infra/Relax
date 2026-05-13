---
name: ssh-ray-cluster
description: Connect to a remote Ray cluster head node via SSH (paramiko) to execute
  commands, check cluster status, inspect logs, and debug training jobs. Use this
  skill when the user asks to SSH into a remote machine, check Ray cluster status,
  or run remote commands on the Ray head node.
---

# SSH to Ray Cluster

This skill provides a standardized way to connect to a remote Ray cluster head node via SSH using `paramiko`, execute commands, and retrieve results. It is used for cluster inspection, log retrieval, and remote debugging.

______________________________________________________________________

## Prerequisites

The user must provide the following details (ask if missing — do not invent
values, and do not write them into this skill file):

| Parameter             | Purpose                                  |
| --------------------- | ---------------------------------------- |
| `host`                | Remote machine IP                        |
| `port`                | SSH port                                 |
| `username`            | SSH username                             |
| `password`            | SSH password                             |
| `RELAX_PROJECT_ROOT`  | Absolute path to the Relax project root  |

Connection details and the project root are typically recorded in the
session's auto-memory (see `reference_ray_cluster_ssh.md`). Read them from
memory or ask the user — do not hard-code them in this skill or in scripts
checked into the repo.

______________________________________________________________________

## HARD REQUIREMENT — always run project commands from the Relax project root

A one-shot `paramiko.exec_command` starts the remote shell in the user's
home directory (typically `/root` or another non-project dir), **not** in
the Relax project root. Any command that touches a project-relative path
(`scripts/...`, `log/...`, `relax/...`, `tests/...`, `pyproject.toml`,
etc.) MUST be prefixed with `cd "$RELAX_PROJECT_ROOT"` (or the resolved
path) **inside the same command string** — splitting `cd` into a separate
`exec_command` call does NOT work, because each call opens a fresh shell
back at the home directory.

`RELAX_PROJECT_ROOT` is a session-level value supplied by the user / read
from auto-memory (see Prerequisites). Do **not** hard-code its value in
this skill, in checked-in scripts, or in any reusable artifact — resolve
it at command-build time from memory or by asking the user.

### Required pattern for any project-relative command

```python
# RELAX_PROJECT_ROOT must be resolved from memory or user input first.
cmd = (
    f'cd {shlex.quote(RELAX_PROJECT_ROOT)} && '
    '<your command here>'
)
ssh.exec_command(cmd, timeout=...)
```

Examples that REQUIRE the `cd` prefix:

- `bash scripts/entrypoint/ray-job.sh ...`
- `bash scripts/training/text/run-<model>-<size>-<gpus>.sh`
- `python scripts/tools/run_on_each_ray_node.py ...`
- `bash scripts/tools/kill_for_ray.sh`
- `tail -n 100 log/<run-name>-*.log`
- `pre-commit run --all-files`
- `pytest tests/test_foo.py`

Examples that do NOT need the `cd` (they take absolute paths or are
host-global tools that touch no project files):

- `ray status`, `ray job list`, `ray job logs <ID>`, `ray job status <ID>`
- `nvidia-smi ...`
- `ls /tmp/ray/session_latest/logs/`
- `ps -ef | grep ...`

When in doubt: add the `cd`. It is harmless on host-global commands and
mandatory on project-relative ones.

### Symptom that the `cd` was lost

```
bash: scripts/...: No such file or directory
python: can't open file '<home-dir>/scripts/...'
ls: cannot access 'log/': No such file or directory
```

Fix: add `cd "$RELAX_PROJECT_ROOT" && ` to the front of the command and
re-run. Do NOT retry blindly — a missing `cd` will keep failing the same
way.

______________________________________________________________________

## Connection Pattern

Use Python's `paramiko` library to establish SSH connections. Always use a **one-shot** pattern: connect, execute, close. Do not try to maintain persistent connections across tool calls.

### Basic connection template

```python
python3 -c "
import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('<HOST>', port=<PORT>, username='<USERNAME>', password='<PASSWORD>', timeout=10)

stdin, stdout, stderr = ssh.exec_command('<COMMAND>', timeout=30)
output = stdout.read().decode()
errors = stderr.read().decode()
print(output)
if errors:
    print('STDERR:', errors)

ssh.close()
"
```

### Multi-command template

When you need to run multiple commands in sequence:

```python
python3 -c "
import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('<HOST>', port=<PORT>, username='<USERNAME>', password='<PASSWORD>', timeout=10)

commands = [
    ('Description 1', 'command1'),
    ('Description 2', 'command2'),
]

for desc, cmd in commands:
    print(f'=== {desc} ===')
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30)
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err:
        print('STDERR:', err)
    print()

ssh.close()
"
```

______________________________________________________________________

## Common Operations

### 1. Check Ray cluster status

```bash
ray status 2>&1 | head -30
```

Shows active/idle nodes, GPU/CPU usage, pending demands.

### 2. List Ray jobs

```bash
ray job list 2>&1 | head -50
```

Shows all submitted jobs with their status (RUNNING, FAILED, SUCCEEDED).

### 3. Get running job logs

```bash
ray job logs <JOB_ID> 2>&1 | tail -100
```

### 4. Check GPU usage across nodes

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

### 5. Check specific worker node logs

SGLang engine logs are typically found in Ray's log directory:

```bash
ls -lt /tmp/ray/session_latest/logs/ | head -20
```

### 6. Kill residual processes

```bash
cd "$RELAX_PROJECT_ROOT" && bash scripts/tools/kill_for_ray.sh
```

### 7. Run command on all nodes

```bash
cd "$RELAX_PROJECT_ROOT" && python scripts/tools/run_on_each_ray_node.py command "<COMMAND>"
```

### 8. Launch / relaunch a training run

Per the HARD REQUIREMENT above, the `cd` into the project root and the
launch must be in the **same** command string.

```bash
cd "$RELAX_PROJECT_ROOT" && \
  nohup bash scripts/entrypoint/ray-job.sh <RUN_SCRIPT> > <OUT_FILE> 2>&1 &
```

Verify CWD before launch by chaining `pwd && ls scripts/entrypoint/ray-job.sh`
in the same command — if `pwd` doesn't report the project root, the cd was
dropped and the launch will fail.

______________________________________________________________________

## Working directory pitfall (`cd` over SSH) — supplementary patterns

See the HARD REQUIREMENT section near the top for the rule. Two equivalent
patterns satisfy it; mixing them does not:

1. **Single-line, single-shell** (preferred for paramiko `exec_command`):
   chain `cd` with `&&` inside the *same* quoted command string, e.g.
   `ssh ... 'cd "$RELAX_PROJECT_ROOT" && bash scripts/...'`. If you split
   the `cd` into a separate `ssh` / `exec_command` invocation, the next call
   starts back in the home directory.

2. **Heredoc to remote bash** (useful for multi-step launches):

   ```bash
   ssh ... bash <<EOF
   cd "$RELAX_PROJECT_ROOT"
   nohup bash scripts/entrypoint/ray-job.sh <RUN_SCRIPT> > <OUT_FILE> 2>&1 &
   echo "PID=\$!"
   EOF
   ```

Symptom that the cd was lost: `bash: <script>: No such file or directory`,
`python: can't open file '<home>/scripts/...'`, or `ls: cannot access
'log/'`. Fix by adding the `cd "$RELAX_PROJECT_ROOT" && ` prefix and
re-launching — do not retry blindly.

______________________________________________________________________

## Verifying long-running launches over paramiko

`paramiko.exec_command` will raise `socket.timeout` / `PipeTimeout` on the
`stdout.read()` side if the remote command hasn't finished within `timeout`
seconds — even though the command itself keeps running on the remote end. This
is **expected** for chained commands that include long-running steps like
`ray serve shutdown -y` (~30s) or backgrounded `nohup bash ray-job.sh ...`.

**Don't retry blindly on PipeTimeout.** Instead, treat it as "submission is
in flight, verify separately":

```python
# 1. Fire the launch chain — accept that read() may timeout
try:
    stdout.read()
except Exception:
    pass

# 2. Open a fresh connection and verify
ssh2.exec_command("pgrep -af 'ray-job.sh' | head; ray job list 2>&1 | grep RUNNING | head")
```

A successful launch is confirmed by either: a `ray-job.sh` PID still alive,
or a fresh `RUNNING` entry in `ray job list` with a recent `start_time`.

______________________________________________________________________

## Forbidden Operations

When SSH'd into a Ray cluster, **never** execute the following destructive commands without an explicit, in-conversation user request — the user is typically running a job that must not be killed:

- `ray stop` (kills the entire Ray runtime on the node)
- `pkill -9 python` / `pkill -9 -f ...` / any wide `kill -9` against training processes
- `bash scripts/tools/kill_for_ray.sh` (despite being listed under Common Operations — treat it the same way)
- `ray job stop <id>`
- `rm -rf` on `/tmp/ray/` or any session/log directory while a job is running

These bypass graceful shutdown, lose in-flight state, and can crash other tenants on shared clusters. If you think one is needed (e.g. cleanup before a fresh run), ask first and quote the exact command.

Read-only inspection (`ray status`, `ray job list`, `ray job logs`, `nvidia-smi`, `ls`, `cat`, `tail`, `grep`, `ps`, `py-spy dump`) is always fine.

______________________________________________________________________

## Important Notes

1. **Timeout**: Always set `timeout=10` on `ssh.connect()` and `timeout=30` on `exec_command()` to avoid hanging.
2. **Always close**: Call `ssh.close()` in a finally block or at the end to release the connection.
3. **No interactive commands**: Never run interactive commands (vim, top without -b, etc.) via paramiko.
4. **Head node has no GPU**: The Ray head node typically has no GPUs. GPU commands should be run on worker nodes via `run_on_each_ray_node.py`.
5. **One-shot pattern**: Each `execute_command` call creates a fresh SSH connection. Do not attempt to reuse connections across tool calls.
6. **Large output**: For commands that produce large output, always pipe through `head` or `tail` to limit the response size.

______________________________________________________________________

## Troubleshooting

### Connection refused

- Verify the SSH tunnel is active on the user's side
- Check that the port is correct (non-standard ports like 2360 are common with tunneling)

### Command timeout

- Some commands (e.g., `ray job logs` for large jobs) may take a long time
- Increase `timeout` parameter on `exec_command()` or use `tail`/`head` to limit output

### Permission denied

- Verify username and password
- Some clusters require key-based authentication — use `paramiko.RSAKey.from_private_key_file()` instead
