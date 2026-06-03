---
name: ssh-ray-cluster
description: 3-step debug loop for remote Ray cluster — submit task via SSH, check
  logs locally, analyze errors and fix code, repeat until resolved.
---

# SSH Debug Loop

Three-step cycle: **submit** -> **check logs** -> **analyze & fix** -> repeat.

## Prerequisites

Read SSH credentials and `RELAX_PROJECT_ROOT` from auto-memory (`reference_ray_cluster_ssh.md`). Ask the user if missing — never hard-code in this file.

## Step 1: Submit Task via SSH

Use paramiko to SSH into the cluster, `cd` to the project root, and execute the user's command.

```python
python3 -c "
import paramiko, shlex
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=10)
cmd = f'cd {shlex.quote(RELAX_PROJECT_ROOT)} && <USER_COMMAND>'
try:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=60)
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err: print('STDERR:', err)
except Exception: pass  # long-running commands may timeout — that's OK
finally: ssh.close()
"
```

**Key rule**: All project-relative commands (`bash scripts/...`, `tail log/...`) MUST have `cd $RELAX_PROJECT_ROOT &&` in the **same** command string. Paramiko opens a fresh shell each call.

For backgrounded launches, verify separately:
```bash
pgrep -af 'ray-job.sh' | head
ray job list 2>&1 | grep RUNNING | head
```

## Step 2: Check Logs Locally

The log file is on a shared filesystem mounted locally. Read it directly:

```bash
# Find the latest log
ls -lt log/<model>-*.log | head -5

# Read the tail for errors
tail -200 log/<run-name>.log
```

Use the `Read` tool on the log file path. Search for keywords: `Error`, `Exception`, `Traceback`, `FAILED`, `RuntimeError`, `AssertionError`.

**Check frequency**: Wait at least **1 minute** between log checks. Don't poll more frequently — training jobs take minutes to hours, and frequent checks waste context.

## Step 3: Analyze & Fix

1. **Identify the error** from the log (traceback, error message, hang pattern).
2. **Fix the code** if the root cause is clear — edit the source file directly.
3. **Add debug logging** if the root cause is unclear — add targeted `logger.info`/`logger.error` calls to narrow down the issue.
4. **Go back to Step 1** — resubmit the task and repeat until resolved.

## Safety Rules

**NEVER** execute these without explicit user request:
- `ray stop`, `bash scripts/tools/kill_for_ray.sh`, `pkill`, `ray job stop`
- `rm -rf` on `/tmp/ray/` or session directories

Only `ray serve shutdown -y` is allowed pre-submit (when user requests a relaunch).

Read-only inspection (`ray status`, `ray job list`, `ray job logs`, `nvidia-smi`, `tail`, `grep`, `ps`) is always safe.
