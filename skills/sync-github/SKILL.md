---
name: sync-github
description: Use when syncing Relax code between internal GitLab and external GitHub, especially gitlab/dev, gitlab/main, github/main, internal CR/MR handoff, linear main history, sensitive-content checks, GitHub Actions CI validation, or guarded GitHub pushes.
---

# sync-github

同步内部 GitLab 与外部 GitHub 时使用。这个 skill 是入口和规则层；完整可执行流程拆成两个 prompt reference。

## 两阶段硬规则

每次完整同步必须分成两个独立人工闸门阶段，不能一次跑完：

1. **Prompt A：GitHub -> GitLab dev**  
   找到 GitHub 分支从 `gitlab/dev` 分出去的基点 `commitA`，识别 `commitA` 之后 GitHub PR 合入的外部 commit，只 cherry-pick 这些外部 commit 到新的 GitLab CR 分支。创建或确认无需 CR 后必须停止，等待云效 CR 提交和合入。
2. **Prompt B：GitLab dev -> GitLab/GitHub main**  
   只能在用户明确说“内部 CR 已合入，继续同步”后执行，直接在本地 `main` 从 `gitlab/main` 开始，用真实 tree diff 找最小有效同步集并线性重放到 `main`，直接推送到 `gitlab/main`，再走 GitHub push 门禁推到 `github/main`。不要创建 `sync/dev-to-main`。

禁止在 Prompt A 中 push GitHub。禁止跳过 Prompt A 直接做 `dev -> main`。
Prompt A 禁止通过 merge 把 `github/main` 纳入 `gitlab/dev`；必须走 commitA + external PR commit cherry-pick 流程。

## 必须先判断阶段

- 开始一次同步，或用户没有明确说“内部 CR 已合入”：读取并执行 [references/prompt-a-main-to-dev.md](references/prompt-a-main-to-dev.md)。
- 用户明确说“内部 CR 已合入，继续同步”：读取并执行 [references/prompt-b-dev-to-main.md](references/prompt-b-dev-to-main.md)。Prompt B 直接更新并推送 `gitlab/main`；如果 `github/main` 落后于 `gitlab/main`，不要回退 `gitlab/main`，继续在 `gitlab/main` 基础上追加 `dev` 内容，最后停在 GitHub push 门禁。
- 不确定阶段时，只做只读检查：`git status --porcelain`、`git status --porcelain --untracked-files=no`、`git fetch --all --prune`，再运行 `python skills/sync-github/scripts/plan_github_to_dev.py --github github/main --dev gitlab/dev` 判断是否还有未吸收的外部 PR commit。不要根据 `gitlab/main`/`github/main` 关系跳过 Prompt A。

## Prompt B 效率硬规则

- Prompt B 的工作队列必须先由真实 tree diff 驱动：先看 `git diff --stat gitlab/main..gitlab/dev` / `git diff --name-status gitlab/main..gitlab/dev`，再用 commit 历史解释这些路径来自哪些有效提交。
- `git rev-list --right-only gitlab/main...gitlab/dev` 只用于审计和追溯，不能直接当作 cherry-pick 队列；GitLab merge 历史里有大量已被 main 吸收的旧 merge 噪音。
- 遇到 `Merge branch main into dev`、Prompt A merge、已被 tree diff 证明吸收的旧 merge，直接记录跳过。不要为了这些历史 merge 解冲突。
- 如果 Prompt A 已实际验证某个 GitHub external commit cherry-pick 到 `gitlab/dev` 为空/已吸收，而规划脚本仍报告 `not-in-dev`，按人工审计 false positive 记录后继续 Prompt B，不要再次要求用户确认。

## 全局硬规则

- 远端约定：GitLab 为 `gitlab`，GitHub 为 `github`；分支为 `gitlab/dev`、`gitlab/main`、`github/main`。
- `github/main` 只能普通 push，禁止 force push。
- 任何 `git push github ...` 都必须逐次暂停并等用户明确回复 `确认执行 GitHub push`。
- `gitlab/main` 推送不设人工门禁；可以直接推送或强制对齐，推荐用 `--force-with-lease` 避免覆盖并发更新。GitHub push 门禁只适用于 `git push github ...`。
- `gitlab/dev` 必须通过 GitLab CR/MR 合入，禁止直接推送到 `dev`。
- `main` 必须线性连续：不要 squash，不要向 `main` 制造 merge commit。
- `-Xours` 不是“保留 ours 整棵树”：它只偏向冲突块，非冲突新增仍会进入。Prompt A 禁止用 `-Xours` 或 `-s ours` merge `github/main`。需要吸收外部代码时只 cherry-pick 已识别的 GitHub PR commit。
- 发现密钥、内部链接、私有路径、敏感内容时，立刻停止；不要 push，不要输出 secret 原文。
- 不要运行需要 GPU 的代码或测试。
- 已跟踪文件有本地改动时停止；只有未跟踪文件时，记录路径并继续原流程。不要删除、stage、stash、clean 未跟踪文件。
- 如果缺少 `gitleaks`，先读 [references/gitleaks.md](references/gitleaks.md)，确认安装方案后再继续。

## 工作区检查

每次预检查都按这个规则判断：

```bash
git status --porcelain
git status --porcelain --untracked-files=no
```

- 第二条命令有输出：说明已跟踪文件有改动，停止并输出文件列表。
- 只有 `??` 未跟踪文件：记录文件列表，继续同步。
- 后续 `checkout`、`merge`、`cherry-pick` 如果因为未跟踪文件会被覆盖而失败，停止并报告冲突路径；不要自动清理。

## GitHub Push 门禁

任何推送到 `github/main` 前，必须先完成 GitHub Actions main push 门禁。普通功能分支 / 验证分支 push 只用于触发 CI，不等同于允许推送 `github/main`。

### GitHub Actions main push 门禁

`gh workflow run` 只会运行 GitHub 上已经 push 的代码，不会包含本地未提交或未 push 的改动。`ci.yml` 目前只能触发整个 workflow，不能只跑其中一个 job，除非 workflow 自己添加 inputs 控制。

在准备 `git push github HEAD:refs/heads/main` 前，必须：

```bash
test -z "$(git status --porcelain --untracked-files=no)"
SOURCE_SHA=$(git rev-parse HEAD)
VALIDATE_BRANCH=sync/validate-github-main-$(git rev-parse --short HEAD)
git push github HEAD:refs/heads/$VALIDATE_BRANCH
gh workflow run ci.yml -R redai-infra/Relax --ref $VALIDATE_BRANCH
gh run list -R redai-infra/Relax --workflow ci.yml --branch $VALIDATE_BRANCH --limit 5
gh run watch <run-id> -R redai-infra/Relax
gh run view <run-id> -R redai-infra/Relax --json status,conclusion,headSha,url
```

门禁规则：

- `headSha` 必须等于 `SOURCE_SHA`，否则该 CI 结果不能作为本次 `github/main` push 凭证。
- `conclusion` 必须是 `success`，且至少包含 `Pre-commit Checks`、`Lint`、Python 测试矩阵全部成功。
- 如果失败，先查看失败日志：

```bash
gh run view <run-id> -R redai-infra/Relax --log-failed
```

- 如果需要改代码：本地修复、提交、push 到同一个验证分支或新验证分支后，重新 `gh workflow run ci.yml`。不要用旧 run 证明新代码。
- 如果确认是瞬时失败且代码未变，可以只重跑失败 job：

```bash
gh run rerun <run-id> -R redai-infra/Relax --failed
```

- 在 CI 全绿前禁止执行 `git push github HEAD:refs/heads/main`。

### 最终 GitHub push 暂停

GitHub Actions main push 门禁全绿后，任何 `github/main` push 前都要输出并暂停：

```text
准备执行 GitHub push：
<完整 git push github ... 命令>

source ref / SHA: <...>
target ref / 当前 SHA: <...>
fast-forward: <yes|no|not-applicable>
安全检查: <duplicate-def / F811 / gitleaks 结果>
GitHub Actions: <ci.yml run url> / <success> / <headSha>

请回复：确认执行 GitHub push
```

只有用户回复完全匹配的 `确认执行 GitHub push` 后，才执行该条 push。每一条 GitHub push 都要单独确认，不能一次确认覆盖后续 push。

## Cherry-Pick 审计门禁

Prompt A / Prompt B 中的 cherry-pick 序列完成后，必须在推送前运行：

```bash
git diff --stat <base>..HEAD
git diff --name-only <base>..HEAD
PY_FILES=$(git diff --name-only <base>..HEAD -- '*.py')
if [ -n "$PY_FILES" ]; then
  python skills/sync-github/scripts/check_duplicate_defs.py $PY_FILES
  ruff check --select F811 $PY_FILES
fi
pre-commit run gitleaks --all-files || gitleaks dir . --log-level warning --report-format csv --report-path -
```

如果 diff 出现重复 helper、重复 top-level 定义、意外大块搬移、或无法解释的新增，停止并说明，不要推送。

## 结束标准

- Prompt A 创建或确认无需创建 `github/main -> dev` CR 后必须停止，等待云效 CR 合入。
- Prompt B 推送 GitHub 后必须验证 `git rev-parse gitlab/main` 与 `git rev-parse github/main` 完全一致，并输出审计信息；如果停在 GitHub push 门禁，则明确说明两边暂时不一致是等待用户确认导致的。
