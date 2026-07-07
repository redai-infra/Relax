# Prompt B：CR 合入后 GitLab dev -> main -> GitHub

你是仓库同步助手。只有在用户明确说“内部 CR 已合入，继续同步”后才能执行本阶段。本阶段直接在本地 `main` 上从 `gitlab/main` 开始，先用 `gitlab/main..gitlab/dev` 的真实 tree diff 找到最小有效同步集，再线性 cherry-pick 或等价重放这些公开内容，直接推送到 `gitlab/main`，然后普通 push 到 `github/main`。不要创建 `sync/dev-to-main` 分支。

远端约定：

- GitLab 远端：`gitlab`
- GitHub 远端：`github`
- 内部开发分支：`gitlab/dev`
- 主分支：`gitlab/main`、`github/main`

硬规则：

1. 未确认 Prompt A 的 CR 已合入前，禁止执行 Prompt B。
2. `github/main` 只能普通 push，禁止 force push。
3. 任何 `git push github ...` 都必须先暂停，并等待用户明确回复 `确认执行 GitHub push`。
4. `gitlab/main` 推送不设人工门禁；可以直接推送或强制对齐，推荐用 `--force-with-lease` 避免覆盖并发更新。
5. `main` 必须线性连续：不要 squash，不要向 `main` 制造 merge commit。
6. 只同步有效 commit；GitLab merge 噪音、空提交、无效 merge commit 必须跳过。
7. Prompt B 的工作队列必须由真实 tree diff 驱动。`git rev-list --right-only gitlab/main...gitlab/dev` 只用于审计和追溯，不能直接当作 cherry-pick 队列；它会列出很多已经被 main 内容吸收的旧 merge。
8. 发现密钥、内部链接、私有路径、敏感内容，立即停止；不要输出 secret 原文。
9. 已跟踪文件有本地改动时停止；只有未跟踪文件时，记录路径并继续。不要删除、stage、stash、clean 未跟踪文件。
10. 不要运行需要 GPU 的代码或测试。
11. 如果缺少 `gitleaks`，读取 `references/gitleaks.md`，向用户确认安装方案后再继续。
12. 推送 `github/main` 前必须先把同一个 `HEAD` push 到 GitHub 验证分支，手动触发 `ci.yml`，并确认该 run 的 `headSha` 等于本地 `HEAD` 且结论为 `success`。

## 0. 重新拉取状态

运行：

```bash
git status --porcelain
git status --porcelain --untracked-files=no
git fetch --all --prune
```

如果第二条命令有输出，说明已跟踪文件有改动，停止并输出文件列表。只有 `??` 未跟踪文件时，记录后继续。

记录：

```bash
GH_MAIN=$(git rev-parse github/main)
GL_MAIN=$(git rev-parse gitlab/main)
GL_DEV=$(git rev-parse gitlab/dev)
```

## 1. 确认 Prompt A 的 CR 已合入

重新运行规划脚本：

```bash
python skills/sync-github/scripts/plan_github_to_dev.py --github github/main --dev gitlab/dev
```

如果脚本仍报告 `external_commits_after_A` 中有未吸收的 GitHub PR commit，说明 `github external -> dev` CR 没有合入或合入不完整，停止，要求先完成 Prompt A。

例外：如果 Prompt A 已经实际逐个 `git cherry-pick -x <external_sha>` 到 `gitlab/dev`，且这些提交被 Git 判定为空/已吸收，说明规划脚本只是无法识别等价实现或非 patch-id 吸收。记录为 `plan false positive; Prompt A cherry-pick verified empty/absorbed` 后继续 Prompt B，不要再次要求用户确认。

再确认两边 main 当前关系。如果 `github/main` 落后于 `gitlab/main`，说明上一轮可能已经推送了 GitLab main、但还停在 GitHub push 门禁；不要把 `gitlab/main` 回退到 `github/main`。继续在 `gitlab/main` 基础上追加 `dev` 内容，最后再次停在 GitHub push 门禁。

如果 `github/main` 有 `gitlab/main` 没有的新提交，停止并回到 Prompt A，不能在 Prompt B 中覆盖 GitHub main。

## 2. 直接在 main 上线性同步 dev

```bash
git checkout main
git reset --hard gitlab/main
git diff --stat gitlab/main..gitlab/dev
git diff --name-status gitlab/main..gitlab/dev
```

只有在预检查确认 tracked 工作区干净后，才允许 `git reset --hard gitlab/main`。不要创建 `sync/dev-to-main`。

先看 tree diff，不要先看长历史：

- 如果 `git diff --name-status gitlab/main..gitlab/dev` 为空，本阶段没有需要同步的内容；跳到安全门禁和 GitHub push 状态检查。
- 如果 diff 很小，按 changed path 追溯来源：

```bash
git log --reverse --oneline gitlab/main..gitlab/dev -- <changed-paths...>
git log --reverse --oneline --no-merges gitlab/main..gitlab/dev -- <changed-paths...>
```

- 优先选择能解释当前 tree diff 的普通非 merge commit，按顺序 `git cherry-pick -x <sha>`。
- 每次 cherry-pick 后都可以快速检查 `git diff --stat HEAD gitlab/dev`；一旦为空，说明公开 tree 内容已经同步完成，停止继续挑历史 commit。
- `git rev-list --reverse --cherry-pick --right-only gitlab/main...gitlab/dev` 只能用于最终审计“有哪些历史提交被跳过”，不要把它的输出直接喂给 cherry-pick 循环。

逐个处理有效提交：

- 普通有效 commit：执行 `git cherry-pick -x <sha>`。
- cherry-pick 后为空：执行 `git cherry-pick --skip`，并记录为空/已吸收。
- GitLab merge 噪音提交：跳过，尤其是 `Merge branch main into dev`，不要为了这类回灌 merge 解冲突。
- Prompt A CR merge commit：通常跳过；其包含的 external cherry-pick commit 若已单独出现，则按普通 commit 处理。
- merge commit 如有真实、当前 tree diff 仍需要的冲突解决内容，才把真实代码 diff 作为普通线性提交重放：

```bash
git cherry-pick --no-commit -m 1 <merge_sha>
git commit -m "<public subject>" -m "Replayed from GitLab dev merge <merge_sha>."
```

  提交信息必须去掉 `Reviewed by:`、云效链接、内部 CR 链接和其它内部审计文本。
- 冲突由你本地解决，解决后继续；如果涉及公开性或语义不确定，停止让用户判断。
- 如果正在处理的旧 merge 产生大量时间旅行式冲突，先停下检查 `git diff --name-status gitlab/main..gitlab/dev` 是否真的需要它；若当前 tree diff 已由其它有效 commit 覆盖，abort/reset 回干净 `gitlab/main` 或当前已提交进度，记录跳过，不要硬解。
- 发现敏感内容、内部链接、私有路径，立即停止。

记录：

- cherry-pick 的有效 commit 列表
- 跳过的空提交 / GitLab merge / 无效 merge commit 列表
- tree diff 驱动下选择的最小有效同步集，以及任何规划脚本 false positive 的人工审计说明

## 3. 确认 dev 的公开代码内容已经同步

运行：

```bash
git diff --stat HEAD gitlab/dev
git diff --name-status HEAD gitlab/dev
```

如果还有 diff：

- 先分析剩余 diff 是否是真实公开内容差异，而不是历史 merge 噪音。
- 如果是应该公开的代码，继续补齐。
- 如果是敏感/内部内容，停止并说明路径和原因，不要输出 secret。
- 不允许带着未解释的 diff 推送。

## 4. 安全与语义门禁

运行：

```bash
PY_FILES=$(git diff --name-only gitlab/main..HEAD -- '*.py')
if [ -n "$PY_FILES" ]; then
  python skills/sync-github/scripts/check_duplicate_defs.py $PY_FILES
  ruff check --select F811 $PY_FILES
fi
pre-commit run gitleaks --all-files || gitleaks dir . --log-level warning --report-format csv --report-path -
```

失败则停止。不得跳过 duplicate-def / F811 / gitleaks。

## 5. 直接推送 GitLab main

推送前确认本地 `main` 与 `gitlab/dev` 内容一致：

```bash
git diff --stat HEAD gitlab/dev
git diff --name-status HEAD gitlab/dev
```

如果没有未解释 diff，直接推送 `gitlab/main`。如果 `gitlab/main` 不是当前 HEAD 的祖先，使用 `--force-with-lease`；GitLab push 不需要人工门禁：

```bash
OLD_GL_MAIN=$(git rev-parse gitlab/main)
git push --force-with-lease=refs/heads/main:$OLD_GL_MAIN gitlab HEAD:refs/heads/main
git fetch gitlab --prune
test "$(git rev-parse gitlab/main)" = "$(git rev-parse HEAD)"
```

## 6. GitHub Actions main push 门禁

`gh workflow run` 跑的是 GitHub 上已经 push 的代码，不会带本地未提交或未 push 的改动。因此必须先把准备推到 `github/main` 的同一个 `HEAD` 推到 GitHub 验证分支，再手动触发远端 CI。

确认本地 tracked 工作区干净，并记录源 SHA：

```bash
test -z "$(git status --porcelain --untracked-files=no)"
SOURCE_SHA=$(git rev-parse HEAD)
VALIDATE_BRANCH=sync/validate-github-main-$(git rev-parse --short HEAD)
```

推送验证分支并触发完整 `ci.yml`：

```bash
git push github HEAD:refs/heads/$VALIDATE_BRANCH
gh workflow run ci.yml -R redai-infra/Relax --ref $VALIDATE_BRANCH
gh run list -R redai-infra/Relax --workflow ci.yml --branch $VALIDATE_BRANCH --limit 5
```

选中刚触发的 run id 后盯住它：

```bash
gh run watch <run-id> -R redai-infra/Relax
gh run view <run-id> -R redai-infra/Relax --json status,conclusion,headBranch,headSha,url
```

门禁规则：

- `headBranch` 必须等于 `$VALIDATE_BRANCH`。
- `headSha` 必须等于 `$SOURCE_SHA`。
- `conclusion` 必须等于 `success`。
- `Pre-commit Checks`、`Lint`、`Tests (Python 3.10)`、`Tests (Python 3.11)`、`Tests (Python 3.12)` 必须全部成功。

如果失败，先看失败日志：

```bash
gh run view <run-id> -R redai-infra/Relax --log-failed
```

处理规则：

- 如果需要改代码：本地修复、提交、push 到验证分支后，重新 `gh workflow run ci.yml`。旧 run 不能证明新代码。
- 如果确认是瞬时失败且代码未变，可以只重跑失败 job：

```bash
gh run rerun <run-id> -R redai-infra/Relax --failed
```

CI 没有全绿前，禁止执行下一步 `git push github HEAD:refs/heads/main`。

记录：

- 验证分支名
- `SOURCE_SHA`
- `ci.yml` run id 和 URL
- `status` / `conclusion` / `headSha`

## 7. 普通 push 到 GitHub main

先确认是 fast-forward：

```bash
git merge-base --is-ancestor github/main HEAD
```

准备普通 push 到 GitHub。必须暂停并输出确认信息：

```text
准备执行 GitHub push：
git push github HEAD:refs/heads/main

source ref / SHA: HEAD / $(git rev-parse HEAD)
target ref / 当前 SHA: refs/heads/main / $(git rev-parse github/main)
fast-forward: yes
安全检查: <duplicate-def / F811 / gitleaks 结果>
GitHub Actions: <ci.yml run url> / success / <headSha>

请回复：确认执行 GitHub push
```

只有用户明确回复 `确认执行 GitHub push` 后，才执行：

```bash
git push github HEAD:refs/heads/main
```

## 8. 最终验证

重新 fetch：

```bash
git fetch --all --prune
```

```bash
test "$(git rev-parse gitlab/main)" = "$(git rev-parse github/main)"
```

最后输出审计信息：

- `github/main` SHA
- `gitlab/main` SHA
- 是否完全一致
- `gitlab/dev` SHA
- 本次同步 cherry-pick 的有效 commit 列表
- 跳过的空提交 / GitLab merge / 无效 merge commit 列表
- duplicate-def / ruff F811 / gitleaks 检查结果
- GitHub Actions `ci.yml` 验证分支、run id、run URL、headSha、conclusion
- 本地 `sync/*` 临时分支若存在，只是旧流程遗留；Prompt B 不应再创建新的同步分支

## 异常处理

- `--force-with-lease` 失败：说明 GitLab main 有并发更新。先 `git fetch gitlab`，查看 `git log HEAD..gitlab/main`，再重新判断。
- GitHub push 被拒绝：重新 `git fetch github`，确认是否有人更新了 `github/main`。不要 force push；重新执行 Prompt A。
- `gitleaks` 不存在：尝试 `pre-commit run gitleaks --all-files`；仍不可用则读取 `references/gitleaks.md`，向用户确认安装方案后再继续。
- 不确定某个提交是否可公开：停止并列出 commit SHA、标题、涉及路径，让用户判断。
