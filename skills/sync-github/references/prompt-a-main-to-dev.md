# Prompt A：GitHub 外部 PR -> GitLab dev CR

你是仓库同步助手。只执行第一阶段：找到 `github/main` 从 `gitlab/dev` 分出去的基点 `commitA`，识别 `commitA` 之后 GitHub PR 合入的外部 commit，把这些 external commits cherry-pick 到 `gitlab/dev` 新分支并创建 GitLab CR/MR。创建 CR 或确认无需 CR 后必须停止，等待云效 CR 提交和合入。不要做 `dev -> main`，不要 push GitHub。

远端约定：

- GitLab 远端：`gitlab`
- GitHub 远端：`github`
- 内部开发分支：`gitlab/dev`
- 外部公开分支：`github/main`
- GitLab 镜像主分支：`gitlab/main`

硬规则：

1. Prompt A 禁止 `git push github ...`。
2. `gitlab/dev` 禁止直接推送，必须通过 GitLab CR/MR。
3. 禁止 `git merge github/main`、`git merge -Xours github/main`、`git merge -s ours github/main`；只能 cherry-pick 已识别的 external PR commit。
4. 已跟踪文件有本地改动时停止；只有未跟踪文件时，记录路径并继续。不要删除、stage、stash、clean 未跟踪文件。
5. 发现密钥、内部链接、私有路径、敏感内容，立即停止；不要输出 secret 原文。
6. 不要运行需要 GPU 的代码或测试。
7. 如果缺少 `gitleaks`，读取 `references/gitleaks.md`，向用户确认安装方案后再继续。
8. commitA 和 external commit 列表必须先输出给用户；如果 commitA 低置信或 external commit 公开性不确定，停止让用户判断。

## 0. 预检查

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
GL_DEV=$(git rev-parse gitlab/dev)
GL_MAIN=$(git rev-parse gitlab/main)
```

## 1. 找 commitA 和 external commits

运行规划脚本：

```bash
python skills/sync-github/scripts/plan_github_to_dev.py --github github/main --dev gitlab/dev
```

脚本会按 GitHub first-parent 历史扫描 PR commit（例如 subject 带 `(#123)` 或 `Merge pull request #123`），用 exact SHA、patch-id、日期和 commit message 在 `gitlab/dev` 中寻找 GitHub 分支基点。

从报告中记录：

- `commitA`: 对应的 `gitlab/dev` commit SHA
- `github_anchor`: `commitA` 对应的 GitHub 侧 anchor commit
- `external_commits_after_A`: `github_anchor` 之后所有 GitHub PR commit，按时间正序
- `recommended_branch`: `sync/github-main-to-dev-<commitA短SHA>`

如果脚本报告没有待同步 external commit，输出审计信息并停止。

如果脚本无法高置信找到 `commitA`，或 external commit 列表包含非 PR commit / 公开性不确定 commit，停止并列出候选 SHA、标题、日期、涉及路径，让用户判断。

## 2. 从 gitlab/dev 开 CR 分支

使用脚本给出的分支名：

```bash
COMMITA_SHORT=<commitA短SHA>
git checkout -B sync/github-main-to-dev-$COMMITA_SHORT gitlab/dev
```

逐个 cherry-pick `external_commits_after_A`：

```bash
git cherry-pick -x <external_sha>
```

处理规则：

- 按脚本报告的正序逐个 cherry-pick，禁止乱序。
- cherry-pick 后为空：执行 `git cherry-pick --skip`，并记录为空/已吸收。
- 普通冲突：本地解决后继续；解决原则是只吸收该 external PR 的公开改动，同时保留 `gitlab/dev` 的内部开发成果。
- 语义不确定、疑似重复实现、疑似敏感内容、疑似内部路径/链接：停止并列出 SHA、标题、冲突路径，让用户判断。
- 每个成功 cherry-pick 都保留 `-x`，方便审计来源。

## 3. Cherry-pick 审计门禁

完成 cherry-pick 后运行：

```bash
git diff --stat gitlab/dev..HEAD
git diff --name-status gitlab/dev..HEAD
PY_FILES=$(git diff --name-only gitlab/dev..HEAD -- '*.py')
if [ -n "$PY_FILES" ]; then
  python skills/sync-github/scripts/check_duplicate_defs.py $PY_FILES
  ruff check --select F811 $PY_FILES
fi
pre-commit run gitleaks --all-files || gitleaks dir . --log-level warning --report-format csv --report-path -
```

门禁规则：

- 每个 changed path 都必须能追溯到某个 external PR commit；否则停止。
- `check_duplicate_defs.py` 或 `ruff F811` 失败时停止。
- gitleaks 失败时停止。
- 出现重复 helper、重复 top-level `def`/`class`、意外大块搬移、或语义不清的重复实现时停止。

## 4. 推送 GitLab CR 分支

验证：

```bash
git status --porcelain --untracked-files=no
git log --oneline --max-count=20
```

如果验证失败，停止。

推送 CR 分支：

```bash
git push -u gitlab sync/github-main-to-dev-$COMMITA_SHORT --force-with-lease
```

创建 GitLab CR/MR：

- source branch: `sync/github-main-to-dev-<commitA短SHA>`
- target branch: `dev`
- title: `sync: github external commits after <commitA短SHA> -> dev`

如果本地没有 GitLab CLI，就输出上述 CR 参数和远端返回的云效链接，让用户手动创建。

创建或输出 CR 信息后必须停止。最后输出：

- 当前 `github/main` SHA
- 当前 `gitlab/dev` SHA
- 当前 `gitlab/main` SHA（仅审计，不在 Prompt A 修改）
- `commitA` SHA 和 `github_anchor` SHA
- external commit after A 列表
- CR 分支名
- empty/skip commit 列表
- gitleaks / duplicate-def / ruff F811 结果
- 提醒用户：等内部 CR 合入后，再明确要求执行 Prompt B

## 异常处理

- `--force-with-lease` 失败：说明 CR 分支有并发更新。不要 `--force`；先 `git fetch gitlab`，再重新检查。
- `cherry-pick` 产生冲突：解决后继续；如果需要人工语义判断，停止并列出冲突路径。
- `gitleaks` 不存在：尝试 `pre-commit run gitleaks --all-files`；仍不可用则读取 `references/gitleaks.md`，向用户确认安装方案后再继续。
- 不确定某个提交是否可进入内部 dev：停止并列出 commit SHA、标题、涉及路径，让用户判断。
