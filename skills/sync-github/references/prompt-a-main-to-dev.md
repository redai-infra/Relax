# Prompt A：GitHub 外部 PR -> GitLab dev CR

你是仓库同步助手。只执行第一阶段：

1. 从 `gitlab/dev` 最近若干条 commit 的 body 里，收集所有 `(cherry picked from commit <sha>)` 尾巴以及 Prompt A 替身 commit 显式列出的 github SHA，得到「已进 dev 的 github commit 集合」。
2. 在 `github/main` first-parent 链上找到该集合里最新的一条 —— `github_anchor`；其在 `gitlab/dev` 上的对应 commit 即 `commitA`。
3. `github_anchor..github/main` 上按序的 PR commit 就是本轮 external commit 候选。
4. 用 `plan_github_to_dev.py` 交叉核验。
5. 从 `gitlab/dev` 开 CR 分支，把 external commit 逐个 `git cherry-pick -x`，创建 GitLab CR/MR 后必须停止，等待云效 CR 提交和合入。

不要做 `dev -> main`，不要 push GitHub。

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
4. 每个 cherry-pick 都必须带 `-x`；这是下次 Prompt A 找 `github_anchor` 的唯一凭据。如果本轮必须走 squash 替身 commit（`chore(sync): replay github external changes` 一类），body 里必须逐个列出被吸收和被跳过的 github SHA。
5. 已跟踪文件有本地改动时停止；只有未跟踪文件时，记录路径并继续。不要删除、stage、stash、clean 未跟踪文件。
6. 发现密钥、内部链接、私有路径、敏感内容，立即停止；不要输出 secret 原文。
7. 不要运行需要 GPU 的代码或测试。
8. 如果缺少 `gitleaks`，读取 `references/gitleaks.md`，向用户确认安装方案后再继续。
9. `commitA` / `github_anchor` / external commit 列表必须先输出给用户；如果 `github_anchor` 低置信，或 external commit 公开性不确定，停止让用户判断。

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

## 1. 定位 `github_anchor`（`gitlab/dev` 侧已吸收到哪条 github commit）

历次 Prompt A cherry-pick 都强制 `-x`（硬规则 4），所以 `gitlab/dev` 最近提交的 body 里会留下 `(cherry picked from commit <github_sha>)` 尾巴。squash 型替身 commit 也会在 body 里显式罗列它包含或跳过的 github SHA。把这两类 SHA 都收集起来，再落到 `github/main` first-parent 链上找最新的一条，就是 `github_anchor`。

一步到位：

```bash
# 从 dev 最近 200 条 commit body 中提取所有 hex token，过滤成 github/main 上真实存在的 commit
git log gitlab/dev -200 --format='%B' \
  | grep -oiE '\b[0-9a-f]{7,40}\b' \
  | sort -u \
  | while read sha; do
      git rev-parse --verify --quiet "${sha}^{commit}" >/dev/null 2>&1 \
        && git merge-base --is-ancestor "$sha" github/main 2>/dev/null \
        && git rev-parse "$sha"
    done \
  | sort -u > /tmp/dev_ref_gh.txt

# 在 github/main first-parent 链上找到最新命中，这就是 github_anchor
GITHUB_ANCHOR=$(git rev-list --first-parent github/main | grep -Ff /tmp/dev_ref_gh.txt | head -1)
git log -1 --format='github_anchor: %H %s' $GITHUB_ANCHOR
```

`commitA`（`gitlab/dev` 侧对应 commit）通过 `git log --grep` 反查：

```bash
COMMITA=$(git log gitlab/dev --format='%H' --grep="$GITHUB_ANCHOR" | head -1)
git log -1 --format='commitA: %H %s' $COMMITA
```

若 `--grep` 命中多个，取时间最新的那一条；若命中 0 个，说明 `github_anchor` 是被 squash 替身 commit 收进 dev 的，此时把 `commitA` 记为「最近一个包含该 SHA 的 dev commit」，实际操作里可以直接用 `git log --grep` 拿短 SHA 前缀再试：

```bash
git log gitlab/dev --format='%H %s' --grep="${GITHUB_ANCHOR:0:12}" | head -5
```

初步 external commit 候选：

```bash
git log --reverse --first-parent --format='%H %s' $GITHUB_ANCHOR..github/main \
  | awk '/\(#[0-9]+\)\s*$/ || /^[0-9a-f]+ Merge pull request #[0-9]+/'
```

边界处理：

- 找不到任何命中：仓库首次跑 Prompt A，或早期 cherry-pick 没加 `-x` 且没在 body 里留 SHA。停止，让用户手工指定 `github_anchor`。
- `github_anchor` 落在 `github/main` first-parent 链之外（例如它是某个 PR 分支的中间 commit）：往上溯到最近的 first-parent 节点作为 anchor，并在报告里写清楚。
- 一次收集到多个候选 anchor：始终取 first-parent 链上最新那一条。

## 1.5. 用规划脚本交叉核验

```bash
python skills/sync-github/scripts/plan_github_to_dev.py --github github/main --dev gitlab/dev
```

脚本用 exact SHA、patch-id、subject+date 三条独立通道判断 GitHub PR commit 是否已经在 dev 里，然后自己算出一个 `commitA` 和 `external_commits_after_A`。

对比脚本输出和第 1 步的手工结果：

- `commitA` 一致 → 高置信，直接采用第 1 步结果，跳过下面 fallback。
- `commitA` 不一致 → 打印两边差异让用户判断；通常以第 1 步（cherry-pick 尾巴）为准，除非 `-x` 尾巴/body SHA 已被人为删除。
- 第 1 步 fallback（找不到 anchor）→ 采用脚本的 `commitA`；如果脚本 `commitA_match` 是 `subject-date` 且 confidence 低于 `high`，停止让用户确认。

从脚本报告中记录：

- `commitA`: `gitlab/dev` 上的 commit SHA
- `github_anchor`: `commitA` 对应的 GitHub 侧 anchor commit
- `external_commits_after_A`: `github_anchor` 之后所有 GitHub PR commit（时间正序）
- `recommended_branch`: `sync/github-main-to-dev-<commitA短SHA>`

对 `external_commits_after_A` 里每条 commit，标注状态：

- `not-in-dev`：真需要 cherry-pick
- `in-dev:exact-sha:*` / `in-dev:patch-id:*`：已吸收，本轮跳过并记录
- `in-dev:subject-date:*`：仅主题匹配，需要额外 `git show` 对比 diff 内容后再决定跳过还是搬

如果所有 external commit 都是 `in-dev:*`，输出审计信息并停止 —— 本轮无需 CR。

如果 external commit 列表包含非 PR commit / 公开性不确定 commit，停止并列出候选 SHA、标题、日期、涉及路径，让用户判断。

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

- 按第 1.5 步确认的时间正序逐个 cherry-pick，禁止乱序。
- cherry-pick 后为空：执行 `git cherry-pick --skip`，并记录为空/已吸收。
- 如果发现该 external commit 已被内部以 exact SHA、patch-id、commit message、人审等方式等价合入或修正版合入，必须执行 `git cherry-pick --skip`，并记录为已吸收；禁止因为冲突解决后仍有残余 tree diff 就部分重放该 commit。
  - 例：外部 commit 添加 `init_tracking(args)`，但内部已合入并修正为 `serve.start()` 后初始化，应整 commit 跳过，不得把外部较早位置的调用重新带入 CR。
- 普通冲突：本地解决后继续；解决原则是只吸收该 external PR 的公开改动，同时保留 `gitlab/dev` 的内部开发成果。
- 语义不确定、疑似重复实现、疑似敏感内容、疑似内部路径/链接：停止并列出 SHA、标题、冲突路径，让用户判断。
- 每个成功 cherry-pick 都保留 `-x`，方便下次 Prompt A 找 `github_anchor`。
- 若本轮改为 squash 型替身 commit（例如多个外部 commit 大部分互相依赖，独立 cherry-pick 冲突量爆炸），必须在 commit body 里逐一列出被吸收 / 被跳过的 github SHA，格式：

  ```text
  - Include only net-new effective changes from GitHub commits <sha1>, <sha2>, ...
  - Treat previously absorbed or internally fixed commits as whole-commit skips: <sha3>, <sha4>, ...
  ```

  否则违反硬规则 4，下一轮 Prompt A 将无法从 dev 反查出这些 SHA。

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
