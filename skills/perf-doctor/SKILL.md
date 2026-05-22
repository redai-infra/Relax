---
name: perf-doctor
description: Diagnose Relax training launch scripts for misconfigured flags that hurt performance (time/MFU) or waste GPU memory (cards needed). Use when user asks to review/audit/check a training script, mentions "perf doctor", suspects a config is slow or OOM-prone, or wants a sanity check before launching. Produces a two-section markdown report (Performance + Memory) with cited flags, severity, and concrete fixes.
argument-hint: <path-to-launch-script>
---

# perf-doctor

诊断 Relax 训练启动脚本（`scripts/training/**/*.sh`），找出影响 **执行性能（耗时 / MFU）** 或 **显存占用（所需卡量）** 的不合理配置。

## 使用方式

```
/perf-doctor scripts/training/text/run-qwen36-35B-A3B-8xgpu.sh
```

参数：单个启动脚本绝对或相对路径。

## 执行步骤

1. **读取脚本** — 收集 `*_ARGS=( ... )` 数组与 `ray job submit ... train` 行里的所有 `--flag value`。同时 follow `source ${MODEL_CONFIG_DIR}/...` 拿模型架构（dense / MoE、是否 multimodal）。
2. **抽取 context** —
   - 文件名解析：`run-<model>-<size>-<NxgpuY>(-async|-image|-video)?.sh` → 总 GPU 数、节点数、模式、模态
   - flag 解析：TP/PP/CP/EP/ETP、`--colocate` vs `--fully-async`、`--rollout-max-response-len`、`--max-tokens-per-gpu`、`--resource`、`--num-iters-per-train-update`、`--max-staleness`、`--num-data-storage-units`
   - **默认 GPU 假设：H20 96GB**，除非用户在 prompt 里给出别的（A100 80G / H100 80G 等）
3. **加载规则** — 读 `references/rules.md`，逐条判断 applies / borderline / not-applicable
4. **输出报告** — 严格按下方 [输出模板](#输出模板) 渲染

## 触发判断原则

- **不要机械触发**：CPU offload 三件套在 35B-A3B 4×H20 这种边界 case 是必需的；规则 `Skip when` 节里写了什么时候它就是对的，要尊重
- **不要 false positive**：脚本里如果有 `# NOTE(...)` 注释解释为什么开 / 关某个 flag，把它当作有效理由，降级到 info 或跳过
- **借助推理而非穷举**：`references/rules.md` 是知识库不是判定表 — 模型大小 × dtype 估算显存预算、TP×CP×PP 是否合理、async 资源比是否平衡，都要 case-by-case 算

## 输出模板

```markdown
# perf-doctor: <script-name>

**Context:** model=<X> (<dense|MoE>, <text|mm|video>) · GPUs=<N> (<nodes>×<g/n>) · mode=<colocate|fully-async> · TP<x>/PP<x>/CP<x>/EP<x>/ETP<x> · max-resp-len=<X> · GPU=H20 96GB (assumed)

---

## 🚀 Performance findings

### [WARN] R-P0X — <short title>
- **Setting:** `--flag value`（脚本行号或所在 ARGS 组）
- **Cost:** <估算的 MFU / 耗时影响>
- **Fix:** <可直接照抄的 flag 修改>
- **Skip if:** <什么情况下当前设置反而是对的>

(more findings...)

## 💾 Memory findings

### [WARN] R-M0X — <short title>
- **Setting:** `--flag value`
- **Cost:** <显存影响 / 卡量影响>
- **Fix:** <修改建议>
- **Skip if:** <justified condition>

(more findings...)

---

## Summary
- Critical: N · Warn: N · Info: N
- **Top action:** <一句话最该改的>
```

## 严禁

- ❌ 不要 **修改脚本**。只诊断，给文字建议
- ❌ 不要执行训练 / dry-run / benchmark
- ❌ 不要分析日志（那是 `debug-hang` 的事）
- ❌ 不要建议不在 `references/rules.md` 里、且自己不能给出 Relax-specific 依据的"通用 ML 优化技巧"

## Rule catalog

完整规则在 `references/rules.md`。每条规则字段：`Category` / `Severity` / `Trigger` / `Why` / `Fix` / `Skip when`。新规则直接往该文件追加即可，无需改 SKILL.md。
