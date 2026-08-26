# GDPO 示例：correctness + format 双奖励

本目录是 GDPO（[arXiv 2601.05242](https://arxiv.org/abs/2601.05242)）的最小可运行示例，用 Qwen3-0.6B 单卡在 GSM8K 上训练。

## 为什么需要 GDPO

奖励函数 `reward_gdpo.py` 返回两个分量：

- `correctness` —— `<answer>` 标签里的答案是否正确（0 或 1）
- `format` —— 输出是否同时带 `<think>` 与 `<answer>`（0、0.5 或 1）

这两个分量会**不同步**：模型可能答对但没按格式输出，也可能格式完美但答错。

GRPO 把它们**加起来**再做一次组内归一化。这一步会丢掉两类信息。

**一、分量之间的相关结构。** GRPO 对和做组内标准化，每一组的输出恒为单位方差——无论这组的两个分量是彼此印证还是互相矛盾。GDPO 先让每个分量各自单位方差再加权求和，组合结果的方差是

```
Var = Σ wᵢ² + 2 Σᵢ<ⱼ wᵢwⱼ ρᵢⱼ        等权重两分量时 = 2 + 2ρ
```

**由分量间的相关系数 ρ 决定**：

| ρ    | 含义                   | 组合信号          |
| ---- | ---------------------- | ----------------- |
| → +1 | 两个分量指向同一批样本 | 增强（实测 2 倍） |
| 0    | 互不相关               | √2 倍             |
| → −1 | 两个分量互相矛盾       | 减弱，极限时归零  |

第三步的 batch 白化是**跨组**的，所以这个组间强度差异会保留到最终 advantage。GRPO 看到的则是所有组都一样强。

**二、分量之间的尺度差异。** `correctness ∈ {0,1}` 与一个取值上百的 `format`（或论文实验里的响应长度）相加时，和的方差几乎全部来自大尺度那一维，GRPO 的方向就由它单独决定。GDPO 先让每个分量单位方差，权重才真正表达「相对重要性」而不是量纲。极端一点：`correctness=[1,1,0,0]`、`format=[0,100,200,300]`（两者排序相反）时，GRPO 给答错的长响应最高 advantage，GDPO 不会。

## GDPO 不能做什么

**各分量在组内恰好加和为常数时，GDPO 也救不回来。** 这正是上表 `ρ = −1` 那一行：若 `correctness + format ≡ C`，则 `format = C − correctness`，两者标准化后恒有 `z_format = −z_correctness`，**等权重下完全抵消为零**——与 GRPO 得到同样的结果。

这一点本文档此前写反了，说这正是 GDPO 的优势场景。它不是：那是 `ρ = −1`，是组合方差 `2 + 2ρ` 恰好取零的极端。只有不等权重（如 `--gdpo-reward-weights 2.0 1.0`）能打破这个平衡。

（另外，若两个分量在组内**都**恒定，GDPO 与 GRPO 一样返回零，不会无中生有。）

## 运行

```bash
export MODEL_DIR=/path/to/models      # 需含 Qwen3-0.6B
export DATA_DIR=/path/to/data         # 需含 gsm8k/train.jsonl
export EXP_DIR=/path/to/experiments

bash examples/gdpo/run-qwen3-0.6B-1xgpu-gdpo.sh
```

### 数据要求

每条 prompt 需要**自带**格式要求，否则基座模型不会产出 `<think>`/`<answer>` 标签，`format` 分量会恒为 0（组内塌缩），GDPO 就退化成只看 `correctness`。准备数据时给 question 追加一句即可：

```python
instruction = (
    "\n\nThink step by step inside <think> </think> tags, then give only the "
    "final number inside <answer> </answer> tags."
)
df["question"] = df["question"] + instruction
```

**不要用 `--system-prompt` 代替**：`relax/utils/data/data_utils.py:181` 把 system message 的 content 构造成多模态 list（`content: [{"type": "text", ...}]`），Qwen3-0.6B 这类纯文本 chat template 渲染时会报
`TypeError: can only concatenate str (not "list") to str`。这是既有的框架限制，与 GDPO 无关。

## 参数说明

```bash
GDPO_ARGS=(
   --advantage-estimator gdpo
   --gdpo-reward-keys correctness format   # 独立归一化的分量，至少两个
   --gdpo-reward-weights 1.0 1.0           # 可省略，默认全 1
   --custom-rm-path examples.gdpo.reward_gdpo.reward_func
   --reward-key score                      # 必填：metrics 与 raw_reward 用的标量
   --n-samples-per-prompt 8                # 必须 ≥ 2
)
```

`--gdpo-reward-weights` 乘的是**归一化之后**的 advantage，不是原始 reward。经过第一步各分量已经是单位方差，所以权重表达的是相对重要性，而不是分量的量纲——把 `format` 的取值范围从 `[0,1]` 改成 `[0,100]` 基本不改变训练结果。

说「基本」而不是「完全」，是因为第一步除的是 `std + 1e-4` 而不是 `std`。当某分量的组内标准差本身就落到 `1e-4` 量级时，这个加性 epsilon 会随量纲变化而改变阻尼比例，缩放就不再是严格等价的。二值奖励（std≈0.4）离这个区间很远，连续奖励（如响应长度）则可能撞上。

## 换成自己的奖励

改 `reward_gdpo.py` 的 `compute_gdpo_reward`，返回的 dict 需要包含 `--gdpo-reward-keys` 列出的全部 key，外加 `--reward-key` 指定的那个标量。

分量缺失、非数值、bool、NaN/Inf 都会直接报错。这是有意的：静默填 0 会让一个坏掉的奖励函数看起来像是「这一维恰好塌缩了」，训练照跑，问题要很久以后才暴露。

## 已知偏差

1. **第三步的 batch 边界（已正确处理）**。调用方为效率会先合并多个训练批再调用 advantage，但 `_whiten_by_segment` 用 `mini_batch_sizes` 把它们切回**每个 optimizer 训练批各自白化**，因此 `num_rollout_minis > 1` 时仍对齐论文 Eq. 6，**不要求** `rollout_batch_size × n_samples_per_prompt == global_batch_size`。本脚本把 `4 × 8` 与 `--global-batch-size 32` 设成相等只是让例子最简单，并非必需——把 `--global-batch-size` 改成 16 就会跑出两段。跨 DP 的 all-reduce 保证统计量覆盖全部 rank。**`--fully-async` 会在参数校验阶段被拒绝**——那条路径的切片可能小到只有一个样本，白化输出恒为 0。

   调用方**必须**提供这份切分（`rollout_mini_local_sample_counts`）。缺失时 GDPO 直接报错，不回退到「整个 rollout 白化一次」：那不是同一个目标的粗糙版本，而是另一个目标——`test_merging_the_batches_would_flip_signs_not_just_rescale` 里 8 个样本有 4 个符号翻转，而 loss、grad_norm、advantage 均值全都正常。

2. **单个奖励时 GDPO 不等于 GRPO**。step1 除以 `std_g + 1e-4`、GRPO 除以 `std_g + 1e-6`，各组 `std_g` 不同 → 尺度因子逐组不同，step3 还会再做一次 batch 白化，所以不是「差一个正标量」那么简单。要 GRPO 语义就用 `--advantage-estimator grpo`。

3. **`--n-samples-per-prompt 2` 时幅度信息丢失**：任意两个不同值标准化后恒为 ±0.7071。示例用 8 就是为了避开这一点。

4. **恒和分量的数值假信号（未解决，如实记录）**。若两个分量恰好满足 `r₂ = C − r₁`，数学上应完全抵消为零（见上文「GDPO 不能做什么」）。这里的标准化在这种输入下是**病态**的：它要除以一个接近零的 std，相对误差被放大 `max|x| / std` 倍。

   奖励曾经在进入归一化前被 cast 成 float32，而 `C − r` 这种值不一定能被 float32 精确表示——两者相加仍舍回 `C`（看起来恒和），各自却已偏离。残差量级约 `1e-1`，**比信号本身还大**，第三步再除以同量级的 batch 标准差，输出就是 O(1) 的 advantage：一个本无信号的组拿到了方向由舍入决定的梯度。实测一组和为 `308.95172119140625` 的奖励，最终 advantage 为 `[-0.5770, 1.1539, -0.5770]`。

   **分量全链路改成 float64**（`extract_reward_components`）之后，这个特定输入的残差降到 `1e-10` 量级，第三步分母里的 `GDPO_EPS = 1e-4` 钳住放大倍数，输出降到 `1e-6`。但**问题没有被解决，只是被推远了**：残差大致是 `ulp(C) / 组内展布`，随基数增长。同样的构造在 `C = 1e13`、或两列都很大且跨 binade 时，残差的 std 超过 `GDPO_EPS`，钳制失效，输出回到 **O(1)**。

   曾经有两个机制试图挡住它，都已移除：

   - **`combine_group` 里的噪声地板**（按 `8 · Σ|wₖ|·noiseₖ` 整组置零）。它摧毁真实信号：两分量 `base = 4.05e13` 时地板 0.148 而真实信号 0.033；地板对分量求和而任何逐列噪声度量取 max，16 个各自干净的分量能把它推到 0.82。而且它修改的是训练值本身。
   - **一个相对幅度判据**（`|Σwz|` 相对 `Σ|wₖ|·max|zₖ|` 低六个数量级即判为残差）。它错得更根本：分子正比于权重之**差**、分母正比于权重的**大小**，所以它测的是权重配置而非数据；`G = 2` 时精确退化为 `|w₁−w₂|/(|w₁|+|w₂|)`，与任何奖励值无关。实测它在两个方向同时判反——扔掉一个最终 advantage 0.43 的真信号，放行一个 1.08 的纯舍入结果。这不是阈值问题：`G ≥ 3` 时中心化子空间至少二维，可构造 `z₂ = -z₁ + δu`（`u ⊥ z₁`，δ 任意小），所以真实信号的比值没有正下界。

   所以现在的状态是：**`combine_group` 严格返回 Eq. 7，恒和组带着它的舍入残差进训练，没有任何机制识别它**。实测到达 optimizer 的量级（`[C−δ, δ]`，δ∈{0.1,0.2,0.3,0.7}，等权）：

   | C    | 自己独占一个白化单元 | 与 8 个健康组共享一个 |
   | ---- | -------------------- | --------------------- |
   | 1e8  | 2.3e-4               | 2.7e-8                |
   | 1e9  | 2.6e-3               | 3.1e-7                |
   | 1e11 | 2.0e-1               | 3.3e-5                |
   | 1e13 | 1.2                  | 5.2e-3                |

   **第二列要连着 C 一起读。** 本文档早先的版本只引了它 `C=1e9` 那一格、写成「混批约 1e-7」，读起来像个普遍上界——不是。与健康组共享白化单元只是把残差除以**它们的**标准差（这里是几百倍的常数因子），不改变它随 C 的增长；C=1e13 时共享单元里仍有 5e-3。

   而且「白化单元」不等于「整个 rollout」：`_whiten_by_segment` 按**训练批**分别白化，所以只有当健康组落进同一个训练批时才有这个除法。退化组独占一段时，直接回到第一列。没有任何机制保证这种混合，它取决于 rollout 怎么被切分。

   GRPO 没有这个问题：它归一化的是**和**，而和在 float32 下确实恒定，塌缩检测会直接置零。

## 冲突项

`--normalize-advantages`、`--custom-reward-post-process-path`、`--agentic-custom-advantage-path` 和 `--fully-async` 都不能与 GDPO 同用，参数校验阶段会直接报错。第一个会造成双重白化；第二、三个都会赶在归一化器之前从 `post_process_rewards` 返回，导致 GDPO 的前两步被静默跳过（这两个由 `AlgorithmSpec.allows_reward_post_process_hooks` 一起把守）；第四个的统计窗口可能小到只剩一个样本。

配 `--dynamic-sampling-filter-path` 时**不冲突**：内置的 `check_reward_nonzero_std` 会算出 GDPO 前两步的组合结果，再在**训练实际使用的 float32** 上判 `min != max`——与单奖励算法那条分支问的是同一个问题，只是维度从 1 变成 K。判据不含任何容差：权重把某分量静音、或两个分量精确抵消，组合结果就是精确的零，能判出来；而近似抵消判不出来，那种组会带着上面说的舍入残差进训练。只有指向**自定义** filter 时才会警告——那种 filter 若只看 `--reward-key` 标量，就会丢掉只在其它分量里有信号的组。
