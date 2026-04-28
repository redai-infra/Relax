# Relax 论文解读：面向适配工作的阅读笔记

论文：`Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale`

版本：2026-04-15

## 一句话总结

Relax 不是只提出一个新的 RL 算法，而是提出一个面向大规模 post-training 的系统架构。

它的核心主张是：随着训练从 text-only 走向 omni-modal 和 agentic，多模态数据异构、服务故障隔离、rollout/training 异步解耦这三个问题必须一起设计。

Relax 的解法是把 RL 训练拆成独立服务角色，用 TransferQueue 做异步数据总线，用 DCS 做权重同步，再通过一个 `max_staleness` 参数统一 on-policy、near-on-policy、fully-async/off-policy 的执行方式。

## 论文想解决什么问题？

论文认为下一代 RL post-training 有三个连续递进的问题。

### 1. 数据从 text-only 变成 omni-modal

训练样本不再只是 prompt/response 文本，还会包含图像、视频、音频、工具返回、环境状态等。

这些数据有几个特点：

- 大小差异很大，例如高分辨率图像或视频远大于纯文本 token。
- 预处理延迟差异很大，例如视频解码、音频特征提取、图像编码耗时不一样。
- 并行策略不一样，例如 ViT、Whisper、LLM backbone 的切分方式不同。

所以 text-first 框架靠临时 wrapper 接多模态，会变得很脆。

### 2. 大规模训练需要服务级容错

多模态和 agentic rollout 会带来长尾延迟、OOM、NCCL timeout、服务挂死等问题。

如果整个训练是一个 monolithic loop，任何一个角色挂掉都可能导致全局重启。

Relax 的目标是让 actor、rollout、reference、actor_fwd、advantages 等角色可以独立部署、独立扩缩容、独立恢复。

### 3. 训练和 rollout 必须解耦

传统同步 RL 里，训练必须等完整 rollout batch、reference logprob、advantage 等阶段都结束。

如果 rollout 有长尾样本，trainer GPU 会 idle。

Relax 用 TransferQueue 把数据交换改成异步字段级数据流，让下游服务可以在 micro-batch ready 后立刻消费。

## 核心创新点

### 创新 1：Role-Isolated Service Architecture

Relax 把每个 RL role 部署成独立 Ray Serve 服务。

论文里的核心角色包括：

- `actor`：训练策略模型，执行 policy update。
- `rollout`：推理生成样本，通常接 SGLang。
- `reference`：冻结参考模型，只算 reference logprob。
- `actor_fwd`：当前 actor 权重的 forward-only 服务，只算当前策略 logprob。
- `advantages`：根据 reward/logprob 计算 advantage 和 return。
- `critic`：PPO/value-model 场景才需要。

这个设计的收益：

- 故障隔离：某个 role 挂了，不一定要全局重启。
- 独立扩缩容：rollout 慢就扩 rollout，不必扩 actor。
- 生命周期清晰：每个服务自己初始化、健康检查、重启、权重同步。

### 创新 2：DCS 独立权重同步

RL 训练里 actor 更新后，要把新权重同步给 rollout 和 actor_fwd。

Relax 把这个逻辑抽成 Distributed Checkpoint Service。

DCS 负责：

- 发现训练侧和推理侧的 TP/PP topology。
- 协调 actor、rollout、actor_fwd 等角色之间的同步。
- 支持 NCCL/GPU 内同步，也支持 TCP/CPU fallback。

这对异构部署很关键，因为训练侧 Megatron 和推理侧 SGLang 的并行切分常常不同。

### 创新 3：TransferQueue 作为异步数据总线

Relax 不让角色之间直接 RPC 传中间结果，而是通过 TransferQueue 读写字段。

这带来两个关键能力：

- 字段级 decoupling：tokens、reward、logprob、advantage 等字段可以由不同角色在不同时间写入。
- streaming micro-batch：rollout 不必等完整 global batch 生成完，micro-batch 完成后就能给下游消费。

论文强调，这解释了 fully async 模式下 trainer 几乎不等数据。

### 创新 4：用 `max_staleness` 统一训练模式

Relax 把 on-policy/off-policy 的切换抽象成一个 staleness knob。

记：

- `v_t`：训练侧当前 actor 权重版本。
- `v_r`：rollout 样本生成时使用的 actor 权重版本。
- `s = v_t - v_r`：staleness。

`max_staleness=0` 接近严格 on-policy。

`max_staleness=1/2/...` 允许 rollout 用稍旧策略生成数据，换吞吐。

这个设计使同一套代码可以跑 colocate、separated on-policy、fully async off-policy。

### 创新 5：Omni-native 数据与并行设计

Relax 不是把多模态当文本 pipeline 的附件，而是从数据、并行、模型转换都做了原生支持。

包括：

- 统一 JSON/media 数据加载。
- 按 modality 处理 image/video/audio/text。
- ViT tensor parallel 策略：视觉 encoder 通常复制到 TP ranks，而不是硬切。
- Encoder-aware pipeline parallel：把视觉/音频 encoder 放到 PP0，避免大 tensor 跨 pipeline stage。
- Megatron Bridge 扩展：支持 HF \<-> Megatron 权重转换，覆盖多模态组件。

### 创新 6：R3 低开销支持 MoE 稳定训练

MoE 训练有一个特殊问题：SGLang 推理和 Megatron 训练的数值差异可能导致 router 选不同专家。

R3 的思路是：

- rollout 时记录每个 token 的 top-k expert routing。
- training 时 replay 这些 routing。
- 让训练和推理激活同一组专家。

论文声称 Relax 把 R3 的额外开销压到 1.9%，而 veRL 是 32% step-time degradation。

## 实验怎么做？

论文实验围绕四个问题展开。

### 实验 1：Omni-modal 是否能稳定收敛？

目标：证明 Relax 的多模态 pipeline 是正确且稳定的。

设置：

- Qwen3-Omni-30B。
- Echo Ink：image + text + audio。
- NextQA：video。
- 16×H20 GPUs。

结果：

- Echo Ink 上 reward 从约 0.72 收敛到约 0.93，约 450 steps 到 plateau。
- NextQA 视频任务上连续训练 2000 steps，reward 从约 0.75 提升到约 0.93。
- 方差保持较低，std 约 0.04–0.06。

论证方式：用 reward 曲线证明多模态训练没有 collapse，说明数据 pipeline、模型并行、rollout 和 reward 计算能稳定闭环。

### 实验 2：和 veRL 比 end-to-end 性能

目标：证明 Relax 不只是能跑，还更快。

设置：

- Qwen3-4B。
- DAPO-MATH-17k。
- 16×H800 GPUs。
- 最大 response length 20,480。
- Relax 和 veRL 都使用 Megatron + SGLang。
- 都是 off-policy，staleness=1。

结果：

- Relax：125.6 s/step，28.7 steps/hour。
- veRL：150.5 s/step，23.9 steps/hour。
- Relax 端到端 speedup：1.20×。

关键解释：

- Relax 的 rollout wall-clock cost 在 critical path 上是 0 s，因为 rollout 和 training 并行。
- reference logprob extra cost 也是 0 s，因为 reference forward 部署在独立资源上。
- veRL 里 rollout residual 38.2 s 和 reference forward 27.3 s 仍在 critical path。
- Relax training engine data wait 只有 0.11 s/step。

论证方式：不仅给 step time，还给 Gantt chart 分析 critical path，说明快在哪里。

### 实验 3：colocate、on-policy、off-policy 三种模式比较

目标：证明 `max_staleness` 这套抽象能在质量不掉的情况下换吞吐。

Qwen3-4B / DAPO-MATH-17k：

- Colocate：225.9 s/step，15.9 steps/hour。
- Async on-policy：201.0 s/step，17.9 steps/hour，1.12×。
- Async off-policy：128.6 s/step，28.0 steps/hour，1.76×。

结论：

- 三种模式 reward 收敛到同一水平。
- off-policy async 用约一半 wall-clock 到达相同 reward。
- colocate 慢是因为 training 和 inference 共享 GPU，需要 sleep/wakeup，约 10 s/step。

Qwen3-Omni-30B / Echo Ink：

- Fully async：133.6 s/step，26.9 steps/hour。
- Colocate：267.4 s/step，13.5 steps/hour。
- Speedup：2.00×。
- 最终 reward 都约 0.93。

论证方式：同时看 step-time、steps/hour、reward-by-step 和 reward-by-wall-clock，证明吞吐提升不牺牲收敛。

### 实验 4：MoE 的 R3 稳定性和开销

目标：证明 Relax 能高效支持 MoE train-rollout routing 对齐。

设置：

- Qwen3-30B-A3B。
- 16×H800。
- DAPO-MATH-17k。
- 2×2 对比：Relax/veRL × with/without R3。

结果：

- R3 将 rollout-training logprob mismatch 降低约 38%。
- Relax 和 veRL 在有无 R3 下 reward 曲线一致，说明 R3 保持收敛。
- veRL 开 R3 造成 32% step time overhead。
- Relax 开 R3 只增加 1.9% overhead。

论证方式：同时证明 correctness 和 overhead：

- routing mismatch 降了。
- reward 没变差。
- step time overhead 很小。

### 附录实验

论文还补了几个验证：

- Qwen3-30B-A3B text-only DAPO-MATH：reward、pass@k、AIME-24 都上升。
- Deepeyes / Qwen3-VL-MoE-30B agentic tool-calling：Relax 接近 veRL，最终 reward 上限 2.0。
- FP16 vs BF16：Qwen3-4B 上 FP16 将 train-rollout logprob 差异从 0.0122 降到 0.0016，约 7.7× 降低。

## 论文如何论证 Relax 优于现有系统？

它不是只用一个指标证明，而是分四条证据链：

### 证据链 1：能训对

通过 reward 曲线说明：

- text-only 能训。
- image/audio/video omni-modal 能训。
- agentic multi-turn 能训。
- MoE 加 R3 后仍能训。

这证明系统没有破坏 RL training correctness。

### 证据链 2：端到端更快

对比 veRL：

- 同样 Megatron + SGLang。
- 同样 Qwen3-4B + DAPO-MATH。
- Relax 1.20× speedup。

这样排除了“后端不同导致快”的干扰，更像是系统数据流和调度设计带来的收益。

### 证据链 3：异步不掉收敛

三种模式下 reward 最终一样：

- colocate
- async on-policy
- async off-policy

说明 staleness 在实验范围内没有破坏 DAPO-MATH 收敛。

### 证据链 4：复杂场景开销低

R3 是 MoE correctness 关键功能。

veRL 开 R3 慢 32%，Relax 只慢 1.9%。

这说明 Relax 的 service/data-bus/DCS 设计不仅能加功能，而且能把功能开销移出 critical path。

## 论文里的关键系统假设

适配时要注意，论文默认成立的一些前提包括：

- Megatron 训练后端可用。
- SGLang 推理后端可用。
- HF \<-> Megatron 权重转换可用。
- DCS 能完成训练侧到推理侧的权重同步。
- TransferQueue 能稳定承载字段级数据写入和读取。
- 每个 role 的资源配置和可见 GPU 与并行度匹配。

我们在 AMD/Qwen3.5 适配时遇到的问题，基本都落在这些前提上。

## 对 AMD/Qwen3.5 适配的启发

### 1. 不要一开始追求 full training profile

论文的 full 实验配置很重，例如 Qwen3-4B 使用 20k response length、16×H800。

AMD 适配时应先跑 smoke profile：

- 小 `NUM_ROLLOUT`
- 小 `rollout_batch_size`
- 小 `n_samples_per_prompt`
- 短 `rollout_max_response_len`
- 确认完整链路能过，再逐步放大。

### 2. 优先验证角色闭环，而不是追求 reward

适配初期最重要的不是 reward 上升，而是确认：

- `rollout` 能生成。
- `reference` 能算 logprob。
- `actor_fwd` 能算 logprob。
- `advantages` 能算 advantage/return。
- `actor` 能消费 batch 并走训练 step。
- actor 权重能同步回 rollout/actor_fwd。

### 3. 9B/Qwen3.5 的关键风险在 GatedDeltaNet/FLA 和 rollout memory

我们已经遇到：

- Megatron `GatedDeltaNet` 需要 `flash-linear-attention` 的 `fla` 包。
- SGLang rollout engine 的 TP/GPU 可见性必须匹配，否则 invalid device ordinal。
- 单卡 rollout engine 在长 response / 大 batch 下容易 logits allocation OOM。

所以适配优先级应该是：

1. 固化 ROCm TE。
2. 固化 FLA。
3. 固化 SGLang ROCm kernel/AITER。
4. 先用 1-GPU rollout engine 验证链路。
5. 再扩大 `rollout_num_gpus_per_engine`、batch、response length。

### 4. 需要特别关注 logprob 一致性

论文反复强调 train-rollout mismatch：

- dense 模型受 BF16/FP16 精度影响。
- MoE 模型受 routing mismatch 影响。
- GRPO/PPO 依赖 logprob ratio。

AMD 适配时要额外看：

- SGLang rollout logprob 和 Megatron actor_fwd logprob 是否接近。
- reference logprob 是否正常。
- TE attention backend 是否一致、稳定。
- BF16 是否带来过大的 train-rollout logprob drift。

### 5. TransferQueue 字段完整性是调试重点

Relax 的异步设计依赖字段级数据流。

如果某个字段没写入或读取时机不对，会出现：

- actor 等不到 `response_lengths`
- advantages 等不到 `ref_log_probs`
- train batch 缺 `returns/advantages`
- staleness 或 partition 消费状态异常

我们适配 Qwen3.5 时已经遇到 tiny smoke 下 debug logging 读取 transient buffer 缺字段的问题。

这不是算法错误，而是异步字段流里常见的时序问题。

## 适配工作建议

### 第一阶段：最小 smoke

目标：完整跑通一个 tiny step。

建议配置：

- 先用 Qwen3-4B dense 验证通用链路。
- 再用 Qwen3.5-9B 验证 GatedDeltaNet/FLA。
- rollout response length 控制在 1k–2k。
- rollout batch 和 n-samples 都小一点。

验收标准：

- 5 个服务全部 ready。
- step 0 rollout 完成。
- reference / actor_fwd logprob 完成。
- advantages 完成。
- actor training step 完成。
- 权重同步完成。

### 第二阶段：逐步放大 rollout

目标：找到 AMD 上稳定的 SGLang memory envelope。

逐步放大：

- `rollout_max_response_len`
- `n_samples_per_prompt`
- `rollout_batch_size`
- `rollout_num_gpus_per_engine`
- `sglang_mem_fraction_static`

每次只放大一个变量，避免 OOM 时不知道是谁导致的。

### 第三阶段：验证数值一致性

目标：确认训练和推理不是“能跑但不对”。

重点看：

- rollout logprob vs actor_fwd logprob
- reference logprob
- KL / importance ratio
- reward curve 是否稳定
- 是否出现 NaN / inf / reward collapse

### 第四阶段：性能优化

目标：接近论文的 async throughput 优势。

优化方向：

- 增加 rollout engines。
- 打开更大的 response length。
- 调整 `max_staleness`。
- 调整 micro-batch 和 TransferQueue sampler。
- 减少 actor wait rollout 的时间。

## 对 PR/工程改动的检查清单

适配 AMD/Qwen3.5 时，每个 PR 可以按这个顺序自查：

- Dockerfile 是否安装 ROCm TE、SGLang ROCm kernels、AITER、FLA。
- Megatron 是否能初始化 Qwen3.5 的 GatedDeltaNet。
- SGLang 是否能加载 Qwen3.5 并完成 health_generate。
- rollout engine 的 `CUDA_VISIBLE_DEVICES` 和 `rollout_num_gpus_per_engine` 是否匹配。
- `qkv_format`、`variable_seq_lengths`、TE backend 是否与 ROCm 支持路径匹配。
- `TORCHDYNAMO_DISABLE` / `--disable-jit-fuser` 是否避免 Dynamo/SymInt 问题。
- TransferQueue 是否能看到需要的字段。
- tiny smoke 是否至少过完整 step 0。
