# SGLang 运行时 Patch 说明

Relax 在 SGLang 引擎启动时通过 monkey-patch 注入若干运行时增强，而非修改 SGLang 源码或 `cp` 覆盖系统文件。每个 patch 由独立 env flag 控制，在 `relax/backends/sglang/sglang_engine.py::_launch_server_with_patches` 中按 flag 应用，互斥组合任意启用。

## 总览

| Patch | env flag | 注入位置 | 作用 |
|-------|----------|----------|------|
| Routing Replay | `RELAX_OPTIMIZE_ROUTING_REPLAY=1` | scheduler 子进程 (`_patched_run_scheduler_process`) | routed-experts buffer 的异步 D→H 拷贝，消除默认流同步 |
| OPD Pre-expanded | `RELAX_OPD_PREEXPANDED_PATCH=1` | 主进程 | 接收 `opd_preexpanded_raw` 格式的预展开 `input_ids`+原始图片，跳过 decode→retokenize |
| DeepEyes Qwen-VL | `RELAX_DEEPEYES_QWEN_VL_PATCH=1` | 主进程 | 支持 DeepEyes 多轮预分词 `input_ids`，折叠连续 `<|image_pad|>` 并恢复原始 token 序列与 mrope |

## env flag 如何到达 SGLang 子进程

DeepEyes Qwen-VL patch 的主开关是 CLI 参数 `--deepeyes-qwen-vl-patch`（默认 `False`，即使用上游 SGLang 原生行为）。其传递链路：

1. 训练脚本传 `--deepeyes-qwen-vl-patch`（如 `examples/deepeyes/run_deepeyes_r3.sh`，受 `DEEPEYES_PATCH` 环境变量控制，默认 `0` 即关闭，走 stock SGLang 行为；DeepEyes 多轮训练需手动 `DEEPEYES_PATCH=1` 开启）。
2. `relax/backends/sglang/sglang_engine.py::_init_normal` 检测到 `args.deepeyes_qwen_vl_patch` 为真，设置 `os.environ["RELAX_DEEPEYES_QWEN_VL_PATCH"] = "1"`（与 `--optimize-routing-replay` 同模式）。
3. `_init_normal` 通过 `multiprocessing.Process(target=_launch_server_with_patches, start_method='spawn')` 拉起 SGLang 子进程，子进程继承父进程 `os.environ`。
4. `_launch_server_with_patches` 检查 env flag，命中则调用 `apply_deepeyes_qwen_vl_patch`。

> 也可不传 CLI 参数、直接 `export RELAX_DEEPEYES_QWEN_VL_PATCH=1` 并通过 `RELAX_PROPAGATE_ENV_VARS` 透传（程序化场景的等价路径，与 OPD patch 一致）。两者命中同一个 env flag，任选其一。

## DeepEyes Qwen-VL Patch

**模块**：`relax/backends/sglang/patches/qwen_vl_patch.py`

**为什么需要**：DeepEyes 多轮 VLM rollout（`examples/deepeyes/rollout.py::_run_inference_step`）向 SGLang 发送预分词的 `input_ids: list[int]`，而非文本。SGLang 原生 `QwenVLImageProcessor.process_mm_data_async` 会 decode→retokenize，导致两个问题：

- **A. token 漂移**：decode→retokenize 非无损，token 数可能变化，mrope position 错位。
- **B. N×M 展开**：多轮累积的 N 个 `<|image_pad|>` 被 `load_mm_data` 再次展开成 N×M。

**注入的两处真实差异**（其余上游逻辑不动）：

1. `_strip_image_token`：`load_mm_data` 前把连续 `<|image_pad|>` 折叠为单个占位符。
2. `original_input_ids` 保存/恢复：保存调用方原始 `input_ids`，让 mm 管线照常产出 `pixel_values`/grid，再恢复 token 序列并**用恢复后的 `input_ids` 重算 mrope**。

> **为什么是重写方法体而非纯包装器**：mrope 必须用恢复后的原始 `input_ids` 重算，而原始方法内部已用 retokenize 序列算好 mrope。纯包装器只改返回值的 `input_ids` 会让 input_ids 与 mrope 错位——正是要修的 bug。重算 mrope 需要方法内部中间产物 `ret`（含 `image_grid_thw`/`video_grid_thw`/`second_per_grid_ts`），外层拿不到。因此 patch 重写 `process_mm_data_async`，**委托上游原语** `self.load_mm_data` / `self.process_and_combine_mm_data` / `MRotaryEmbedding.get_rope_index` / 上游模块级 `preprocess_video`，把两处改动内联。`smart_resize`、`preprocess_video`、`get_mm_data`、`__init__` 等无关函数**不复制**，留在上游。

**幂等**：`_PATCH_FLAG = "_relax_deepeyes_patched"` 标记已 patch 的类，重复调用为 no-op。

**对普通请求无影响**：`input_text` 为 `str` 时，`_strip_image_token` 直通、`original_input_ids` 保持 `None`，方法走原路径。

## SGLang 升级检查清单

升级 SGLang 版本后，逐项校验各 patch 依赖的上游 API 仍存在：

### DeepEyes Qwen-VL Patch
- `sglang.srt.multimodal.processors.qwen_vl.QwenVLImageProcessor` 存在，且含：
  - `process_mm_data_async`（被替换的目标方法）
  - `load_mm_data`（实例方法，签名 `prompt, image_data, video_data, audio_data, multimodal_tokens`）
  - `process_and_combine_mm_data`（实例方法，返回 `(mm_items, input_ids, ret)`）
- `sglang.srt.multimodal.processors.qwen_vl.preprocess_video`（模块级 async 函数）
- `sglang.srt.layers.rotary_embedding.MRotaryEmbedding.get_rope_index`（关键字参数：`spatial_merge_size, image_token_id, video_token_id, vision_start_token_id, model_type, tokens_per_second, input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, use_audio_in_video, audio_seqlens, audio_token_id, audio_start_token_id, position_id_per_seconds`）
- `sglang.srt.managers.schedule_batch.MultimodalProcessorOutput.from_dict`（sglang ≥0.5.12；缺失时 patch 自动回退为返回 dict）

**fail-fast**：`apply_qwen_vl_patches` 在绑定前调用 `_verify_upstream_api`，缺失任一符号即 `raise RuntimeError`，提示 "Relax requires sglang 0.5.12.post1"。`apply_deepeyes_qwen_vl_patch` 额外包一层 try/except，使 patch 失败不阻断引擎启动（仅 warning）。

### OPD Pre-expanded Patch
- 同样替换 `QwenVLImageProcessor.process_mm_data_async`，依赖 `self._processor.image_processor`、`self.get_mm_items_offset`、`MRotaryEmbedding.get_rope_index`、`MultimodalProcessorOutput`。详见 `relax/utils/opd/opd_sglang_patch.py`。

### Routing Replay Patch
- 替换 `sglang.srt.managers.io_struct`（或对应模块）的 `_RoutedExpertsCapturerReal` 方法。详见 `relax/backends/sglang/routing_replay_patch.py`。

## 验证

- 单测：`pytest tests/backends/sglang/test_qwen_vl_patch.py -v`（stub sglang，无需 GPU）。
- 端到端：在装有目标 SGLang 版本 + GPU 的环境运行 `examples/deepeyes/run_deepeyes_r3.sh`，`temperature=0` 下与旧 `cp` 覆盖方案逐 token 一致。

## 已弃用

`examples/deepeyes/qwen_vl.py`（整文件 `cp` 覆盖 `/sgl-workspace/...` 的旧方案）已删除，由本 patch 取代。如发现环境中仍存在 `/sgl-workspace` 引用，请清理。
