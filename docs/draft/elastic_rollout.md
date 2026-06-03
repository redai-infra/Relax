# 弹性 Rollout 扩展方案 (Elastic Rollout Scaling)

**背景：**

- **必要性**：
  - 60~70% 的训练时间消耗在 rollout 阶段，容易成为整体训练的瓶颈。
  - 在请求量较大或模型生成长度较长时，动态扩展 rollout 资源可以显著减少端到端训练耗时。
- **可行性**：
  - **全异步模式 (Fully-async)**：在该模式下，rollout 独占 GPU 资源并作为独立模块运行。它主要实现“生成”接口和“权重同步”接口，这使得它非常适合进行弹性扩展和容错处理。
  - **解耦架构**：通过 `sglang_router`（`model_gateway`）管理多个推理引擎实例，Relax 可以动态接入新的 sglang engine 实例而无需中断核心训练流程。

**设计原则：**

1. **服务化**：以 server-based 和状态机的视角剖析扩缩容问题。
2. **最小侵入性**：尽量利用现有组件（如 Ray, SGLang），不做额外改动。
3. **接口简洁**：提供统一的 Scaler API，隐藏后端实现细节。

**设计亮点：**

幂等性设计：scale-out 的 ray_native 模式使用绝对目标值 + in-flight 计数计算 delta，external 模式通过地址去重。
互斥锁：\_find_active_scale_request() 确保同一时间只有一个 scale 操作在进行，避免了资源竞争。
取消支持：PENDING 和 CREATING 阶段都支持取消，PG 等待轮询中周期性检测取消状态。
Partial success policy：external 模式支持 rollback_all 等策略，提供了灵活的部分失败处理。
P2P 权重同步：scaled-out 引擎跳过 DCS 注册，通过 P2P Direct Sync 同步权重，避免 DCS 拓扑管理的复杂性和稳定性问题。
Concurrency groups：scale_out 和 scale_in 使用独立的并发组，不阻塞主训练流程的 generate/eval。

**暂时无需考虑：**

1. 不用考虑兼容 PD 分离等高级 rollout 特性
2. 不用考虑与 `--rollout_external` 的兼容性问题，相比原功能在初始化的时候接入，我们是在运行时接入
3. 不用考虑除 sglang 外的其他推理引擎的兼容（如 vllm）
4. 不用考虑共卡场景（colocated），只考虑全异步（fully-async）
5. 不用考虑除 rollout 外其他模块的扩缩

## Part 1：扩容接口

### 设计思路

#### 扩容接口

按照`服务化`原则提供扩容接口，在 `relax/components/rollout.py` 中基于 `ray server ingress` 添加对应的路由和实现。扩容接口有两个：

1. 接口1：是直接传入目标副本总数（`ray_native`），利用 ray 的扩缩能力，前提是同一个 ray 集群内有空闲资源，如果没有则持续等待一定时间
2. 接口2：传入已经起好的 rollout engine 的 meta 信息（`external`)，只做连接，权重同步和流量分发，可以在不同的集群，甚至联邦

扩容接口应该是一个**异步接口**，客户端可以通过返回的扩容请求 id 查询扩容状态。

扩容请求执行路线：`relax/components/rollout.py` -> `relax/distributed/ray/rollout.py` -> `relax/backends/sglang/sglang_engine.py`

#### 扩容实现

在 `relax/distributed/ray/rollout.py` 中的 `RolloutManger` 类，用于管理扩容接入。

- 如果是 `ray_native`，则直接新增 `EngineGroup`，调用 start_engines 等函数，向 ray 提交资源申请，同时 Manager 相关的状态也要修改。如果 ray 资源不够，则该 ray task ref 会一直等待直到超时或者 ray 可以拿到这些资源。
- 如果是 `external` 模式，则参考 `args.rollout_external` 开启时的操作（包括 `relax/distributed/ray/rollout.py` 和 `relax/backends/sglang/sglang_engine.py`等文件中的针对性适配），向 sglang router 中注册外部 sglang engine，如果 sglang router（sglang model gateway）相关有问题，请求参考 https://github.com/sgl-project/sgl-project.github.io/blob/main/markdown/advanced_features/sgl_model_gateway.md 。

无论是 `ray_native` 还是 `external` ，都需要考虑如下问题：

1. 状态查询：需要查询扩容的 engine 状态是否正常，是否可以接受调度请求进来
2. 权重同步：engine 状态正常后，需要执行权重同步，才可以放请求进来，这个是后面的重点，会展开描述

#### 权重同步

权重同步是 rollout 区别与普通推理服务的重点，也是难点。对于扩容后的推理引擎首次权重同步：

**P2P Direct Sync 方案（当前实现）：**

Scaled-out 引擎跳过 DCS 注册，通过 P2P 方式直接从 seed engine 同步权重：

1. **引擎启动时跳过 DCS 注册**：通过 `skip_dcs_registration=True` 参数，scaled-out 引擎不注册到 DCS Coordinator
2. **权重同步延迟执行**：引擎启动完成后立即注册到 Router，但权重同步延迟到 Actor 的 DCS 权重同步完成后
3. **Actor 触发 P2P 同步**：在 `update_weights_fully_async()` 完成后，Actor 调用 `RolloutManager.sync_weights_for_scaled_out_engines()`
4. **Direct Sync 执行**：从 seed engine 通过 NCCL Broadcast 直接传输权重到 scaled-out 引擎

**优点：**

- 代码简单，避免 DCS 拓扑管理的复杂性
- 不影响现有引擎的权重同步流程
- 避免动态拓扑变更带来的稳定性问题

**实现细节：**

```python
# 在 MegatronTrainRayActor.update_weights_fully_async() 中
# DCS 权重同步完成后，触发 P2P 同步
if (
    not rollout_only
    and dist.get_rank() == 0
    and hasattr(self, "rollout_manager")
    and self.rollout_manager is not None
):
    result = ray.get(
        self.rollout_manager.sync_weights_for_scaled_out_engines.remote(),
        timeout=300,
    )
```

#### 请求调度

核心逻辑都在 sglang router 中，可以认为只要 engine 接入了 sglang router，就无需关心请求的流转问题。

### 接口详细设计

#### 状态机设计

扩容请求遵循以下状态机：

```
PENDING → CREATING/CONNECTING → HEALTH_CHECKING → WEIGHT_SYNCING → READY
    ↓            ↓                    ↓                ↓           ↓
  FAILED      FAILED              FAILED           FAILED      ACTIVE
                                                              ↓
                                                           REMOVING
```

**状态说明：**

| 状态              | 说明                                               | 触发条件           |
| ----------------- | -------------------------------------------------- | ------------------ |
| `PENDING`         | 请求已接收，等待处理                               | 创建扩容请求       |
| `CREATING`        | (ray_native) 正在创建 Ray actor 和启动 SGLang 进程 | 资源申请开始       |
| `CONNECTING`      | (external) 正在连接外部引擎                        | 开始连接外部地址   |
| `HEALTH_CHECKING` | 引擎已启动/连接，正在进行健康检查                  | 引擎启动完成       |
| `WEIGHT_SYNCING`  | 引擎健康，正在进行权重同步                         | 健康检查通过       |
| `READY`           | 权重同步完成，可以接受请求                         | 权重同步完成       |
| `ACTIVE`          | 已注册到 Router，正在处理请求                      | Router 注册成功    |
| `FAILED`          | 扩容失败                                           | 任意阶段出错       |
| `REMOVING`        | 正在移除（缩容或失败回滚）                         | 缩容请求或失败回滚 |

**状态转换约束：**

- 只有 `PENDING` 状态可以取消
- `FAILED` 状态可以重新发起扩容请求
- `ACTIVE` 状态的引擎可以被缩容

**元数据追踪：**

```python
@dataclass
class ScaleOutRequest:
    request_id: str                    # UUID，唯一标识
    mode: str                          # "ray_native" | "external"
    status: str                        # 状态机状态
    num_replicas: int                  # 目标副本总数（绝对值，非增量）(ray_native)
    external_engine_addrs: list[str]   # 外部引擎地址列表 (external)
    created_at: float                  # 创建时间戳
    updated_at: float                  # 最后更新时间戳
    timeout_secs: float                # 超时时间
    
    # 追踪信息
    engine_ids: list[str]              # 已创建/连接的引擎 ID 列表
    failed_engines: list[str]          # 失败的引擎 ID
    error_message: str | None          # 错误信息
    weight_version: str | None         # 同步的权重版本
```

#### API 接口设计

##### 1. 发起扩容请求

**请求：**

```http
POST /rollout/scale_out
Content-Type: application/json

{
  "mode": "ray_native",  // 必填: "ray_native" | "external"
  
  // ray_native 模式参数
  "num_replicas": 6,           // 目标副本总数（绝对值，幂等：重复调用结果一致）
  
  // external 模式参数
  "external_engine_addrs": [   // 外部引擎地址列表
    "192.168.1.100:8000",
    "198.51.100.101:8000"
  ],
  
  // 通用参数
  "timeout_secs": 300.0,      // (可选) 超时时间，默认 300s
  "model_name": "default"     // (可选) 目标模型名，默认 "default"
}
```

**响应：**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PENDING",
  "message": "Scale-out request accepted"
}
```

##### 2. 查询扩容状态

**请求：**

```http
GET /rollout/scale_out/{request_id}
```

**响应：**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "WEIGHT_SYNCING",
  "mode": "ray_native",
  "num_replicas": 6,
  "engine_ids": ["engine_0", "engine_1"],
  "progress": {
    "total_engines": 2,
    "healthy_engines": 2,
    "synced_engines": 1,
    "active_engines": 0
  },
  "created_at": 1709827200.0,
  "updated_at": 1709827215.0,
  "error_message": null
}
```

##### 3. 取消扩容请求

**请求：**

```http
POST /rollout/scale_out/{request_id}/cancel
```

**响应：**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "REMOVING",
  "message": "Scale-out request cancelled, rolling back"
}
```

##### 4. 获取当前引擎状态

**请求：**

```http
GET /rollout/engines
```

**响应：**

```json
{
  "model_name": "default",
  "total_engines": 4,
  "engines": [
    {
      "engine_id": "engine_0",
      "url": "http://198.51.100.10:8000",
      "status": "ACTIVE",
      "weight_version": "42",
      "is_healthy": true,
      "load": 0.5
    }
  ],
  "router_info": {
    "ip": "198.51.100.1",
    "port": 30000,
    "policy": "cache_aware"
  }
}
```

#### 实现细节

##### ray_native 模式

扩容流程：

```python
async def scale_out_ray_native(request: ScaleOutRequest) -> None:
    # 1. 资源申请阶段
    request.status = "CREATING"
    
    # 1.1 创建新的 EngineGroup
    # 注意：skip_dcs_registration=True，scaled-out 引擎不注册到 DCS
    new_group = EngineGroup(
        args=self.args,
        pg=self._get_or_create_placement_group(request.num_gpus),
        all_engines=[None] * request.num_replicas,
        num_new_engines=0,
        worker_type="regular",
        rank_offset=self._get_next_rank_offset(),
        gpu_offset=self._get_next_gpu_offset(),
        router_ip=self.server.router_ip,
        router_port=self.server.router_port,
        is_scaled_out=True,
        skip_dcs_registration=True,  # 跳过 DCS 注册
    )
    
    # 1.2 启动引擎 (非阻塞，返回 Ray ObjectRef)
    init_handles, port_cursors = new_group.start_engines()
    
    # 1.3 等待引擎启动 (带超时)
    try:
        ray.get(init_handles, timeout=request.timeout_secs)
    except ray.exceptions.GetTimeoutError:
        request.status = "FAILED"
        request.error_message = "Engine startup timeout"
        await self._rollback_engines(new_group)
        return
    
    # 2. 健康检查阶段
    request.status = "HEALTH_CHECKING"
    for engine in new_group.engines:
        if not await self._health_check_engine(engine, timeout=60):
            request.status = "FAILED"
            request.error_message = f"Health check failed for engine {engine}"
            await self._rollback_engines(new_group)
            return
    
    # 3. 权重同步延迟到 Actor 的 DCS 同步完成后
    # 注意：不在扩容时同步权重，而是等待 Actor 调用 sync_weights_for_scaled_out_engines()
    # 这样可以避免 DCS 拓扑管理的复杂性
    
    # 4. Router 注册
    await self._register_engines_to_router(new_group.engines)
    
    # 5. 加入活跃引擎池
    self.server.engine_groups.append(new_group)
    request.status = "READY"
```

**Placement Group 管理：**

- 如果已有 PG 有足够空闲资源，复用现有 PG
- 如果资源不足，可以创建新的 PG（需要 Ray 集群支持动态资源申请）
- 资源申请是阻塞的，Ray 会一直等待直到有资源或超时

##### external 模式

扩容流程：

```python
async def scale_out_external(request: ScaleOutRequest) -> None:
    # 1. 连接阶段
    request.status = "CONNECTING"
    
    new_engines = []
    for addr in request.external_engine_addrs:
        host, port = addr.split(":")
        
        # 1.1 创建 SGLangEngine actor (连接模式)
        # 注意：skip_dcs_registration=True，外部引擎不注册到 DCS
        engine = SGLangEngine.options(
            num_cpus=0.2,
            num_gpus=0.2,
        ).remote(
            self.args,
            rank=self._get_next_rank(),
            worker_type="regular",
            skip_dcs_registration=True,  # 跳过 DCS 注册
        )
        
        # 1.2 初始化连接
        try:
            ray.get(engine.init.remote(
                dist_init_addr=f"{host}:{port}",
                port=int(port),
                nccl_port=None,
                host=host,
                router_ip=request.external_router_addr or self.server.router_ip,
                router_port=request.external_router_addr or self.server.router_port,
            ), timeout=30)
        except Exception as e:
            request.status = "FAILED"
            request.error_message = f"Failed to connect to {addr}: {e}"
            await self._rollback_engines(new_engines)
            return
        
        new_engines.append(engine)
    
    # 2. 健康检查
    request.status = "HEALTH_CHECKING"
    # ... 同 ray_native
    
    # 3. 权重同步延迟到 Actor 的 DCS 同步完成后
    # 注意：不在扩容时同步权重，而是等待 Actor 调用 sync_weights_for_scaled_out_engines()
    
    # 4. Router 注册 (如果使用本地 router)
    if not request.external_router_addr:
        await self._register_engines_to_router(new_engines)
    
    # 5. 创建 EngineGroup 并加入
    new_group = EngineGroup(
        args=self.args,
        pg=None,  # external 模式不需要 PG
        all_engines=new_engines,
        num_new_engines=len(new_engines),
        worker_type="regular",
        is_scaled_out=True,
        skip_dcs_registration=True,  # 跳过 DCS 注册
    )
    self.server.engine_groups.append(new_group)
    request.status = "READY"
```

**外部引擎配置验证：**

- 调用 `/get_server_info` 获取外部引擎的实际配置
- 验证关键参数匹配：`tp_size`, `model_path`, `trust_remote_code` 等
- 参考 `_init_external()` 中的 `_sanity_check_server_args()`

##### Router 注册

使用 SGLang Router API (>= 0.3.0)：

```python
async def _register_engines_to_router(self, engines: list[SGLangEngine]) -> None:
    for engine in engines:
        if engine.node_rank != 0:
            continue
        
        payload = {
            "url": f"http://{engine.server_host}:{engine.server_port}",
            "worker_type": engine.worker_type,
        }
        
        response = requests.post(
            f"http://{self.router_ip}:{self.router_port}/workers",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        
        # Router 返回 worker_id
        worker_info = response.json()
        engine.router_worker_id = worker_info.get("id")
```

##### Scaled-Out 引擎权重同步

权重同步在 Actor 的 DCS 同步完成后触发：

```python
# 在 RolloutManager 中
async def sync_weights_for_scaled_out_engines(self, model_name: str = "default") -> dict:
    """
    为 scaled-out 引擎同步权重。
    
    在 Actor 的 update_weights_fully_async() 完成后调用。
    使用 P2P Direct Sync 从 seed engine 同步权重。
    """
    # 收集所有 scaled-out 引擎
    scaled_out_engines = []
    for group in srv.engine_groups:
        if group.is_scaled_out:
            for engine in group.engines:
                if engine is not None:
                    scaled_out_engines.append(engine)
    
    if not scaled_out_engines:
        return {"success": True, "synced_count": 0}
    
    # 使用 Direct Sync 从 seed engine 同步
    success = await self._sync_weights_from_seed_engine(
        scaled_out_engines,
        timeout=180.0,
        model_name=model_name,
    )
    
    return {
        "success": success,
        "synced_count": len(scaled_out_engines) if success else 0,
    }
```

#### 错误处理

##### 超时策略

| 阶段     | 默认超时 | 配置参数                |
| -------- | -------- | ----------------------- |
| 内置超时 | 1800s    | `--scale-out-timeout`   |
| 总体超时 | 600s     | 请求参数 `timeout_secs` |

##### 失败回滚

```python
async def _rollback_engines(self, engines: list[SGLangEngine] | EngineGroup) -> None:
    """
    清理失败的引擎。
    
    步骤：
    1. 从 Router 注销
    2. 从 DCS coordinator 注销
    3. 关闭 SGLang 进程
    4. Kill Ray actor
    5. 释放 Placement Group 资源（如果是独立 PG）
    """
    if isinstance(engines, EngineGroup):
        engines = engines.all_engines
    
    for engine in engines:
        if engine is None:
            continue
        
        try:
            # 1. 从 Router 注销
            await engine.shutdown.remote()  # 内部会调用 Router API
            
            # 2. 从 DCS coordinator 注销
            await engine.unregister_dcs.remote()
            
            # 3. Kill Ray actor
            ray.kill(engine)
            
        except Exception as e:
            logger.warning(f"Failed to rollback engine {engine}: {e}")
```

##### 部分成功处理

当部分引擎扩容成功，部分失败时：

1. **策略A（推荐）**：全部回滚，返回失败，让用户重新发起请求
2. **策略B**：保留成功的引擎，返回部分成功状态，用户决定是否继续

```python
if len(request.failed_engines) > 0:
    if self._config.partial_success_policy == "rollback_all":
        await self._rollback_engines(all_engines)
        request.status = "FAILED"
        request.error_message = f"Partial failure: {len(request.failed_engines)}/{len(all_engines)} engines failed"
    else:  # "keep_partial"
        request.status = "READY"
        request.error_message = f"Partial success: {len(request.engine_ids)}/{request.total_engines} engines ready"
```

#### 关键实现文件

| 文件                                           | 改动内容                                                                                          |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `relax/components/rollout.py`                        | 添加 FastAPI 路由：`/scale_out`, `/scale_out/{id}`, `/engines`                                    |
| `relax/distributed/ray/rollout.py`                         | RolloutManager 添加：`scale_out()`, `get_scale_status()`, `sync_weights_for_scaled_out_engines()` |
| `relax/backends/sglang/sglang_engine.py` | 添加：`skip_dcs_registration` 参数支持，`register_to_router()`, `unregister_from_router()`        |
| `relax/backends/megatron/actor.py`       | 在 `update_weights_fully_async()` 中调用 `sync_weights_for_scaled_out_engines()`                  |
| `relax/utils/arguments.py`                     | 添加扩缩容相关参数                                                                                |

#### 与现有功能的集成点

1. **Fault Tolerance**：

   - 新引擎应被 `RolloutHealthMonitor` 监控
   - 使用现有的 `recover_rollout_engines` 机制

2. **Offload/Onload**：

   - 新引擎应支持 offload/onload 操作
   - 通过 `EngineGroup.offload()` / `onload()` 管理

3. **Weight Update**：

   - Scaled-out 引擎通过 P2P Direct Sync 同步权重
   - 权重同步在 Actor 的 DCS 同步完成后触发
   - 不影响 DCS 拓扑管理

## Part 2：缩容接口

### 设计思路

#### 缩容原则

缩容（Scale-In）遵循“先扩后缩、优雅退出、状态一致”的原则：

1. **优先级策略**：优先缩减通过扩容接口新增的引擎（LIFO），其次才是初始配置的引擎。在 `EngineGroup` 内部，优先移除通过 `external` 模式接入的引擎。
2. **优雅 Drain**：缩容前必须先将目标引擎在 Router 中标记为不健康（Draining），停止新请求的调度，并等待在途请求处理完成。
3. **资源回收**：对于被强制终止（Abort）的在途请求，利用 `partial_rollout` 机制回收已生成的片段，减少计算浪费。
4. **防抖动（Anti-Thrashing）策略**：对同一个实例 IP 或节点进行指数退避，避免在资源紧张或负载波动时出现频繁的扩缩容切换。
5. **幂等性**：对同一个参数缩容，最终结果必须一致（可以类似扩容那样，参数为最终的目标副本数）

#### 缩容流程

缩容执行路线：`relax/components/rollout.py` -> `relax/distributed/ray/rollout.py` -> `relax/backends/sglang/sglang_engine.py`

具体步骤包括：

1. **选择目标**：根据缩容模式（按数量或按特定目标）确定待移除的 `SGLangEngine` 列表。
2. **流量隔离（Drain）**：通过 Router API 将目标引擎状态置为 `is_healthy: false`。
3. **处理在途请求**：
   - 检查是否有正在运行的 rollout。如果有，调用引擎的 `abort_request` 并通过 `partial_rollout` 机制回收部分样本。
   - 检查权重更新状态。如果正在进行 `update_weights_fully_async`，则需等待该同步完成后再执行缩容，避免分布式状态不一致。
4. **清理注销**：
   - 从 DCS Coordinator 注销该引擎，更新训练拓扑。
   - 调用引擎 `shutdown()` 关闭 SGLang 进程并释放端口。
   - 停止 Ray Actor 并通知 `HealthMonitor` 将其加入 `intentionally_removed` 黑名单，防止被自动拉起。

### 接口详细设计

#### 状态机设计

缩容请求遵循简化且顺序化的状态机（相比扩容，缩容各阶段更短且串行，无需细粒度异步等待点）：

```
PENDING → DRAINING → REMOVING → COMPLETED
    ↓         ↓          ↓
  FAILED    FAILED     FAILED
```

**状态说明：**

| 状态        | 说明                                         | 触发条件                   |
| ----------- | -------------------------------------------- | -------------------------- |
| `PENDING`   | 请求已接收，正在选择待移除的引擎             | 创建缩容请求               |
| `DRAINING`  | 已在 Router 中标记为不健康，正在处理在途请求 | 引擎选择完成，开始隔离流量 |
| `REMOVING`  | 正在注销 DCS、关闭进程及 Kill Actor          | 流量隔离完成或 Drain 超时  |
| `COMPLETED` | 缩容成功，资源已完全释放                     | 所有清理步骤执行完成       |
| `FAILED`    | 缩容失败                                     | 任意阶段出现不可恢复错误   |

**状态转换约束：**

- 只有 `PENDING` 和 `DRAINING` 状态可以取消（取消后直接恢复引擎健康状态）
- `REMOVING` 状态不可取消（已开始不可逆的资源清理）
- `FAILED` 状态下已成功移除的引擎不会回滚（部分成功语义）
- 扩缩容互斥：当有任何活跃的扩容（`ScaleOutRequest` 非终态）或缩容（`ScaleInRequest` 非终态）请求时，新提交的扩容或缩容请求直接返回 HTTP 409 `CONFLICT`，不会排队等待
- 初始引擎保护：由 `--rollout-num-gpus` 和 `--rollout-num-gpus-per-engine` 启动参数定义的初始引擎不可被缩容，只能缩容之前通过扩容动态添加的引擎。若 `num_replicas` 小于初始引擎数，请求返回 HTTP 400 `REJECTED`

**元数据追踪：**

```python
@dataclass
class ScaleInRequest:
    request_id: str                    # UUID，唯一标识
    status: str                        # 状态机状态
    num_replicas: int                  # 预期缩容到的目标副本数（绝对值，非要缩的数量）
    target_engines: list[str]          # 待移除引擎地址/ID（仅当 num_replicas=0 时生效）
    created_at: float                  # 创建时间戳
    updated_at: float                  # 最后更新时间戳
    timeout_secs: float                # 总体超时时间
    force: bool                        # 是否强制缩容
    dry_run: bool                      # 是否仅预览

    # 追踪信息
    selected_engines: list[str]        # 选中待移除的引擎 ID 列表
    removed_engines: list[str]         # 已成功移除的引擎 ID
    failed_engines: list[str]          # 移除失败的引擎 ID
    error_message: str | None          # 错误信息
    partial_samples_recovered: int     # 回收的部分样本数
```

#### API 接口设计

##### 1. 发起缩容请求

**请求：**

```http
POST /rollout/scale_in
Content-Type: application/json

{
  // 二选一，num_replicas > 0 时优先使用，忽略 target_engines
  "num_replicas": 4,          // 预期缩容到的目标副本数（幂等：重复调用结果一致）
  
  // 仅当 num_replicas = 0 时生效
  "target_engines": [         // 待移除引擎的地址或 ID 列表
    "192.168.1.100:8000",
    "engine_uuid_xxxx"
  ],
  
  // 通用参数
  "force": false,             // 是否强制缩容（跳过等待在途请求，直接 kill）
  "timeout_secs": 120.0,      // (可选) 总体超时时间，默认 120s
  "dry_run": false            // (可选) 仅预览待缩容列表
}
```

**响应：**

```json
{
  "request_id": "660e8400-e29b-41d4-a716-446655441111",
  "status": "DRAINING",
  "removed_engines": [
    {
      "engine_id": "engine_1",
      "url": "http://192.168.1.100:8000",
      "source": "ray_native"
    }
  ]
}
```

**Dry-run 模式响应：**

当 `dry_run: true` 时，仅返回预览结果，不执行实际缩容：

```json
{
  "dry_run": true,
  "would_remove": [
    {
      "engine_id": "engine_3",
      "url": "http://192.168.1.100:8000",
      "engine_group_index": 2,
      "source": "ray_native",
      "is_scaled_out": true,
      "current_load": 0.3
    },
    {
      "engine_id": "engine_2",
      "url": "http://198.51.100.101:8000",
      "engine_group_index": 1,
      "source": "external",
      "is_scaled_out": true,
      "current_load": 0.0
    }
  ],
  "remaining_engines": 4,
  "message": "Preview only, no engines were removed"
}
```

##### 2. 查询缩容状态

**请求：**

```http
GET /rollout/scale_in/{request_id}
```

**响应：**

```json
{
  "request_id": "660e8400-e29b-41d4-a716-446655441111",
  "status": "COMPLETED",
  "progress": {
    "total": 2,
    "drained": 2,
    "removed": 2
  },
  "partial_samples_recovered": 15,
  "updated_at": 1709827500.0
}
```

#### 实现细节

##### 缩容主流程

并发控制通过 Ray 的 `concurrency_groups={"scale_in": 4}` 装饰器实现，无需额外的 `scale_lock`。

`_scale_in` 在 `execute_scale_in` 中以 `@ray.method(concurrency_group="scale_in")` 隔离运行。

```python
async def _scale_in(self, request: ScaleInRequest) -> None:
    """缩容主流程，在 RolloutManager 中执行。"""
    
    # 1. 引擎选择阶段
    srv = self._get_rollout_server()
    selected = self._select_engines_for_removal(request, srv)
    # selected 是 list[tuple[EngineGroup, int]] —— (group, node0_idx)
    
    if not selected:
        if request.num_replicas > 0:
            # 幂等：已达到或低于目标副本数，直接完成
            request.update_status(ScaleInStatus.COMPLETED)
            return
        request.update_status(ScaleInStatus.FAILED, "No engines selected for removal")
        return
    
    # dry-run 模式：仅返回预览
    if request.dry_run:
        request.selected_engines = [f"group_{g.rank_offset}_engine_{idx}" for g, idx in selected]
        request.update_status(ScaleInStatus.COMPLETED)
        return
    
    # 2. Drain 阶段：隔离流量 + 处理在途请求
    request.update_status(ScaleInStatus.DRAINING)
    
    # 通知 HealthMonitor 不要恢复这些引擎（传入 node0_idx 作为 engine id）
    for group, node0_idx in selected:
        self.health_monitor.mark_intentionally_removed(node0_idx)
    
    await self._drain_engines(
        engine_infos=selected,
        timeout=request.timeout_secs * 0.4,  # Drain 占总超时的 40%
        force=request.force,
    )
    
    # 3. 移除阶段：清理资源
    request.update_status(ScaleInStatus.REMOVING)
    
    for group, node0_idx in selected:
        engine_id = f"group_{group.rank_offset}_engine_{node0_idx}"
        try:
            await self._remove_engine(group, node0_idx, shutdown_timeout=self.args.scale_in_shutdown_timeout)
            request.removed_engines.append(engine_id)
        except Exception as e:
            logger.warning(f"Failed to remove engine {engine_id}: {e}")
            request.failed_engines.append(engine_id)
    
    # 4. 清理 EngineGroup 引用
    self._cleanup_engine_groups(srv)
    
    # 5. 更新状态
    if request.failed_engines and not request.removed_engines:
        request.update_status(ScaleInStatus.FAILED, f"All {len(request.failed_engines)} engines failed to remove")
    else:
        request.update_status(ScaleInStatus.COMPLETED)
```

##### 引擎选择逻辑 (Selection Logic)

引擎用 `(group, node0_idx)` 元组标识（`EngineGroup` 没有 `engine_id` 属性）。`group.engines` 为 node-0 切片（`all_engines[::nodes_per_engine]`），遍历它获取 `node0_idx`。

```python
def _select_engines_for_removal(self, request: ScaleInRequest, srv) -> list:
    """
    缩容优先级：
    1. LIFO: 优先缩减最后加入的 EngineGroup (最近扩容的)
    2. 在同一 EngineGroup 内从后往前选 node0_idx
    
    行为由请求参数决定（无需 mode 字段）：
    - num_replicas > 0: 按数量缩容，num_replicas 是目标副本数（绝对值）
    - num_replicas = 0 + target_engines: 按目标缩容
    
    返回: list[tuple[EngineGroup, int]]  —— (group, node0_idx)
    """
    candidates = []
    for group in srv.engine_groups:
        for node0_idx, engine in enumerate(group.engines):
            if engine is not None:
                candidates.append((group, node0_idx))

    if request.num_replicas > 0:
        # 按数量缩容：计算需要移除的引擎数，从后往前取 (LIFO)
        current_total = len(candidates)
        num_to_remove = current_total - request.num_replicas
        if num_to_remove <= 0:
            return []
        candidates = candidates[-num_to_remove:]
    else:
        # 按目标缩容：精确匹配引擎 ID
        target_set = set(request.target_engines)
        candidates = [(g, idx) for g, idx in candidates
                       if f"group_{g.rank_offset}_engine_{idx}" in target_set]
    return candidates
```

##### Drain 流程实现

Router 通过 `PUT /workers` 接口将引擎标记为不健康（`{"url": url, "is_healthy": false}`），停止新请求调度。`httpx.AsyncClient` 直接使用（`init_http_client` 不是上下文管理器）。

```python
async def _drain_engines(self, engine_infos: list, timeout: float, force: bool) -> None:
    """
    Drain 目标引擎：隔离流量 → 等待/中止在途请求。
    """
    import httpx

    # 1. 在 Router 中标记为不健康，停止新请求调度
    async with httpx.AsyncClient(timeout=10.0) as client:
        for group, node0_idx in engine_infos:
            engine = group.engines[node0_idx]
            if engine is None:
                continue
            url = f"http://{_wrap_ipv6(self.server.router_ip)}:{self.server.router_port}"
            try:
                await client.put(
                    f"{url}/workers",
                    json={"url": f"http://{engine.server_host}:{engine.server_port}", "is_healthy": False},
                )
            except Exception as e:
                logger.warning(f"Failed to mark engine unhealthy in router: {e}")
    
    if force:
        # 强制模式：直接跳过等待
        return
    
    # 2. 等待在途请求完成（带超时）
    start_time = time.time()
    while time.time() - start_time < timeout:
        await asyncio.sleep(2)
    # 超时后继续执行 REMOVING 阶段（force abort 由调用方决定是否需要）
```

##### 引擎移除与资源清理

`engine.shutdown()` 内部已处理 Router 注销（`unregister_from_router()`），无需额外发送 DELETE 请求。`group.pg` 是一个元组 `(placement_group, reordered_bundle_indices, reordered_gpu_ids)`，释放时取 `group.pg[0]`。

```python
async def _remove_engine(self, group: EngineGroup, node0_idx: int, shutdown_timeout: float) -> None:
    """
    移除单个引擎（含多节点 tensor parallel 切片）。
    
    步骤：
    1. 从 DCS Coordinator 注销（仅对注册了 DCS 的引擎有效，scaled-out 引擎跳过）
    2. 关闭 SGLang 进程（内部会调用 unregister_from_router，释放 GPU 显存和端口）
    3. Kill Ray Actor（确保资源释放）
    """
    nodes_per_engine = max(1, self.args.rollout_num_gpus_per_engine // self.args.num_gpus_per_node)
    physical_start = node0_idx * nodes_per_engine
    physical_end = (node0_idx + 1) * nodes_per_engine
    
    for phys_idx in range(physical_start, physical_end):
        engine = group.all_engines[phys_idx]
        if engine is None:
            continue
        
        try:
            # 1. 从 DCS coordinator 注销（scaled-out 引擎的 checkpoint_engine_client 为 None，跳过）
            await engine.unregister_dcs.remote()
        except Exception as e:
            logger.warning(f"Failed to unregister DCS for engine at phys_idx={phys_idx}: {e}")
        
        try:
            # 2. 关闭 SGLang 进程（内部调用 unregister_from_router）
            ray.get(engine.shutdown.remote(), timeout=shutdown_timeout)
        except Exception:
            pass  # shutdown 失败时由下面的 ray.kill 兜底
        
        # 3. Kill Ray actor
        ray.kill(engine)
        group.all_engines[phys_idx] = None


def _cleanup_engine_groups(self, srv) -> None:
    """
    清理空的 EngineGroup。
    
    - 如果某个 EngineGroup 的所有引擎都被移除（all_engines 全为 None），移除整个 group
    - group.pg 是元组 (placement_group, bundle_indices, gpu_ids)，释放时取 pg[0]
    """
    empty_groups = []
    
    for group in srv.engine_groups:
        if all(e is None for e in group.all_engines):
            empty_groups.append(group)
            if group.pg is not None:
                ray.util.remove_placement_group(group.pg[0])
    
    for group in empty_groups:
        srv.engine_groups.remove(group)
```

#### 错误处理

##### 超时策略

| 阶段             | 默认超时 | 配置参数                      | 失败处理策略                             |
| ---------------- | -------- | ----------------------------- | ---------------------------------------- |
| 驱逐超时         | 30s      | `--scale-in-drain-timeout`    | 超时后强制 abort 在途请求，进入 REMOVING |
| Shutdown（关闭） | 20s      | `--scale-in-shutdown-timeout` | 优雅关闭失败则 `ray.kill` 强杀 Actor     |
| 总体超时         | 120s     | 请求参数 `timeout_secs`       | 强制终止所有未完成步骤，标记 FAILED      |

##### 部分成功处理

与扩容不同，缩容采用"部分成功"语义：已成功移除的引擎不会回滚，仅报告失败的部分。

```python
# 缩容不做全量回滚——已移除的引擎无法恢复，保留部分成功结果
if request.failed_engines:
    if request.removed_engines:
        # 部分成功：报告详情，让用户决定是否重试失败部分
        request.status = "COMPLETED"
        request.error_message = (
            f"Partial success: {len(request.removed_engines)} removed, "
            f"{len(request.failed_engines)} failed: {request.failed_engines}"
        )
    else:
        # 全部失败
        request.status = "FAILED"
        request.error_message = f"All engines failed to remove"
```

##### 缩容冲突处理

- **并发扩缩容**：系统通过 Ray 的 `concurrency_groups={"scale_in": 4}` 装饰器（在 `@ray.remote` 上）控制并发，`execute_scale_in` 和 `create_scale_in_request` 均在 `scale_in` 组内运行，天然限制并发数。
- **权重同步互斥**：缩容逻辑在 Drain 前检查 `RolloutManager._is_weight_updating` 标志（由 `/can_do_update_weight_for_async` 和 `/end_update_weight` 路由分别通过 `set_weight_updating(True/False)` 设置）。如果正在权重同步，缩容等待同步完成后再隔离流量，避免 DCS 拓扑变更导致 NCCL 通信失败。
- **重复缩容**：对同一引擎的重复缩容请求会被幂等处理（检查引擎是否已在 `intentionally_removed` 集合中）。

#### 关键实现文件

| 文件                                           | 改动内容                                                                               |
| ---------------------------------------------- | -------------------------------------------------------------------------------------- |
| `relax/components/rollout.py`                        | 添加缩容 Pydantic 模型及路由：`/scale_in`, `/scale_in/{id}`                            |
| `relax/distributed/ray/rollout.py`                         | RolloutManager 实现：`scale_in()`, `_select_engines_for_removal()`, `_drain_engines()` |
| `relax/engine/rollout/sglang_rollout.py`              | 增强 `abort()` 函数，支持按特定引擎列表进行局部 Abort（当前 abort 是全局操作）         |
| `relax/backends/sglang/sglang_engine.py` | 复用现有 `shutdown()`, `unregister_from_router()`, `unregister_dcs()`                  |
| `relax/utils/health_monitor.py`                | 添加 `intentionally_removed` 集合，`mark_intentionally_removed()` 方法，防止自动恢复   |
| `relax/utils/arguments.py`                     | 添加缩容相关参数：`--scale-in-drain-timeout`, `--scale-in-shutdown-timeout` 等         |

#### 与现有功能的集成点

1. **Health Monitor（健康监控）**：

   - 缩容前将目标引擎加入 `intentionally_removed` 集合
   - `RolloutHealthMonitor` 在执行 `_kill_engine` / `recover_rollout_engines` 时检查该集合，跳过主动移除的引擎
   - 缩容完成后，从监控列表中移除对应引擎的监控任务

2. **Weight Update（权重更新）**：

   - 缩容必须与 `update_weights_fully_async` 互斥：Drain 前检查权重同步状态
   - 缩容完成后，`num_new_engines` 需要相应调整（减去已移除的引擎数）
   - Scaled-out 引擎不参与 DCS 拓扑，移除不影响 DCS 权重同步

3. **DCS Coordinator（分布式检查点）**：

   - Scaled-out 引擎不注册到 DCS，移除时无需 DCS 拓扑更新
   - 初始引擎（非 scaled-out）移除时仍需从 DCS 注销

4. **Partial Rollout（样本回收）**：

   - Drain 超时后的 `abort` 操作触发 `partial_rollout` 机制回收已生成的样本片段
   - 回收的片段通过 `TransferQueue` 重新进入数据流，由 `RolloutManager` 重新分发给其他活跃引擎补齐
   - 如果回收的样本不满足最小 batch 要求，可丢弃以减少调度开销

5. **Metrics（指标上报）**：

   - 记录 `scale_in.engine_count`、`scale_in.duration_secs`、`scale_in.recovered_samples` 等指标
   - 记录 `scale_in.drain_timeout_count`（Drain 超时次数，用于评估超时参数是否合理）
   - 指标通过现有的 `MetricsClient` 上报到 Metrics Service

## Part3：扩缩容时机
