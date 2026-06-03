# Distributed Checkpoint Service (DCS) - 架构与设计

## 概述

**Distributed Checkpoint Service (DCS)** 是为大规模多 GPU/多节点模型训练设计的高性能分布式检查点引擎。它提供：

- **控制平面/数据平面分离**：协调器处理拓扑管理；客户端处理数据传输
- **动态角色感知网络**：自动对等体发现和拓扑更新
- **设备直连通信后端**：NCCL/GLOO 用于集群内 GPU 到 GPU 或 CPU 通信
- **弹性扩展与重分片**：支持动态组变更和张量重映射
- **生产级容错**：心跳监控、自动恢复、重试策略
- **综合指标**：Prometheus 兼容的可观测性，用于延迟、吞吐量和错误

______________________________________________________________________

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      DCS Architecture                           │
├──────────────────────┬──────────────────────────────────────────┤
│   Control Plane      │              Data Plane                  │
│                      │                                          │
│  ┌────────────────┐  │   ┌──────────────────────────────────┐   │
│  │  Coordinator   │◄─┼───┤  CheckpointEngineClient          │   │
│  │  (HTTP REST)   │  │   │                                  │   │
│  │                │  │   ├─ Role Registration               │   │
│  ├────────────────┤  │   ├─ Peer Discovery                  │   │
│  │ ┌────────────┐ │  │   ├─ Tensor Send/Recv                │   │
│  │ │ Topology   │ │  │   └─ Weight Update                   │   │
│  │ │ Manager    │ │  │                                      │   │
│  │ └────────────┘ │  │   ┌──────────────────────────────────┐   │
│  │                │  │   │      Communication Backend       │   │
│  └────────────────┘  │   │                                  │   │
│                      │   ├─ DeviceDirectBackend             │   │
│                      │   │  (NCCL/GLOO)                     │   │
│                      │   └──────────────────────────────────┘   │
└──────────────────────┴──────────────────────────────────────────┘
```

______________________________________________________________________

## 核心组件

### 1. **配置** (`config.py`)

定义 DCS 部署的可调参数。

#### 关键类：

- **`BackendType`**：通信后端的枚举 (GLOO, NCCL, TCP)

- **`RoleInfo`**：表示分布式系统中的一个节点

  - `role_name`：逻辑组 (例如 "actor", "rollout", "trainer")
  - `rank`：角色内的进程 ID
  - `world_size`：该角色的总进程数
  - `ip`, `port`：P2P 通信的网络地址
  - `device_id`：GPU ID (如适用)
  - `metadata`：自定义属性 (张量并行、流水线并行等)
  - 属性 `node_id`：格式为 `"{role_name}_{rank}"`
  - 属性 `address`：格式为 `"{ip}:{port}"`

- **`DCSConfig`**：主配置类，包含以下设置：

  - **协调器**：主机、端口
  - **通信**：后端类型 (默认 GLOO)、TCP 缓冲区大小、张量融合阈值
  - **心跳**：心跳间隔、超时时间
  - **存储**：检查点目录、异步 I/O
  - **容错**：最大重试次数、重试延迟
  - **可观测性**：指标启用、Prometheus 端口

- **`TopologyConfig`**：定义角色间连接

  - `role_mappings`：例如 `{"actor": "rollout"}` 表示 actor_rank N 连接到 rollout_rank N
  - `get_peer_role(role)`：获取给定角色的对等角色

#### 配置示例：

```python
config = DCSConfig(
    coordinator_host="0.0.0.0",
    coordinator_port=8000,
    backend_type=BackendType.NCCL,
    heartbeat_interval_seconds=5.0,
    heartbeat_timeout_seconds=30.0,
    checkpoint_dir="/checkpoints",
    tensor_fusion_threshold=1024*1024,  # 1MB
    enable_metrics=True,
)
```

______________________________________________________________________

### 2. **指标** (`metrics.py`)

具有 Prometheus 导出的生产级可观测性。

#### 指标类型：

**直方图** (延迟跟踪)：

- `dcs_save_latency_seconds`：保存检查点的时间
- `dcs_load_latency_seconds`：加载检查点的时间
- `dcs_send_latency_seconds`：发送张量的时间
- `dcs_recv_latency_seconds`：接收张量的时间

**计数器** (单调递增)：

- `dcs_bytes_sent_total`, `dcs_bytes_received_total`
- `dcs_bytes_saved_total`, `dcs_bytes_loaded_total`
- `dcs_*_operations_total`：操作计数
- `dcs_errors_total`：总错误数

**仪表** (时间点)：

- `dcs_memory_buffer_usage_bytes`：当前缓冲区内存
- `dcs_active_connections`：开放连接
- `dcs_pending_operations`：进行中的操作

#### 关键类：

- **`Histogram`**：具有可配置桶的延迟跟踪

  - `observe(value)`：记录样本
  - `get_stats()`：返回计数、总和、平均值、桶分布
  - 默认桶 (秒)：`[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]`

- **`Counter`**：单调递增计数器

  - `inc(value)`：按值递增
  - 线程安全，带锁

- **`Gauge`**：可上升或下降的值

  - `set(value)`, `inc(value)`, `dec(value)`
  - 线程安全

- **`MetricsCollector`**：主收集器

  - `record_save(bytes_saved, duration)`
  - `record_load(bytes_loaded, duration)`
  - `record_send(bytes_sent, duration)`
  - `record_recv(bytes_received, duration)`
  - `record_error(error_type)`
  - `export_prometheus()`：以 Prometheus 文本格式导出
  - `get_all()`：获取所有指标作为字典

#### 使用方法：

```python
metrics = MetricsCollector()
metrics.record_send(bytes_sent=1024*1024, duration=0.05)
print(metrics.export_prometheus())  # Prometheus 格式

# 全局实例
from relax.distributed.checkpoint_service.metrics import get_metrics
metrics = get_metrics()
metrics.record_save(bytes_saved=5*1024*1024, duration=1.2)
```

______________________________________________________________________

### 3. **通信后端** (`backends/`)

具有抽象基类和一个具体实现。

#### 架构：

```
CommBackend (抽象基类)
└── DeviceDirectBackend (NCCL/GLOO)
    └── 用于集群内 GPU 到 GPU 或 CPU 通信
```

#### 基类 (`backends/base.py`)：

- **`SendRequest`**：点对点发送描述符

  - `tensor_dict`：要发送的张量
  - `dst_rank`：目标 rank
  - `group_name`：可选进程组名称
  - `async_op`：阻塞 vs 异步标志
  - `metadata`：额外数据

- **`RecvRequest`**：点对点接收描述符

  - `src_rank`：源 rank
  - `tensor_names`：预期的张量名称
  - `group_name`：可选进程组名称
  - `metadata`：额外元数据

- **`CommHandle`**：异步操作句柄

  - `request_id`：唯一操作 ID
  - `is_complete`：完成状态
  - `result`：操作结果
  - `error`：失败时的异常
  - `wait()`：阻塞等待
  - `async wait_async()`：异步等待

- **`CommBackend`** (ABC)：统一通信接口

  - `broadcast()`：一对多广播
  - `broadcast_async()`：异步广播
  - `create_group()`：创建通信组
  - `destroy_group()`：销毁通信组
  - `register_peer()`：注册对等节点
  - `init_process_group()`：初始化分布式通信

- **`TensorFusion`**：多个小张量的优化器

  - 将多个小张量连接到一个大缓冲区
  - 减少协议开销
  - 可配置阈值 (默认 1MB)
  - `should_fuse(tensor_dict)`：检查是否应融合
  - `fuse(tensor_dict)`：融合张量，返回 (fused_tensor, metadata)
  - `unfuse(fused_buffer, metadata)`：解融合回原始张量

#### 3.1 设备直连后端 (`device_direct.py`)

使用 PyTorch 分布式的高性能后端 (GPU 用 NCCL，CPU 用 GLOO)。

**构造函数参数：**

- `args`：后端参数
- `backend_type`：GLOO 或 NCCL
- `role_info`：当前节点信息
- `model`：模型实例序列
- `model_name`：模型标识符
- `quantization_config`：可选量化配置
- `coordinator_url`：协调器 URL
- `lock`：远程锁 (用于协调权重更新)
- `timeout_seconds`：操作超时 (默认 300)

**关键方法：**

- `init_process_group_for_rollout(topology_data)`：初始化与 rollout 节点的进程组
- `init_process_groups_for_actor_fwd_ref(topology_data)`：初始化 actor → actor_fwd 权重同步的进程组
- `update_weights_for_rollout(rollout_only, actor_fwd_only)`：更新 rollout/actor_fwd 节点的权重
- `recv_weight()`：actor_fwd 侧接收权重广播

**特性：**

- NCCL：GPU 集体通信，最优带宽
- GLOO：基于 CPU 的回退，支持异步
- CUDA 流集成，与计算重叠
- 全聚集、广播和点对点操作
- 带完成句柄的异步操作
- 通过 Ray Actor (`RolloutEngine`) 与 rollout 节点进行 HTTP 通信

**使用场景：**

- 同一节点上的多个 GPU (NVLink, PCIe)
- 具有 InfiniBand/以太网的多节点 GPU 集群

______________________________________________________________________

### 4. **客户端** (`client/engine.py`)

用于检查点操作的数据平面客户端。

#### 职责：

- **注册**：向协调器注册，获取 rank
- **对等体发现**：获取拓扑，识别对等体
- **权重更新**：与 rollout/actor_fwd 节点同步模型权重
- **心跳**：向协调器发送健康信号

#### 关键类：

- **`CheckpointEngineClient`**：主客户端类

  - `args`：命令行参数对象
  - `coordinator_url`：协调器端点
  - `role_info`：节点元数据 (角色、rank、设备、IP、端口)
  - `backend_type`：通信后端 (NCCL/GLOO)
  - `model`：模型引用
  - `model_name`：模型名称
  - `quantization_config`：量化配置
  - `lock`：远程锁

#### 关键方法：

- `async start()`：初始化和注册

  1. 创建 HTTP 客户端
  2. 向协调器注册
  3. 初始化通信后端

- `async stop()`：优雅关闭

  - 取消心跳
  - 关闭后端
  - 关闭 HTTP 客户端

- `async init_process_groups_for_actor_fwd_ref(rollout_id)`：初始化 actor/actor_fwd 权重同步

  - 根据 `ref_update_interval` 判断是否需要更新
  - 从协调器获取模型更新组的 rank 映射
  - 调用后端建立进程组

- `async recv_weight_fully_async()`：actor_fwd 侧异步接收权重

- `async update_weights_for_rollout(rollout_only, actor_fwd_only)`：更新 rollout 权重

  - 获取拓扑
  - 初始化 rollout 进程组
  - 调用后端传输权重

#### 属性：

```python
client.role          # 逻辑角色名称
client.rank          # 角色内的 rank
client.world_size    # 角色内的总进程数
client.node_id       # 唯一标识符
client.is_registered # 注册状态
client.backend       # 通信后端实例
```

#### 使用示例：

```python
import asyncio
from relax.distributed.checkpoint_service import CheckpointEngineClient, BackendType

async def main():
    client = CheckpointEngineClient(
        args=args,
        coordinator_url="http://localhost:8000",
        role="actor",
        rank=0,
        backend_type=BackendType.NCCL,
        device_id=0,
        model=model,
        model_name="qwen3-4B",
    )

    await client.start()

    # 向协调器注册
    print(f"注册为 {client.node_id}")

    # 更新 rollout 权重
    await client.update_weights_for_rollout()

    # 初始化 actor_fwd 权重同步
    await client.init_process_groups_for_actor_fwd_ref(rollout_id=100)

    await client.stop()

asyncio.run(main())
```

辅助函数：

```python
from relax.distributed.checkpoint_service.client import create_client

# 创建并启动客户端
client = await create_client(
    args=args,
    coordinator_url="http://localhost:8000",
    role="actor",
    rank=0,
)
```

______________________________________________________________________

### 5. **协调器** (`coordinator/`)

用于拓扑管理的控制平面服务。

#### 架构：

```
DCSCoordinator (FastAPI + Ray Serve)
├── TopologyManager
│   ├── 节点注册
│   ├── Rank 分配
│   ├── 对等体查找
│   └── 心跳监控
└── REST 端点
    ├── POST /register
    ├── DELETE /unregister
    ├── GET /heartbeat
    ├── GET /topology
    ├── GET /peer
    ├── GET /node
    ├── GET /global_rank
    ├── GET /get_model_update_group_ranks
    ├── POST /send_weight_meta
    ├── GET /recv_weight_meta
    ├── GET /clear_weight_meta
    ├── GET /health
    └── GET /debug/topology
```

#### 5.1 协调器服务 (`service.py`)

用于拓扑和权重更新管理的基于 FastAPI 的 REST API，通过 Ray Serve 部署。

**端点：**

- `POST /register`：注册新节点

  - 输入：`RegisterRequest` (role_name, rank, world_size, ip, port, device_id, metadata)
  - 输出：`RegisterResponse` (status, message, rank, node_id)
  - 返回分配的 rank

- `DELETE /unregister`：注销节点

  - 参数：`role`, `rank`

- `GET /heartbeat`：更新节点心跳

  - 参数：`role`, `rank`
  - 输出：`HeartbeatResponse` (status, timestamp)

- `GET /topology`：获取当前拓扑

  - 参数：`role_filter` (可选)
  - 输出：`TopologyResponse` (nodes, world_size)
  - 返回完整的 role->rank 映射

- `GET /peer`：获取节点的对等体

  - 参数：`role`, `rank`, `peer_role` (可选)
  - 输出：对等体的 `RoleInfo` 字典

- `GET /node`：获取特定节点信息

  - 参数：`role`, `rank`
  - 输出：`RoleInfo` 字典

- `GET /global_rank`：获取全局 rank

  - 参数：`role`, `rank`
  - 输出：`{"global_rank": int}`

- `GET /get_model_update_group_ranks`：获取权重更新的通信组

  - 参数：`role`, `rank`, `need_update_ref`
  - 输出：`GroupRanksResponse` (global_rank, world_size, train_pp_size, pp_groups)
  - 根据 actor/actor_fwd/reference 角色计算全局 rank 和 PP 组

- `POST /send_weight_meta`：发送权重元数据

  - 输入：`SendWeightMetaRequest` (names, dtypes, shapes, group_name)
  - 输出：`Response` (status, message)

- `GET /recv_weight_meta`：接收权重元数据

  - 参数：`index`
  - 输出：从 index 开始的权重元数据列表

- `GET /clear_weight_meta`：清除权重元数据缓冲区

- `GET /health`：健康检查

  - 输出：状态、时间戳、world_size、死亡节点列表

- `GET /debug/topology`：调试用完整拓扑详情

**API 模型：**

```python
RegisterRequest:
  role_name: str | None
  rank: int | None
  world_size: int | None
  ip: str | None
  port: int | None
  device_id: int | None
  metadata: Dict[str, Any] | None

RegisterResponse:
  status: str
  message: str
  rank: int
  node_id: str

HeartbeatResponse:
  status: str
  timestamp: float

TopologyResponse:
  nodes: Dict[str, Dict[int, Dict[str, Any]]]
  world_size: int

GroupRanksResponse:
  global_rank: int
  world_size: int
  train_pp_size: int
  pp_groups: dict
```

**部署：**

DCS 协调器通过 Ray Serve 部署：

```python
from relax.distributed.checkpoint_service.coordinator.service import create_dcs_deployment

coordinator, coordinator_url = create_dcs_deployment(config)
```

或直接使用 Ray Serve：

```python
from ray import serve

serve.run(DCSCoordinator.bind(config=config), name="dcs_coordinator", route_prefix="/dcs_coordinator")
```

#### 5.2 拓扑管理器 (`topology.py`)

具有生命周期管理的内存中拓扑数据库。

**特性：**

- **角色注册**：为节点分配 rank
- **对等体查找**：查找角色间连接的对等体
- **全局 Rank 映射**：逻辑到物理 rank 转换
- **心跳跟踪**：监控节点健康
- **动态更新**：支持弹性扩展
- **线程安全**：所有方法通过 RLock 保证线程安全

**关键类：**

- **`TopologyNode`**：节点表示

  - `role_info`：节点元数据
  - `last_heartbeat`：最后心跳的时间戳
  - `is_alive`：健康状态
  - `connections`：对等节点 ID 集合

- **`TopologyManager`**：拓扑数据库

  - `register(role_info)`：添加节点并分配 rank
  - `unregister(role_name, rank)`：移除节点
  - `heartbeat(role_name, rank)`：更新心跳
  - `get_node(role_name, rank)`：获取节点信息
  - `get_peer(role_name, rank, peer_role)`：查找对等体
  - `get_role_nodes(role_name)`：获取角色中的所有节点
  - `get_all_nodes()`：获取完整拓扑
  - `get_world_size(role_name=None)`：总节点数 (可按角色过滤)
  - `get_global_rank(role_name, rank)`：获取全局 rank
  - `get_all_peers(role_name, rank)`：获取所有对等体
  - `check_health()`：检查所有节点健康状态
  - `to_dict()`：导出拓扑为字典

**使用示例：**

```python
manager = TopologyManager(
    config=TopologyConfig(role_mappings={"actor": "rollout"}),
    heartbeat_timeout=30.0
)

# 注册节点
manager.register(RoleInfo(role_name="actor", rank=0, ip="10.0.0.1", port=20000))
manager.register(RoleInfo(role_name="rollout", rank=0, ip="192.0.2.2", port=20001))

# 获取对等体
peer = manager.get_peer("actor", 0, "rollout")
print(f"Actor 0 应连接到 Rollout 0，地址为 {peer.address}")

# 心跳
manager.heartbeat("actor", 0)
```

______________________________________________________________________

## 数据流

### 权重更新流 (Actor → Rollout)

```
Actor (训练)
    ↓
    └─→ 协调器
        ├─ 注册
        └─ 获取拓扑
    ↓
DeviceDirectBackend
    ├─ init_process_group_for_rollout()
    ├─ all_gather_param() (TP 聚集)
    ├─ convert_to_hf() (权重转换)
    └─ dist.broadcast() (广播到 rollout)
    ↓
Rollout 节点 (通过 RolloutEngine Ray Actor 进行 HTTP 通信)
```

### 权重更新流 (Actor → Actor FWD/Reference)

```
Actor (训练)
    ↓
    └─→ 协调器
        ├─ 注册
        └─ get_model_update_group_ranks (获取 PP 组)
    ↓
DeviceDirectBackend
    ├─ init_process_groups_for_actor_fwd_ref()
    ├─ all_gather_param() (TP 聚集)
    ├─ send_weight_meta (通过协调器发送元数据)
    └─ dist.broadcast() (广播权重)
    ↓
Actor FWD / Reference (通过 recv_weight() 接收)
```

______________________________________________________________________

## 配置示例

### 单节点多 GPU

```python
config = DCSConfig(
    backend_type=BackendType.NCCL,
    coordinator_host="127.0.0.1",
    coordinator_port=8000,
    comm_base_port=20000,
)

client = CheckpointEngineClient(
    args=args,
    coordinator_url="http://127.0.0.1:8000",
    role="actor",
    rank=0,
    backend_type=BackendType.NCCL,
    device_id=0,
    model=model,
    model_name="qwen3-4B",
)
```

### 多节点 GPU 集群

```python
config = DCSConfig(
    backend_type=BackendType.NCCL,
    coordinator_host="10.0.0.1",
    coordinator_port=8000,
    heartbeat_interval_seconds=5.0,
    heartbeat_timeout_seconds=30.0,
)

topology_config = TopologyConfig(
    role_mappings={
        "actor": "rollout",
    }
)

coordinator, coordinator_url = create_dcs_deployment(config)
```

______________________________________________________________________

## 性能调优

### 张量融合

融合通过组合小张量来减少开销：

```python
config.tensor_fusion_threshold = 1024 * 1024  # 1MB
# 仅在总张量 >= 1MB 且计数 > 1 时融合
```

### 固定内存

启用异步 GPU 到 CPU 传输：

```python
config.pinned_memory = True  # 默认，推荐用于 GPU
```

### TCP 设置

```python
config.tcp_nodelay = True              # 禁用 Nagle 算法
config.tcp_buffer_size = 65536         # 64KB 缓冲区
config.comm_base_port = 20000          # 基础端口
```

### 心跳

针对网络可靠性调整：

```python
config.heartbeat_interval_seconds = 5.0   # 每 5 秒
config.heartbeat_timeout_seconds = 30.0   # 30 秒截止
```

______________________________________________________________________

## 容错

### 节点故障检测

1. 客户端停止发送心跳
2. 协调器在心跳超时后将节点标记为死亡
3. 拓扑被更新
4. 剩余节点可以继续使用重分片拓扑

### 自动重试

```python
config.max_retries = 3
config.retry_delay_seconds = 1.0  # 指数退避
```

### 优雅关闭

```python
await client.stop()  # 清洁关闭
```

______________________________________________________________________

## 监控与可观测性

### 指标导出

```python
from relax.distributed.checkpoint_service.metrics import get_metrics

metrics = get_metrics()

# 获取为字典
stats = metrics.get_all()
print(f"发送的总字节数: {stats['counters']['bytes_sent']}")
print(f"平均发送延迟: {stats['latency']['send']['avg']:.3f}s")

# 导出为 Prometheus
prom_text = metrics.export_prometheus()
# 写入 Prometheus 端点
```

### 日志记录

所有组件使用框架提供的日志工具：

```python
from relax.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 日志会自动包含模块信息
logger.info("Checkpoint saved successfully")
```

______________________________________________________________________

## 高级主题

### 弹性扩展

系统支持动态拓扑变更：

1. 新节点向协调器注册
2. 协调器分配 rank
3. 客户端获取新拓扑并建立进程组
4. 现有通信组被更新

### 张量并行重分片

`get_model_update_group_ranks` 端点处理：

- 训练 PP (Pipeline Parallel) 大小
- actor_fwd 和 reference 节点的全局 rank 计算
- PP 同步的进程组 (每个 PP stage 一个组)
- 基于并行配置的自动组形成

### 自定义元数据

在节点上存储额外信息：

```python
role_info = RoleInfo(
    role_name="actor",
    rank=0,
    metadata={
        "tp_size": 4,
        "pp_size": 2,
        "pp_rank": 0,
        "is_pp_src_rank": True,
        "master_address": "<node-ip>",
        "master_port": 29500,
    }
)
```

______________________________________________________________________

## 网络端口分配

各服务使用固定端口范围，避免进程组初始化 (TCPStore) 时的端口冲突。完整端口分配表：

| 服务 | 端口范围 | 用途 |
|------|---------|------|
| DCS 权重同步 (Actor → Rollout) | 11000 - 11999 | `DeviceDirectBackend` TCPStore，用于 NCCL/GLOO 广播 |
| Rollout (SGLang 引擎) | 15000 - 15999 | SGLang 推理引擎 HTTP 服务 |
| GenRM (SGLang 引擎) | 16000 - 16999 | GenRM 推理引擎 HTTP 服务 |

**Megatron NCCL 端口范围**：Megatron-LM 内部 NCCL 通信使用操作系统临时端口。为避免与上述服务端口冲突，建议收缩临时端口范围：

```bash
echo "32768 50000" > /proc/sys/net/ipv4/ip_local_port_range
```

这会将操作系统自动分配的临时端口限制在 32768-50000，远高于上述保留端口范围。

> **为什么使用固定范围？** 原始实现使用操作系统随机分配端口 (`bind(0)`)，可能与其他服务冲突或因 `TIME_WAIT` 残留导致 `EADDRINUSE` 错误。固定范围配合绑定前验证可以消除此类问题。

______________________________________________________________________

## 故障排除

### 连接问题

- 检查 P2P 端口的防火墙规则 (base_port 到 base_port + max_ranks)
- 验证协调器可达
- 检查 RoleInfo 中的 IP/端口配置

### 心跳超时

- 对于不稳定的网络，增加 `heartbeat_timeout_seconds`
- 检查协调器和客户端之间的网络延迟
- 监控 `dcs_errors_total` 指标

### 低吞吐量

- 对于许多小张量启用张量融合
- 对于同一节点 GPU 通信使用 NCCL 后端
- 使用 `dcs_bytes_sent_total / time` 检查网络带宽

### 内存压力

- 监控 `dcs_memory_buffer_usage_bytes` 仪表
- 如果内存受限，减少 `tcp_buffer_size`
- 在仅 CPU 系统上禁用 `pinned_memory`

______________________________________________________________________

## 参考文献与资源

- PyTorch 分布式：https://pytorch.org/docs/stable/distributed.html
- NCCL 文档：https://docs.nvidia.com/deeplearning/nccl/
- Prometheus 指标：https://prometheus.io/docs/
