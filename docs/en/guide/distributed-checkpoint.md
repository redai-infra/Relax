# Distributed Checkpoint Service (DCS) - Architecture & Design

## Overview

The **Distributed Checkpoint Service (DCS)** is a high-performance distributed checkpoint engine designed for large-scale multi-GPU/multi-node model training. It provides:

- **Control Plane / Data Plane Separation**: Coordinator handles topology management; clients handle data transfer
- **Dynamic Role-Aware Networking**: Automatic peer discovery and topology updates
- **Device-Direct Communication Backend**: NCCL/GLOO for intra-cluster GPU-to-GPU or CPU communication
- **Elastic Scaling & Resharding**: Support for dynamic group changes and tensor remapping
- **Production-Grade Fault Tolerance**: Heartbeat monitoring, automatic recovery, retry policies
- **Comprehensive Metrics**: Prometheus-compatible observability for latency, throughput, and errors

______________________________________________________________________

## Architecture

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

## Core Components

### 1. **Configuration** (`config.py`)

Defines tunable parameters for DCS deployment.

#### Key Classes:

- **`BackendType`**: Enum for communication backends (GLOO, NCCL, TCP)

- **`RoleInfo`**: Represents a node in the distributed system

  - `role_name`: Logical group (e.g., "actor", "rollout", "trainer")
  - `rank`: Process ID within the role
  - `world_size`: Total number of processes in this role
  - `ip`, `port`: Network address for P2P communication
  - `device_id`: GPU ID if applicable
  - `metadata`: Custom attributes (tensor parallelism, pipeline parallelism, etc.)
  - Property `node_id`: Format `"{role_name}_{rank}"`
  - Property `address`: Format `"{ip}:{port}"`

- **`DCSConfig`**: Main configuration class with settings for:

  - **Coordinator**: Host, port
  - **Communication**: Backend type (default GLOO), TCP buffer sizes, tensor fusion threshold
  - **Heartbeat**: Heartbeat interval, timeout
  - **Storage**: Checkpoint directory, async I/O
  - **Fault Tolerance**: Max retries, retry delays
  - **Observability**: Metrics enablement, Prometheus port

- **`TopologyConfig`**: Defines role-to-role connections

  - `role_mappings`: E.g., `{"actor": "rollout"}` means actor_rank N connects to rollout_rank N
  - `get_peer_role(role)`: Get the peer role for a given role

#### Example Configuration:

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

### 2. **Metrics** (`metrics.py`)

Production-grade observability with Prometheus export.

#### Metric Types:

**Histograms** (latency tracking):

- `dcs_save_latency_seconds`: Time to save checkpoint
- `dcs_load_latency_seconds`: Time to load checkpoint
- `dcs_send_latency_seconds`: Time to send tensors
- `dcs_recv_latency_seconds`: Time to receive tensors

**Counters** (monotonic):

- `dcs_bytes_sent_total`, `dcs_bytes_received_total`
- `dcs_bytes_saved_total`, `dcs_bytes_loaded_total`
- `dcs_*_operations_total`: Operation counts
- `dcs_errors_total`: Total errors

**Gauges** (point-in-time):

- `dcs_memory_buffer_usage_bytes`: Current buffer memory
- `dcs_active_connections`: Open connections
- `dcs_pending_operations`: In-flight operations

#### Key Classes:

- **`Histogram`**: Latency tracking with configurable buckets

  - `observe(value)`: Record a sample
  - `get_stats()`: Returns count, sum, avg, bucket distribution
  - Default buckets (seconds): `[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]`

- **`Counter`**: Monotonically increasing counter

  - `inc(value)`: Increment by value
  - Thread-safe with lock

- **`Gauge`**: Value that can go up/down

  - `set(value)`, `inc(value)`, `dec(value)`
  - Thread-safe

- **`MetricsCollector`**: Main collector

  - `record_save(bytes_saved, duration)`
  - `record_load(bytes_loaded, duration)`
  - `record_send(bytes_sent, duration)`
  - `record_recv(bytes_received, duration)`
  - `record_error(error_type)`
  - `export_prometheus()`: Export in Prometheus text format
  - `get_all()`: Get all metrics as dict

#### Usage:

```python
metrics = MetricsCollector()
metrics.record_send(bytes_sent=1024*1024, duration=0.05)
print(metrics.export_prometheus())  # Prometheus format

# Global instance
from relax.distributed.checkpoint_service.metrics import get_metrics
metrics = get_metrics()
metrics.record_save(bytes_saved=5*1024*1024, duration=1.2)
```

______________________________________________________________________

### 3. **Communication Backends** (`backends/`)

Abstract base class with one concrete implementation.

#### Architecture:

```
CommBackend (Abstract Base Class)
└── DeviceDirectBackend (NCCL/GLOO)
    └── For intra-cluster GPU-to-GPU or CPU communication
```

#### Base Classes (`backends/base.py`):

- **`SendRequest`**: Point-to-point send descriptor

  - `tensor_dict`: Tensors to send
  - `dst_rank`: Destination rank
  - `group_name`: Optional process group name
  - `async_op`: Blocking vs async flag
  - `metadata`: Extra data

- **`RecvRequest`**: Point-to-point receive descriptor

  - `src_rank`: Source rank
  - `tensor_names`: Expected tensor names
  - `group_name`: Optional process group name
  - `metadata`: Extra metadata

- **`CommHandle`**: Async operation handle

  - `request_id`: Unique operation ID
  - `is_complete`: Completion status
  - `result`: Operation result
  - `error`: Exception if failed
  - `wait()`: Blocking wait
  - `async wait_async()`: Async wait

- **`CommBackend`** (ABC): Unified communication interface

  - `broadcast()`: One-to-all broadcast
  - `broadcast_async()`: Async broadcast
  - `create_group()`: Create communication group
  - `destroy_group()`: Destroy communication group
  - `register_peer()`: Register a peer node
  - `init_process_group()`: Initialize distributed communication

- **`TensorFusion`**: Optimizer for many small tensors

  - Concatenates multiple small tensors into one large buffer
  - Reduces protocol overhead
  - Configurable threshold (default 1MB)
  - `should_fuse(tensor_dict)`: Check whether fusion should be applied
  - `fuse(tensor_dict)`: Fuse tensors, returns (fused_tensor, metadata)
  - `unfuse(fused_buffer, metadata)`: Unfuse back to original tensors

#### 3.1 Device-Direct Backend (`device_direct.py`)

High-performance backend using PyTorch distributed (NCCL for GPU, GLOO for CPU).

**Constructor Parameters:**

- `args`: Backend arguments
- `backend_type`: GLOO or NCCL
- `role_info`: Current node information
- `model`: Model instance sequence
- `model_name`: Model identifier
- `quantization_config`: Optional quantization settings
- `coordinator_url`: Coordinator URL
- `lock`: Remote lock (for coordinating weight updates)
- `timeout_seconds`: Operation timeout (default 300)

**Key Methods:**

- `init_process_group_for_rollout(topology_data)`: Initialize process group with rollout nodes
- `init_process_groups_for_actor_fwd_ref(topology_data)`: Initialize actor → actor_fwd weight sync process groups
- `update_weights_for_rollout(rollout_only, actor_fwd_only)`: Update weights on rollout/actor_fwd nodes
- `recv_weight()`: Receive weight broadcasts on actor_fwd side

**Features:**

- NCCL: GPU collective communication with optimal bandwidth
- GLOO: CPU-based fallback with async support
- CUDA stream integration for overlap with computation
- All-gather, broadcast, and point-to-point operations
- Async operations with completion handles
- HTTP communication with rollout nodes via Ray Actor (`RolloutEngine`)

**Use Cases:**

- Multiple GPUs on same node (NVLink, PCIe)
- Multi-node GPU cluster with InfiniBand/Ethernet

______________________________________________________________________

### 4. **Client** (`client/engine.py`)

Data plane client for checkpoint operations.

#### Responsibilities:

- **Registration**: Register with coordinator, obtain rank
- **Peer Discovery**: Fetch topology, identify peers
- **Weight Updates**: Synchronize model weights with rollout/actor_fwd nodes
- **Heartbeat**: Signal health to coordinator

#### Key Classes:

- **`CheckpointEngineClient`**: Main client class

  - `args`: Command-line arguments object
  - `coordinator_url`: Coordinator endpoint
  - `role_info`: Node metadata (role, rank, device, IP, port)
  - `backend_type`: Communication backend (NCCL/GLOO)
  - `model`: Model reference
  - `model_name`: Model name
  - `quantization_config`: Quantization settings
  - `lock`: Remote lock

#### Key Methods:

- `async start()`: Initialize and register

  1. Create HTTP client
  2. Register with coordinator
  3. Initialize communication backend

- `async stop()`: Shutdown gracefully

  - Cancel heartbeat
  - Close backend
  - Close HTTP client

- `async init_process_groups_for_actor_fwd_ref(rollout_id)`: Initialize actor/actor_fwd weight sync

  - Checks whether ref update is needed based on `ref_update_interval`
  - Fetches model update group rank mappings from coordinator
  - Calls backend to establish process groups

- `async recv_weight_fully_async()`: Receive weights asynchronously on actor_fwd side

- `async update_weights_for_rollout(rollout_only, actor_fwd_only)`: Update rollout weights

  - Fetches topology
  - Initializes rollout process group
  - Calls backend to transfer weights

#### Properties:

```python
client.role          # Logical role name
client.rank          # Rank within role
client.world_size    # Total processes in role
client.node_id       # Unique identifier
client.is_registered # Registration status
client.backend       # Communication backend instance
```

#### Example Usage:

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

    # Register with coordinator
    print(f"Registered as {client.node_id}")

    # Update rollout weights
    await client.update_weights_for_rollout()

    # Initialize actor_fwd weight sync
    await client.init_process_groups_for_actor_fwd_ref(rollout_id=100)

    await client.stop()

asyncio.run(main())
```

Helper function:

```python
from relax.distributed.checkpoint_service.client import create_client

# Create and start a client
client = await create_client(
    args=args,
    coordinator_url="http://localhost:8000",
    role="actor",
    rank=0,
)
```

______________________________________________________________________

### 5. **Coordinator** (`coordinator/`)

Control plane service for topology management.

#### Architecture:

```
DCSCoordinator (FastAPI + Ray Serve)
├── TopologyManager
│   ├── Node Registration
│   ├── Rank Assignment
│   ├── Peer Lookup
│   └── Heartbeat Monitoring
└── REST Endpoints
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

#### 5.1 Coordinator Service (`service.py`)

FastAPI-based REST API for topology and weight update management, deployed via Ray Serve.

**Endpoints:**

- `POST /register`: Register a new node

  - Input: `RegisterRequest` (role_name, rank, world_size, ip, port, device_id, metadata)
  - Output: `RegisterResponse` (status, message, rank, node_id)
  - Returns assigned rank

- `DELETE /unregister`: Deregister a node

  - Parameters: `role`, `rank`

- `GET /heartbeat`: Update node heartbeat

  - Parameters: `role`, `rank`
  - Output: `HeartbeatResponse` (status, timestamp)

- `GET /topology`: Get current topology

  - Parameters: `role_filter` (optional)
  - Output: `TopologyResponse` (nodes, world_size)
  - Returns full role->rank mapping

- `GET /peer`: Get peer for a node

  - Parameters: `role`, `rank`, `peer_role` (optional)
  - Output: Peer's `RoleInfo` dict

- `GET /node`: Get specific node info

  - Parameters: `role`, `rank`
  - Output: `RoleInfo` dict

- `GET /global_rank`: Get global rank

  - Parameters: `role`, `rank`
  - Output: `{"global_rank": int}`

- `GET /get_model_update_group_ranks`: Get communication groups for weight updates

  - Parameters: `role`, `rank`, `need_update_ref`
  - Output: `GroupRanksResponse` (global_rank, world_size, train_pp_size, pp_groups)
  - Computes global rank and PP groups based on actor/actor_fwd/reference roles

- `POST /send_weight_meta`: Send weight metadata

  - Input: `SendWeightMetaRequest` (names, dtypes, shapes, group_name)
  - Output: `Response` (status, message)

- `GET /recv_weight_meta`: Receive weight metadata

  - Parameters: `index`
  - Output: List of weight metadata starting from index

- `GET /clear_weight_meta`: Clear weight metadata buffer

- `GET /health`: Health check

  - Output: Status, timestamp, world_size, list of dead nodes

- `GET /debug/topology`: Full topology details for debugging

**API Models:**

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

**Deployment:**

The DCS coordinator is deployed via Ray Serve:

```python
from relax.distributed.checkpoint_service.coordinator.service import create_dcs_deployment

coordinator, coordinator_url = create_dcs_deployment(config)
```

Or directly with Ray Serve:

```python
from ray import serve

serve.run(DCSCoordinator.bind(config=config), name="dcs_coordinator", route_prefix="/dcs_coordinator")
```

#### 5.2 Topology Manager (`topology.py`)

In-memory topology database with lifecycle management.

**Features:**

- **Role Registration**: Assign ranks to nodes
- **Peer Lookup**: Find peer for role-role connection
- **Global Rank Mapping**: Logical to physical rank translation
- **Heartbeat Tracking**: Monitor node health
- **Dynamic Updates**: Support elastic scaling
- **Thread Safety**: All methods are thread-safe via RLock

**Key Classes:**

- **`TopologyNode`**: Node representation

  - `role_info`: Node metadata
  - `last_heartbeat`: Timestamp of last heartbeat
  - `is_alive`: Health status
  - `connections`: Set of peer node_ids

- **`TopologyManager`**: Topology database

  - `register(role_info)`: Add node and assign rank
  - `unregister(role_name, rank)`: Remove node
  - `heartbeat(role_name, rank)`: Update heartbeat
  - `get_node(role_name, rank)`: Get node info
  - `get_peer(role_name, rank, peer_role)`: Find peer
  - `get_role_nodes(role_name)`: Get all nodes in role
  - `get_all_nodes()`: Get full topology
  - `get_world_size(role_name=None)`: Total nodes (filterable by role)
  - `get_global_rank(role_name, rank)`: Get global rank
  - `get_all_peers(role_name, rank)`: Get all peers
  - `check_health()`: Check all node health
  - `to_dict()`: Export topology as dict

**Example Usage:**

```python
manager = TopologyManager(
    config=TopologyConfig(role_mappings={"actor": "rollout"}),
    heartbeat_timeout=30.0
)

# Register nodes
manager.register(RoleInfo(role_name="actor", rank=0, ip="10.0.0.1", port=20000))
manager.register(RoleInfo(role_name="rollout", rank=0, ip="192.0.2.2", port=20001))

# Get peer
peer = manager.get_peer("actor", 0, "rollout")
print(f"Actor 0 should connect to Rollout 0 at {peer.address}")

# Heartbeat
manager.heartbeat("actor", 0)
```

______________________________________________________________________

## Data Flow

### Weight Update Flow (Actor → Rollout)

```
Actor (Training)
    ↓
    └─→ Coordinator
        ├─ Register
        └─ Get topology
    ↓
DeviceDirectBackend
    ├─ init_process_group_for_rollout()
    ├─ all_gather_param() (TP gather)
    ├─ convert_to_hf() (weight conversion)
    └─ dist.broadcast() (broadcast to rollout)
    ↓
Rollout Nodes (HTTP communication via RolloutEngine Ray Actor)
```

### Weight Update Flow (Actor → Actor FWD/Reference)

```
Actor (Training)
    ↓
    └─→ Coordinator
        ├─ Register
        └─ get_model_update_group_ranks (get PP groups)
    ↓
DeviceDirectBackend
    ├─ init_process_groups_for_actor_fwd_ref()
    ├─ all_gather_param() (TP gather)
    ├─ send_weight_meta (send metadata via coordinator)
    └─ dist.broadcast() (broadcast weights)
    ↓
Actor FWD / Reference (receive via recv_weight())
```

______________________________________________________________________

## Configuration Examples

### Single-Node Multi-GPU

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

### Multi-Node GPU Cluster

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

## Performance Tuning

### Tensor Fusion

Fusion reduces overhead by combining small tensors:

```python
config.tensor_fusion_threshold = 1024 * 1024  # 1MB
# Only fuse if total tensors >= 1MB and count > 1
```

### Pinned Memory

Enables async GPU-to-CPU transfer:

```python
config.pinned_memory = True  # Default, recommended for GPU
```

### TCP Settings

```python
config.tcp_nodelay = True              # Disable Nagle's algorithm
config.tcp_buffer_size = 65536         # 64KB buffers
config.comm_base_port = 20000          # Base port
```

### Heartbeat

Adjust for network reliability:

```python
config.heartbeat_interval_seconds = 5.0   # Every 5 seconds
config.heartbeat_timeout_seconds = 30.0   # 30 second deadline
```

______________________________________________________________________

## Fault Tolerance

### Node Failure Detection

1. Client stops sending heartbeats
2. Coordinator marks node as dead (after heartbeat_timeout)
3. Topology is updated
4. Remaining nodes can continue with resharded topology

### Automatic Retry

```python
config.max_retries = 3
config.retry_delay_seconds = 1.0  # Exponential backoff
```

### Graceful Shutdown

```python
await client.stop()  # Clean shutdown
```

______________________________________________________________________

## Monitoring & Observability

### Metrics Export

```python
from relax.distributed.checkpoint_service.metrics import get_metrics

metrics = get_metrics()

# Get as dict
stats = metrics.get_all()
print(f"Total bytes sent: {stats['counters']['bytes_sent']}")
print(f"Avg send latency: {stats['latency']['send']['avg']:.3f}s")

# Export for Prometheus
prom_text = metrics.export_prometheus()
# Write to Prometheus endpoint
```

### Logging

All components use the framework logging utility:

```python
from relax.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Logs automatically include module information
logger.info("Checkpoint saved successfully")
```

______________________________________________________________________

## Advanced Topics

### Elastic Scaling

The system supports dynamic topology changes:

1. New node registers with coordinator
2. Coordinator assigns rank
3. Client fetches new topology and establishes process groups
4. Existing communication groups are updated

### Tensor Parallelism Resharding

The `get_model_update_group_ranks` endpoint handles:

- Training PP (Pipeline Parallel) size
- Global rank computation for actor_fwd and reference nodes
- Process groups for PP synchronization (one group per PP stage)
- Automatic group formation based on parallelism configuration

### Custom Metadata

Store extra information on nodes:

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

## Network Port Allocation

Each service reserves a dedicated port range to avoid conflicts during process group initialization (TCPStore). The full allocation map:

| Service | Port Range | Usage |
|---------|-----------|-------|
| DCS weight sync (Actor → Rollout) | 11000 - 11999 | `DeviceDirectBackend` TCPStore for NCCL/GLOO broadcast |
| Rollout (SGLang engine) | 15000 - 15999 | SGLang inference engine HTTP server |
| GenRM (SGLang engine) | 16000 - 16999 | GenRM inference engine HTTP server |

**Megatron NCCL port range**: Megatron-LM's internal NCCL communication uses the OS ephemeral port range. To avoid collisions with the service ports above, it is recommended to shrink the ephemeral range:

```bash
echo "32768 50000" > /proc/sys/net/ipv4/ip_local_port_range
```

This confines OS-assigned ephemeral ports to 32768-50000, well above the reserved service ranges.

> **Why fixed ranges?** The original implementation used OS-assigned random ports (`bind(0)`), which could collide with other services or linger in `TIME_WAIT`, causing `EADDRINUSE` failures during weight sync. Fixed ranges with pre-bind validation eliminate this class of errors.

______________________________________________________________________

## Troubleshooting

### Connection Issues

- Check firewall rules for P2P ports (base_port to base_port + max_ranks)
- Verify coordinator is reachable
- Check IP/port configuration in RoleInfo

### Heartbeat Timeouts

- Increase `heartbeat_timeout_seconds` for unstable networks
- Check network latency between coordinator and clients
- Monitor `dcs_errors_total` metrics

### Low Throughput

- Enable tensor fusion for many small tensors
- Use NCCL backend for same-node GPU communication
- Check network bandwidth with `dcs_bytes_sent_total / time`

### Memory Pressure

- Monitor `dcs_memory_buffer_usage_bytes` gauge
- Reduce `tcp_buffer_size` if memory constrained
- Disable `pinned_memory` on CPU-only systems

______________________________________________________________________

## Bibliography & References

- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
- Prometheus Metrics: https://prometheus.io/docs/
