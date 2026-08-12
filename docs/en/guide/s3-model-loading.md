# S3 Model Loading

Relax can load a Hugging Face checkpoint directly from S3-compatible object storage. Passing an `s3://` URI to `--hf-checkpoint` makes each node download the checkpoint into shared memory (`/dev/shm`) at startup and then read it as an ordinary local path. No separate enable flag is needed; a local path bypasses this path entirely.

### When to use S3 loading

S3 loading replaces the step of reading the checkpoint from its existing filesystem, so whether it is faster comes down to comparing the two read bandwidths:

| Setup | Recommendation |
|---|---|
| The object store delivers more read bandwidth than the shared filesystem holding the checkpoint (NFS and similar) | Use `s3://`. This is the case S3 loading is built for |
| Every node already has the checkpoint on fast local NVMe | Use the local path, since a local read usually beats a network download |
| The object store is bandwidth-limited or far from the cluster | Measure first; pulling tens of GB per node can cost more than it saves |

Downloads run in parallel, and each node downloads once regardless of how many ranks it hosts.

### Prerequisites

- The checkpoint is reachable at an `s3://` URI
- Each node has enough `/dev/shm` to hold the full checkpoint during startup
- For SGLang streaming modes, the image ships an SGLang build with `runai_streamer`

______________________________________________________________________

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  S3 Object Store        s3://bucket/model-prefix/               │
└────────────────────────────┬────────────────────────────────────┘
                             │  parallel download, once per node
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Node shared memory (/dev/shm)                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  full checkpoint = weights + config + tokenizer           │  │
│  └───────────────┬───────────────────────┬───────────────────┘  │
│                  │                       │                      │
│                  ▼                       ▼                      │
│  ┌────────────────────────────┐  ┌─────────────────────────┐    │
│  │  Megatron Actor            │  │  SGLang Rollout Engine  │    │
│  │  always loads full weights │  │  dummy / auto / stream  │    │
│  └────────────────────────────┘  └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

The training Actor always loads the full weights. The rollout engine is the only part to tune, via `--sglang-load-format`.

______________________________________________________________________

## Quick Start

```bash
python3 -m relax.entrypoints.train \
    --hf-checkpoint s3://bucket/model-prefix/ \
    --sglang-load-format dummy \
    # ... other training arguments
```

`--s3-model-endpoint` is only needed for a self-hosted or S3-compatible store.

### Why `dummy`

In RL training the Actor synchronizes its weights to the rollout engine before the first rollout, so whatever SGLang loaded at startup is immediately overwritten. Loading real weights at startup is therefore pointless. `dummy` makes SGLang read only the model structure and metadata, and the Actor supplies the real weights afterwards.

This mode applies to the policy rollout engine only, as that is the engine the Actor synchronizes. Other roles keep loading real weights — see the warning below.

______________________________________________________________________

## Load Formats and Recommended Combinations

`--sglang-load-format` controls the rollout engine only, and defaults to `auto`. Pick by scenario:

| Scenario | Configuration | Effect |
|---|---|---|
| **Standard RL training** (recommended) | `--sglang-load-format dummy` | Rollout engine reads metadata only and starts fastest. The Actor pushes real weights before the first rollout, then SHM is freed. |
| **Engine startup order varies** | `--sglang-load-format auto` | Reuses the `/dev/shm` copy when it is ready, otherwise streams from S3. SHM stays occupied for the whole job. |
| **Rollout starts before training, or standalone SGLang** | `--sglang-load-format runai_streamer` | Streams weights straight from S3. Rollout never touches SHM, and SHM is freed after the Actor's first sync. |
| **Debugging, need the files on disk** | add `--disable-s3-model-cleanup` | Full checkpoint stays in `/dev/shm` for the whole job. |
| **Turn the feature off** | `--disable-s3-model-download` | Relax loads `--hf-checkpoint` the ordinary way. |

::: warning
`dummy` applies to the policy rollout engine only, because the Actor overwrites its weights before the first request. GenRM, teacher, evaluation, and standalone inference services always load real weights, even when they point at the same model.
:::

The larger the checkpoint, the more startup time `dummy` saves — but the choice itself does not depend on model size, so pick by scenario from the table above.

______________________________________________________________________

## Startup Sequence

```
Job submitted
  → every node downloads the checkpoint into /dev/shm (once per node)
  → SGLang rollout engine starts
       dummy           reads metadata only, fastest to become ready
       auto            reuses /dev/shm, no second download
       runai_streamer  streams from S3, does not use SHM
  → Actor performs the first weight sync (a dummy engine receives real weights here)
  → weight shards in /dev/shm are released automatically
  → training starts
```

______________________________________________________________________

## Memory Planning

`/dev/shm` consumes real Pod memory, and the full checkpoint must fit there during startup. Size the Pod as:

```
Pod memory  ≥  checkpoint size  +  peak RL runtime memory  +  margin
```

The checkpoint size is the sum of all files under the S3 prefix. The full checkpoint must fit in `/dev/shm`, otherwise the job fails at startup.

After the first sync, Relax frees the weight shards and keeps only config, tokenizer, and processor files — but only in modes where nothing will read the weights again:

| Configuration | SHM weights released after training starts |
|---|---|
| `--sglang-load-format dummy` | Yes |
| `--sglang-load-format runai_streamer` | Yes |
| `--sglang-load-format auto` | No |
| Jobs with no rollout service (for example SFT) | Yes |
| `--disable-s3-model-cleanup` | No |

External rollout engines and per-engine SGLang configurations also keep the checkpoint in place.

______________________________________________________________________

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `--hf-checkpoint` | — | An `s3://` URI enables S3 loading; a local path bypasses it |
| `--sglang-load-format` | `auto` | Rollout engine load mode, see the table above |
| `--s3-model-endpoint` | `None` | Endpoint of a self-hosted or S3-compatible store |
| `--s3-model-use-path-style` | off | Enable when the gateway requires path-style addressing |
| `--s3-model-use-placeholder-credentials` | off | Enable when the gateway requires signed requests but not real credentials |
| `--s3-model-shm-root` | `/dev/shm` | Shared-memory directory used for the download |
| `--s3-model-download-workers` | `20` | Download concurrency |
| `--disable-s3-model-download` | off | Load `--hf-checkpoint` the ordinary way instead |
| `--disable-s3-model-cleanup` | off | Keep the weights in shared memory for the whole job |

Credentials are never passed on the command line; Relax uses the standard credential chain of the environment.

______________________________________________________________________

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| Job fails at startup with insufficient shared memory | The checkpoint does not fit in `/dev/shm` | Raise the Pod memory and `/dev/shm` size; see [Memory Planning](#memory-planning) |
| Job fails saying the shared-memory directory is missing | `/dev/shm` is not mounted on some node | Check the container mount on every node, not just rank 0 |
| SGLang rejects `runai_streamer` | The installed SGLang does not provide that loader | Upgrade to an image whose SGLang exposes `runai_streamer` |
| GenRM or teacher produces garbage output | Assuming `dummy` applies to them | `dummy` only affects the policy rollout engine; the other roles load real weights |
| Memory is not released after training starts | Running `auto`, external rollout, or `--disable-s3-model-cleanup` | Switch to `dummy` to reclaim the memory |
| Startup is slower than with a local path | The object store gives less read bandwidth than the local filesystem | Compare the two; a local path may be the better choice for this cluster |

______________________________________________________________________

## Further Reading

- [Configuration](./configuration.md) — full CLI parameter reference
- [Update Weights Pipeline](./update-weights-pipeline.md) — how the Actor pushes real weights to a `dummy`-loaded engine
