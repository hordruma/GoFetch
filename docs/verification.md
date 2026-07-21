# Verification log

What has actually been run, on what hardware, with what result. Anything not
listed here is unverified.

## Rig

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop, 8 GB VRAM, driver 580.173.02 |
| RAM | 62 GB, `RLIMIT_MEMLOCK` 7.8 GB |
| tmpfs | `/dev/shm`, 32 GB |
| Backend | llama.cpp `b10075`, **Vulkan** build (see "CUDA gap" below) |
| Models | Qwen2.5-0.5B-Instruct Q4_K_M (491 MB), Qwen2.5-1.5B-Instruct Q4_K_M (1.12 GB) |
| Go | 1.24.5 |

## Status by milestone

| Milestone | State | Evidence |
|---|---|---|
| M0 skeleton | ✅ verified | `go build` / `go vet` / `go test -race` clean; all 6 internal packages have tests |
| M1 real swap | ✅ verified | cold load 0.80 s; auto-swap 1.40 s; single-slot invariant holds under a 12-way concurrent load storm |
| M3 doc-KV | ✅ verified | 2 925-token document: prefill 490 ms → **24 ms** after restore (20×) |
| M4 llama-swap | ✅ verified | force-load 1.26 s; `/running` agrees with GoFetch; drop frees VRAM 493 → 2 MiB |
| M-deep embedded | ❌ blocked | no CUDA toolkit on this machine; see below |
| S streaming-layers | ❌ not started | research tier, see `streaming-layers.md` |

## Measurements

Swap, `direct` engine (process spawn + model load + first token):

| Operation | Time |
|---|---|
| Load Qwen-0.5B onto an empty GPU | 0.80 s |
| Auto-swap 0.5B → 1.5B via a plain OpenAI request | 1.40 s |
| Swap back to 1.5B (warm page cache) | 1.10 s |
| Force-load via llama-swap engine | 1.26 s |

Document KV snapshots, Qwen-1.5B, 2 925-token document prefix:

| Operation | Result |
|---|---|
| Snapshot size | 84.1 MB for 2 932 tokens (~28.7 KB/token) |
| Save to `/dev/shm` | 39.6 ms |
| Restore | 28.9 ms |
| Prefill **without** snapshot | 489.8 ms |
| Prefill **with** snapshot | 24.4 ms (1 token evaluated) |

The net win is ~440 ms per repeat query over that document. Snapshot size scales
with model and context, so the KV budget is the binding constraint: at 28.7
KB/token an 8 GB budget holds roughly 95 documents of this size.

First inference after a cold start costs an extra ~5.5 s of Vulkan shader
compilation; it does not recur.

## The CUDA gap

The machine has driver 580 / CUDA 13 but **no CUDA toolkit** (`nvcc` absent), and
llama.cpp publishes no prebuilt Linux CUDA binaries — only Windows. So:

- `-tags cuda` (the embedded engine and `gofetch-probe`) cannot be built here.
- The H2D bandwidth probe in `docs/dma-and-swap.md` is unmeasured.
- All GPU numbers above come from the **Vulkan** backend, which does not use the
  pinned-memory DMA path the embedded engine is designed around. They establish
  that the mechanics work; they say nothing about the pinned-DMA thesis.

Closing this needs a CUDA toolkit (12.8+ for the 5060's Blackwell arch) and a
llama.cpp built against it.

## Reproducing

```sh
go test ./...                      # unit + regression
./gofetch -config config.json      # then drive the API; see README
```

The end-to-end runs above used a stub llama-server for fault injection (backend
that never becomes healthy, backend that dies on launch) and the real llama.cpp
binary for everything else.
