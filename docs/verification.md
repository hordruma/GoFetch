# Verification log

What has actually been run, on what hardware, with what result. Anything not
listed here is unverified.

## Rig

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop, 8 GB VRAM, driver 580.173.02 |
| RAM | 62 GB, `RLIMIT_MEMLOCK` 7.8 GB |
| tmpfs | `/dev/shm`, 32 GB |
| Backend | llama.cpp `b10075`, **Vulkan** build (inference); CUDA 12.9 runtime (probe) |
| Models | Qwen2.5-0.5B-Instruct Q4_K_M (491 MB), Qwen2.5-1.5B-Instruct Q4_K_M (1.12 GB) |
| Go | 1.24.5 |

## Status by milestone

| Milestone | State | Evidence |
|---|---|---|
| M0 skeleton | ✅ verified | `go build` / `go vet` / `go test -race` clean; all 6 internal packages have tests |
| M1 real swap | ✅ verified | cold load 0.80 s; auto-swap 1.40 s; single-slot invariant holds under a 12-way concurrent load storm |
| M3 doc-KV | ✅ verified | 2 925-token document: prefill 490 ms → **24 ms** after restore (20×) |
| M4 llama-swap | ✅ verified | force-load 1.26 s; `/running` agrees with GoFetch; drop frees VRAM 493 → 2 MiB |
| M-deep embedded | 🟡 partial | `-tags cuda` builds; H2D + pinned-DMA thesis measured (below); in-process engine still a skeleton |
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

## The pinned-DMA thesis (measured)

The machine has the NVIDIA driver (Pop!_OS ships it) but no CUDA *toolkit*. The
probe does not need one: its C is compiled by gcc via cgo and only needs the
CUDA *runtime* headers + `libcudart`, which install from pip wheels with no root:

```sh
# nvidia-cuda-runtime-cu12 (headers + libcudart.so) + nvidia-cuda-nvcc-cu12 (crt/ headers)
CGO_CFLAGS="-I<runtime>/include -I<nvcc>/include" \
CGO_LDFLAGS="-L<runtime>/lib -lcudart -Wl,-rpath,<runtime>/lib" \
  go build -tags cuda ./cmd/gofetch-probe
```

The driver does the actual transfers, so a CUDA 12.9 runtime works against the
580 driver, and Blackwell codegen is irrelevant (the probe launches no kernels).

**H2D bandwidth, Qwen-1.5B (1.12 GB), chunked 8 MiB × 10:**

| Path | GB/s | Per-swap cost |
|---|---|---|
| pageable (today, no surgery) | 15.4 | 72 ms |
| pinned, **register-in-place** | **21.7** | 51 ms transfer + 343 ms register **once** at warm |
| pinned, copy-based (`cudaMallocHost`+memcpy) | 21.7 | 51 ms transfer + 51 ms RAM-copy **every** load = 103 ms |

The thesis holds: pinned DMA is ~1.4× pageable, and **register-in-place is ~2×
the copy-based path per swap** because it avoids the per-load RAM→RAM copy — but
only once warmed (the 343 ms `cudaHostRegister` is paid once, not per swap).

### The catch, and the fix

`cudaHostRegister` **refuses** the mapping the warm tier originally used. Isolated
across mapping types (256 MiB each):

| Mapping | Registrable? |
|---|---|
| anonymous / `malloc` (writable) | ✅ |
| file, `MAP_SHARED`, `PROT_READ` — *the old warm-tier flags* | ❌ `invalid argument` |
| file, `MAP_PRIVATE`, `PROT_READ` | ❌ `invalid argument` |
| file, `MAP_PRIVATE`, `PROT_READ\|PROT_WRITE` (COW) | ✅ |

The driver requires a **writable VMA** it can page-lock for DMA; `mlock` makes no
difference. So `warm.TierLocked` now maps the GGUF copy-on-write
(`PROT_READ|PROT_WRITE, MAP_PRIVATE`) — a writable VMA whose pages stay
clean-shared with the page cache, since read-only weights never write-fault. The
cache tier keeps the lighter `PROT_READ, MAP_SHARED`. The probe mirrors the
locked tier, which is why "pinned (register)" now reports a number instead of
"unavailable".

### Still unverified

The **in-process engine itself is still a skeleton** — it compiles under
`-tags cuda` but does not yet `cudaHostRegister` the warm region or run
inference. The numbers above prove the transport thesis on this hardware; wiring
the engine to exploit it (register the locked tier, DMA, persistent context,
in-place model free/load) is the remaining M-deep work. All *inference* numbers
elsewhere in this doc still come from the Vulkan backend.

## Reproducing

```sh
go test ./...                      # unit + regression
./gofetch -config config.json      # then drive the API; see README
make probe && ./gofetch-probe -model M.gguf   # H2D + pinned-DMA numbers (see above for the pip-wheel CGO flags)
```

The end-to-end runs above used a stub llama-server for fault injection (backend
that never becomes healthy, backend that dies on launch) and the real llama.cpp
binary for everything else.
