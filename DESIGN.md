# GoFetch — Design

GoFetch is the **mechanical memory plane** for a local LLM stack. It loads,
warms, drops, and swaps models across a small VRAM budget and a large RAM pool,
and it owns a document-level **KV cache store**. It does **not** decide *what*
to load or *when* — that is the job of an external orchestrator/router (the
[BigLaw](https://github.com/discover-legal/BigLaw) agentic system). BigLaw makes
the policy decisions; GoFetch executes the mechanics behind a clean API.

## Target hardware

- **GPU:** ~8 GB VRAM (laptop RTX 5060). Holds roughly **one** 7–8B model at
  Q4 (~4.5–5 GB) plus its context/KV. "One resident model" is the ceiling.
- **RAM:** 64 GB. Can hold **8–12 quantized 7B GGUFs resident** in the page
  cache at once. This is the real leverage of the setup.
- **NPU:** present but, as of this writing, immature for general LLM decode
  (Intel AI Boost / AMD XDNA). Not part of v1. A future option is offloading the
  embedding model to it via OpenVINO/Ryzen-AI.

## The hard constraint that shapes everything

A running `llama-server` is **bound to its GPU offload configuration at
startup**. You cannot live-migrate a model's tensors from RAM into VRAM in
place. Therefore a "hot swap" is mechanically a **process stop + start**, not a
memcpy. GoFetch makes that swap fast with two tricks:

1. **Page-cache warming.** The target GGUF is pulled into (and optionally
   `mlock`ed in) the OS page cache ahead of time, so the new `llama-server`
   loads at RAM→VRAM speed instead of disk→VRAM speed.
2. **RAM-backed KV cache.** KV snapshots are saved to a **tmpfs** mount
   (`/dev/shm`), so save/restore is memory-speed.

**Expected swap latency:** ~1–2 s for a 7B from a warm page cache, vs ~5–10 s
cold. Sub-100 ms swaps are not physically possible here. BigLaw's scheduler
should treat a swap as a ~1–2 s operation and pre-warm accordingly.

## KV cache is model-bound *and* position-bound

A saved KV snapshot is only valid for the **exact model** that produced it and
the **exact token positions** it occupied. Consequences:

- A doc's KV for model A is useless for model B.
- If the prefix before a cached span changes, the cache is invalid.

**Design rule:** cache doc prefixes against a **stable RAG generator** (the
model BigLaw keeps resident for retrieval-augmented generation). Hot-swap only
*specialist* models (e.g. a reranker, a function-calling model) for non-RAG
sub-tasks. Do not swap the model you are caching against, or you throw the cache
away on every swap.

## Components

| Component          | Responsibility                                            | Mechanism                                  |
|--------------------|-----------------------------------------------------------|--------------------------------------------|
| VRAM slot manager  | Owns the single GPU-resident model; enforces 8 GB budget  | start/stop `llama-server -ngl 99`          |
| RAM warm-pool      | Keeps N GGUFs hot in RAM; two tiers, LRU evict per tier    | `mmap` + page fault (+ `mlock` for locked) |
| KV cache store     | Save/restore per-`(model, key)` KV snapshots              | llama.cpp `--slot-save-path` on tmpfs      |
| Control API        | What the orchestrator calls                               | HTTP/JSON                                  |
| Inference proxy    | OpenAI-compatible passthrough to the active model         | reverse proxy (à la llama-swap)            |

### Warm-pool tiers

The warm-pool has two independently-budgeted tiers so that promoting a hot model
to fast-swap status never evicts the large background set:

| Tier       | Mechanism                      | Cost              | Capacity   | Buys                                   |
|------------|--------------------------------|-------------------|------------|----------------------------------------|
| `cache`    | `mmap` + page-fault            | cheap (pageable)  | many (8–12)| skips the disk read on load            |
| `locked`   | `cache` + `mlock`              | locks real RAM    | few (2–4)  | residency + foundation for fast DMA    |

**Crucial distinction:** `mlock` (the `locked` tier) is *not* the same as
CUDA-pinned memory. `mlock` guarantees the weights stay resident and removes
page-fault stalls, but a CUDA upload from an `mlock`'d-but-otherwise-pageable
region still takes the driver's pageable→staging→VRAM path. The *fast DMA* win
(`cudaHostRegister` → direct DMA at the PCIe ceiling) requires being inside the
inference process — see **M-deep** below. The `locked` tier is the buffer that
in-process engine will register.

### Vendor-neutral by design

GoFetch must not hard-bind to CUDA: NVIDIA and AMD now, Intel/Apple later. The
biggest reason to build *on* llama.cpp rather than a bespoke CUDA engine is that
it already ships CUDA / HIP-ROCm / Vulkan / SYCL / Metal backends, so multi-vendor
support is mostly a build choice. The in-process (M-deep) engine must therefore be
**backend-abstracted** — see [docs/backends.md](./docs/backends.md), which also
covers using the NPU as an auxiliary lane for always-on small models (embeddings,
reranker, router, guards).

### Engine: build on llama-swap

[llama-swap](https://github.com/mostlygeek/llama-swap) already solves
process-swap + OpenAI proxy well, in Go. GoFetch's default engine **drives
llama-swap** and adds the two things it lacks: the **warm-pool** and the **doc-KV
store**. A `direct` engine (GoFetch spawns `llama-server` itself) is also
provided for full control without the dependency. An opt-in `embedded` engine
(below) goes in-process for the fast-swap path. The choice is a single config
key (`engine.type`): `llama-swap` | `direct` | `embedded`.

## M-deep: the in-process engine (the deeper surgery)

The supervisor engines (`llama-swap`, `direct`) treat a model as an external
process. That ceilings what we can do, because three wins live *inside* the
inference process and its CUDA context:

1. **Fast DMA.** `cudaHostRegister()` the `locked`-tier region, then upload to
   VRAM at the PCIe ceiling instead of the pageable path. **Measured on the 5060
   (see docs/verification.md): 21.7 vs 15.4 GB/s (~1.4×), and register-in-place
   is ~2× the copy-based pinned path per swap** because it skips a per-load
   RAM→RAM copy. This only works because the `locked` tier maps the GGUF
   copy-on-write (`PROT_READ|PROT_WRITE, MAP_PRIVATE`): the driver refuses to
   register a read-only or `MAP_SHARED` file mapping. `mlock` is orthogonal — it
   guarantees residency and removes page-fault stalls, but does not make a region
   registrable.
2. **No-restart swap.** Keep one process + one CUDA context alive across swaps;
   `llama_model_free()` the outgoing model and `llama_model_load_from_file()`
   the incoming one — skipping process spawn, context init, and GGUF re-parse.
3. **Overlapped upload.** Stage weights over multiple CUDA streams so transfer
   overlaps buffer/graph setup.

The `embedded` engine (`internal/engine/embedded_cuda.go`, build tag `cuda`)
links llama.cpp via cgo to do this. It is a **development skeleton today** — the
default build ships a stub that errors with a rebuild hint, and the CUDA file is
pure-Go scaffolding with the integration points marked `TODO(M-deep)`.

**What the floor is, and isn't.** Even fully built, a swap still costs the
**PCIe upload of the weights** (~0.4–1 s for a 7B) plus buffer setup. The
surgery removes the *overhead around* that upload (restart, re-parse, pageable
staging) and gets the upload itself to the hardware ceiling. It does **not**
make swaps free — that requires unified-memory hardware (Apple Silicon,
Grace-Hopper), which a discrete 5060 is not.

**Related but separate:** *streaming-layers* inference (run a model bigger than
VRAM by streaming its layers from RAM) is a different axis from swap latency — a
planned escalation tier, documented in
[docs/streaming-layers.md](./docs/streaming-layers.md).

**Build & verification note.** The CUDA path cannot be compiled or benchmarked
in a CPU-only CI sandbox (no GPU, no CUDA toolkit, no vendored llama.cpp). It
must be built with `-tags cuda` against a llama.cpp checkout on a machine with
the 5060 + CUDA toolchain, and the swap timings measured there.

## Control API (the BigLaw integration seam)

```
GET  /v1/status                       → {vram, warm[], kv[], budgets}
POST /v1/vram/load    {model}         → make model the VRAM resident (swaps out current)
POST /v1/vram/drop    {model}
POST /v1/ram/warm     {model}         → pin GGUF in page cache
POST /v1/ram/evict    {model}
POST /v1/kv/save      {model, key}    → snapshot doc-prefix KV to tmpfs
POST /v1/kv/load      {model, key}
POST /v1/kv/evict     {model, key}
POST /v1/chat/completions ...         → OpenAI passthrough (auto-loads if needed)
```

All policy (which model, when to warm, when to evict, cache keys) is decided by
the caller. GoFetch only enforces budgets and executes the mechanics.

## Layout

```
cmd/gofetch/         daemon entrypoint
internal/config/     config load (JSON, stdlib only)
internal/models/     model registry
internal/engine/     Engine interface + llama-swap and direct implementations
internal/warm/       RAM warm-pool (mmap/mlock page-cache management)
internal/kvcache/    tmpfs-backed KV snapshot store
internal/api/        HTTP control API + OpenAI proxy
```

## Milestones

Measured results for everything marked ✅ are in [docs/verification.md](./docs/verification.md).

1. ✅ **M0 — skeleton:** packages, API surface, two-tier warm-pool, KV store
   accounting, llama-swap + direct engine stubs. Compiles, unit-tested.
2. ✅ **M1 — real swap:** `direct` engine starts/stops `llama-server`, proxies
   OpenAI traffic, enforces the single-VRAM-slot invariant. Bounded startup
   (the engine mutex is held across a load, so the wait must be finite), backend
   output surfaced to the log, SIGTERM before SIGKILL so VRAM is freed cleanly.
   The OpenAI passthrough auto-loads the model named in the request.
3. ✅ **M3 — doc-KV:** llama.cpp is started with `--slot-save-path` and `-np 1`
   (one slot, so "the prompt cache" is unambiguous); `/v1/kv/save` and
   `/v1/kv/load` drive its slot save/restore, keyed by `(model, key)`, with LRU
   eviction by KV budget. Snapshots are refused unless their model is resident.
4. ✅ **M4 — llama-swap engine:** force-load through llama-swap's direct-access
   route, unload through `POST /api/models/unload/:model`, and `Active()` reads
   back `GET /running` rather than reporting GoFetch's own intent.
5. **M-deep — in-process engine:** cgo + vendored llama.cpp; `cudaHostRegister`
   on the `locked` tier, persistent context, in-place model free/load,
   multi-stream upload. Built and benchmarked on the 5060 (not in CI).
6. **S — streaming-layers tier:** run 30–70B models bigger than VRAM by streaming
   layers from RAM, for latency-tolerant prefill-heavy escalation queries. See
   docs/streaming-layers.md (S0 seam → S1 prototype → S2 integrate).

## Non-goals

- Routing / model-selection logic (lives in BigLaw).
- Retrieval / embedding / chunking (lives in BigLaw).
- Multi-GPU or multi-node scheduling.
