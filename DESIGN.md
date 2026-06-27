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
| RAM warm-pool      | Keeps N GGUFs hot in page cache; LRU evict by RAM budget   | `mmap` + page fault + optional `mlock`     |
| KV cache store     | Save/restore per-`(model, key)` KV snapshots              | llama.cpp `--slot-save-path` on tmpfs      |
| Control API        | What the orchestrator calls                               | HTTP/JSON                                  |
| Inference proxy    | OpenAI-compatible passthrough to the active model         | reverse proxy (à la llama-swap)            |

### Engine: build on llama-swap

[llama-swap](https://github.com/mostlygeek/llama-swap) already solves
process-swap + OpenAI proxy well, in Go. GoFetch's default engine **drives
llama-swap** and adds the two things it lacks: the **warm-pool** and the **doc-KV
store**. A `direct` engine (GoFetch spawns `llama-server` itself) is also
provided for full control without the dependency. The choice is a single config
key (`engine.type`).

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

1. **M0 — skeleton (this commit):** packages, API surface, warm-pool via
   mmap/mlock, KV store accounting, llama-swap + direct engine stubs. Compiles.
2. **M1 — real swap:** `direct` engine starts/stops `llama-server`, proxies
   OpenAI traffic, enforces the single-VRAM-slot invariant.
3. **M2 — warm-pool LRU:** budget-driven eviction, status reporting, optional
   `mlock` with `RLIMIT_MEMLOCK` checks.
4. **M3 — doc-KV:** wire llama.cpp `--slot-save-path`; save/restore snapshots on
   tmpfs keyed by `(model, key)`; LRU evict by KV budget.
5. **M4 — llama-swap engine:** force-load / unload via llama-swap management API;
   reconcile its lifecycle with the warm-pool.

## Non-goals

- Routing / model-selection logic (lives in BigLaw).
- Retrieval / embedding / chunking (lives in BigLaw).
- Multi-GPU or multi-node scheduling.
