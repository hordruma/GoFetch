# GoFetch

Mechanical memory plane for a local LLM stack: load, warm, drop, and swap models
across a small VRAM budget and a large RAM pool, plus a tmpfs-backed document
**KV cache store**. Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) /
[llama-swap](https://github.com/mostlygeek/llama-swap).

GoFetch executes mechanics only — *what* and *when* to swap are decided by an
external orchestrator (the [BigLaw](https://github.com/discover-legal/BigLaw)
agentic RAG system). See [DESIGN.md](./DESIGN.md) for the architecture and the
constraints that shape it.

## Quick start

Needs Go 1.24+ and a `llama-server` binary from
[llama.cpp](https://github.com/ggml-org/llama.cpp/releases) (any backend: CUDA,
Vulkan, ROCm, CPU).

```sh
cp config.example.json config.json   # edit model paths + llama_server_bin
go build ./...
./gofetch -config config.json
```

Then drive it with any OpenAI client — GoFetch loads the model named in the
request if it is not already resident:

```sh
curl localhost:8080/v1/chat/completions \
  -d '{"model":"qwen-1.5b","messages":[{"role":"user","content":"hello"}]}'
```

## Control API

```
GET  /v1/status
POST /v1/vram/load  {model}      POST /v1/vram/drop  {model}
POST /v1/ram/warm   {model,tier} POST /v1/ram/evict  {model}   # tier: cache|locked
POST /v1/kv/save    {model,key}  POST /v1/kv/load    {model,key}  POST /v1/kv/evict {model,key}
POST /v1/chat/completions ...    # OpenAI passthrough; auto-loads the named model
```

`/v1/kv/save` snapshots the resident model's prompt cache to tmpfs and
`/v1/kv/load` restores it, so a repeat query over the same document prefix skips
prefill. Snapshots are bound to the model that produced them: saving or loading
one while a different model is resident is refused with `409`.

## Engines

| `engine.type` | What it does |
|---|---|
| `direct` | GoFetch spawns and supervises `llama-server` itself. One process at a time. |
| `llama-swap` | GoFetch drives an external [llama-swap](https://github.com/mostlygeek/llama-swap): force-load, unload, and read back what is actually running. Its config must set `--slot-save-path` to GoFetch's `kv.dir` for KV snapshots. |
| `embedded` | In-process llama.cpp (cgo): one persistent GPU context, no-restart in-place swap, and KV entirely in-process. Build-tagged (`-tags embedded`, needs libllama) — `make build-embedded`. Measured ~1.4–1.9× faster per swap than `direct`; see [docs/verification.md](./docs/verification.md). |

## Status

M0–M4 are implemented and verified on real hardware (RTX 5060 + llama.cpp
b10075): model swap, the two-tier warm-pool, document KV snapshots, and both
external engines. Measured numbers and the exact rig are in
[docs/verification.md](./docs/verification.md) — including a 2 925-token
document whose prefill drops from 490 ms to 24 ms when its KV snapshot is
restored.

The in-process `embedded` engine is implemented and measured up to a 14B that
fills 91% of the 8 GB VRAM: real in-process inference, SSE streaming, no-restart
swap (~0.6 s/swap faster than `direct`), prompt-prefix reuse, and 31 ms KV
restore.

**Not verified:** the CUDA-specific weight-upload path — registering the warm
tier and DMA'ing into llama.cpp's own loader — which needs a from-source CUDA
build of llama.cpp. The pinned-DMA transport is measured standalone (`-tags
cuda` probe); wiring it into the loader is the remaining specialization. Current
GPU inference runs on a Vulkan libllama.
