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

```sh
cp config.example.json config.json   # edit model paths
go build ./...
./gofetch -config config.json
```

## Control API

```
GET  /v1/status
POST /v1/vram/load  {model}      POST /v1/vram/drop  {model}
POST /v1/ram/warm   {model,tier} POST /v1/ram/evict  {model}   # tier: cache|locked
POST /v1/kv/save    {model,key}  POST /v1/kv/load    {model,key}  POST /v1/kv/evict {model,key}
POST /v1/chat/completions ...    # OpenAI-compatible passthrough
```

Status: **M0 skeleton** — compiles, unit-tested. Two-tier warm-pool
(`cache`/`locked`) in place; the in-process CUDA engine for the fast-swap path is
a build-tagged seam (`-tags cuda`, see DESIGN.md "M-deep"). Engine swap and KV
wiring are the next milestones.
