# Streaming-layers escalation tier

A second answer to "8 GB VRAM is small." Hot-swap keeps one *whole* small model
resident and swaps models in/out. **Streaming** keeps a *big* model's weights in
RAM and treats VRAM as a sliding window over its layers — so you can run a model
**far bigger than VRAM**, at reduced speed. The two compose: the hot-swap pool
serves fast interactive work; a streaming tier handles occasional "hard query"
passes against a 30–70B model the 8 GB could never hold resident.

## Not novel — established prior art

This is a known, shipped technique, not an invention. The design borrows from:

- **FlexGen** — high-throughput single-GPU inference of large models by
  offloading weights/KV to CPU/disk and scheduling the streaming.
- **DeepSpeed ZeRO-Inference** — per-layer weight streaming from CPU/NVMe.
- **Hugging Face Accelerate** — `device_map` with CPU/disk offload.
- **AirLLM** — runs 70B on ~4 GB by loading layer-by-layer.

GoFetch's contribution is *integration*: exposing a streaming tier behind the
same control plane and orchestrator seam as hot-swap, tuned for this hardware.

## How it works

A transformer runs layers sequentially, so only the current layer's weights need
to be on the GPU:

1. All weights live in RAM (the 64 GB).
2. VRAM holds a small ring of layer buffers.
3. While the GPU computes layer N, prefetch layer N+1 RAM→VRAM on another stream.
4. Recycle layer N's buffer for N+2.

VRAM becomes a cache over RAM-resident weights; model size is bounded by RAM, not
VRAM.

## The honest tradeoff — read before relying on it

Cost is dominated by **decode vs prefill**:

- **Decode** (one token at a time): each token reads *every* weight once, so you
  stream the whole model per token. At ~13 GB/s pinned: 40 GB 70B-Q4 ≈ **3 s/tok**
  (~0.3 tok/s); 18 GB 30B-Q4 ≈ **1.4 s/tok**. Too slow for interactive chat.
- **Prefill / batch** (long prompt, or many sequences): many tokens flow through
  each layer per stream-in, so the weight read **amortizes** → compute-bound and
  efficient. Long-context ingestion is far better than generation.

**So streaming is for: long input → short answer, latency-tolerant, high-value
queries.** It is **not** for chatty multi-turn generation.

### Canonical use: a critique / verification agent

The ideal streaming workload, because it makes the weakness disappear. Give a big
model a large legal input (brief, contract, opinion) and constrain the output to
a handful of tokens — "what needs improvement," a pass/fail, a risk flag, a
short list of weak sections. Decode (the expensive part under streaming) is only
~10 tokens; prefill (efficient) dominates. You get a 70B's judgment on hardware
that can't hold a 70B resident.

Rough one-shot latency (confirm in S1): **70B-Q4 ~35–45 s**, **30B-Q4 ~15–20 s** —
fine for an async/background critique pass, and the model is far stronger than
anything that fits resident. Other fits of the same shape: PII/privilege sweep,
citation-missing flags, doc-type classification at high quality, "does this
argument hold" gut-checks.

## Where it fits

```
query → BigLaw router → cheap?  → hot-swap pool (fast, 7B resident)
                      → hard?   → streaming tier (30–70B from RAM, slow, prefill-heavy)
```

The router decides; GoFetch exposes the tier and its state. Escalation is a
policy call (confidence threshold, doc size, stakes) that lives in BigLaw.

## Engine question (open — needs a decision)

Streaming is **not** llama.cpp's wheelhouse (its `-ngl` offload is a static
CPU/GPU split, not GPU weight streaming). Options, to be evaluated:

1. **A PyTorch-side component** (FlexGen / ZeRO-Inference / Accelerate offload)
   run as a separate process GoFetch supervises and proxies to. Fastest to stand
   up; adds a Python/Torch dependency and a second runtime.
2. **An llama.cpp contribution** — add per-layer weight streaming to the CUDA
   backend. Keeps one stack and the GGUF ecosystem; significant C++ work.
3. **A bespoke streaming runner** reusing GoFetch's pinned warm tier + the
   in-process CUDA engine (M-deep) for the layer ring. Most control, most effort.

Recommendation: prototype with (1) to validate the latency/quality tradeoff on
real queries before committing to (2)/(3).

## Milestones

- **S0 — seam:** orchestrator-facing API marks a model as `streaming` tier;
  `/v1/status` reports it; router can target it. No engine yet.
- **S1 — prototype:** stand up option (1), measure prefill time and short-output
  (~10-token) critique latency for a 30B and a 70B on the 5060 against real
  long-input legal critique tasks (the canonical use), not chatty generation.
- **S2 — integrate:** pick the engine, wire it behind the control plane, reuse
  the pinned warm tier for the layer ring where applicable.

## Interaction with the rest of GoFetch

- **Warm-pool:** a streaming model's weights are the large RAM-resident set the
  layer ring streams from — the `cache` tier (or a dedicated streaming budget),
  not `locked` (you don't pin 40 GB).
- **KV cache:** the model/position-bound rule still holds; doc-KV snapshots are
  per streaming model just like any other.
- **DMA levers:** pinned host memory and chunked transfer (docs/dma-and-swap.md)
  apply directly to the per-layer prefetch path.
