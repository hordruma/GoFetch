# Streaming / offload engine survey — for the critique tier (S1)

Goal: pick an engine to prototype for the **streaming critique tier** — run a
30–70B model bigger than the 5060's 8 GB VRAM, **long input → ~10-token output**,
latency-tolerant, behind an OpenAI-compatible HTTP API a Go daemon can supervise.

> Provenance & confidence: compiled from web research (June 2026). The rigorous
> deep-research workflow (adversarial multi-vote verification) failed on a harness
> error, so this is a lighter single-pass synthesis — treat version numbers and
> Blackwell specifics as **verify-on-machine**, not gospel. Sources at the bottom.

## TL;DR recommendation

1. **Measure the no-new-engine baseline first: llama.cpp partial offload.** With
   `-ngl` set so part of the model is on the GPU and the rest on CPU, llama.cpp
   already runs models bigger than VRAM — same stack you use for hot-swap, zero
   new runtime. It's CPU-compute-bound on the offloaded layers (not true
   streaming), so it's slow, but for a ~10-token critique it may be *enough*.
   Establish this number before adding a dependency.
2. **Primary new engine — depends on the model:**
   - **Dense 30–70B → vLLM with `--cpu-offload-gb`.** Production OpenAI server
     (exactly the supervise-and-proxy shape GoFetch wants), real per-layer
     offload (UVA / async prefetch backends), good Blackwell support on recent
     builds.
   - **MoE model → KTransformers.** Purpose-built for CPU/GPU MoE offload, big
     **prefill** speedups (4.6–19.7× claimed) which is precisely the critique
     shape, OpenAI/Ollama-compatible server, supports dense too.
3. **Fallback / simplest: AirLLM** — "make a 70B run at all" with minimal setup,
   accept 5–30× slowdown.
4. **Skip for this use:** FlexGen (stale, batch-throughput oriented), DeepSpeed
   ZeRO-Inference (capable but heavy to operate as an API service), PowerInfer
   (needs activation-sparse models).

## RAM math for a 64 GB box (decides what's even feasible)

| Model            | Quant | ~Size  | Fits in 64 GB RAM? |
|------------------|-------|--------|--------------------|
| 30B dense        | Q4    | ~18 GB | yes, comfortably   |
| 70B dense        | Q4    | ~40 GB | yes                |
| 70B dense        | FP16  | ~140 GB| no → disk-bound    |
| 235B MoE (A22B)  | Q4    | ~120 GB| no                 |

Takeaways: **quantize** (FP16 70B won't fit — rules out AirLLM's no-quant mode
from RAM); dense ≤70B Q4 and small/mid MoEs fit; giant MoEs do not.

## Per-engine assessment

### llama.cpp partial offload — the baseline (no new dependency)
- **Mechanism:** static GPU/CPU layer split (`-ngl`), CPU *computes* the
  offloaded layers (not weight-streaming). Runs bigger-than-VRAM today.
- **Blackwell/5060:** works (sm_120 / CC 12.0) but **finicky to build** — use
  **CUDA 12.8** (CUDA 13.1 crashes / 5× slowdowns reported), pair with **g++-12**,
  and note a known **MXFP4 sm_120 compile failure** (avoid MXFP4 quants or patch).
- **API:** `llama-server` is OpenAI-compatible. Already in our stack.
- **Verdict:** measure this first; it may make a separate streaming engine
  unnecessary for the critique tier.

### vLLM — `--cpu-offload-gb` (primary for dense)
- **Mechanism:** treats CPU RAM as virtual VRAM; layers loaded CPU→GPU on the fly
  each forward pass via **UVA zero-copy**, with `auto` / `uva` / `prefetch`
  backends (prefetch = async group-based layer offload → overlap).
- **API:** vLLM *is* the de-facto OpenAI-compatible server — clean separate
  process to supervise/proxy.
- **Blackwell:** supported on recent vLLM + PyTorch cu128 wheels (verify). Known
  bug: `--cpu-offload-gb` asserts on **Windows** (#36279) — Linux only, which
  matches the plan.
- **Caveat:** heavy runtime; offload decode is slow, but fine for ~10 tokens.

### KTransformers — primary if the critique model is MoE
- **Status:** very active (v0.6.1 Apr 2026; day-0 support for new MoEs through
  2026). Runs 100B+ **MoE** on a single 32 GB GPU by offloading inactive experts
  to CPU; AMX-optimized CPU kernels; **4.6–19.7× prefill**, 1.25–4.09× decode
  speedups claimed.
- **Models:** dense **and** MoE; OpenAI/Ollama-compatible API server.
- **Caveats:** the headline wins are **MoE-specific** (cheap to offload experts
  that don't activate); on dense models the edge shrinks. AMX kernels favor Intel
  server CPUs — **verify AMX is present on your Core Ultra**; without it you fall
  back to slower kernels. On 8 GB (vs the 32 GB in their demos) expect tighter
  limits.

### AirLLM — simplest fallback
- **Status:** active (~21k stars, v2.8). Layer-by-layer load; 70B on 4 GB, 405B
  on 8 GB. v2.0 added ~3× via compression; macOS in v2.8.
- **Caveats:** **5–30× slower** (disk-speed dependent); default FP16 is huge
  (70B ≈ 140 GB → exceeds 64 GB RAM, hits disk). Python library, not an OpenAI
  server out of the box — needs wrapping. Good for "possible," not "fast."

### DeepSpeed ZeRO-Inference — capable, heavy to operate
- ZeRO-3 / ZeRO-Infinity offloads model states to **CPU and NVMe**; maintained
  (0.18.x/0.19.x). But it's a training-derived framework; standing it up as a
  simple critique API service is real glue work. Over-built for this.

### FlexGen — skip
- 2023 research, optimized for **throughput at large batch** (≈1 tok/s OPT-175B on
  16 GB *with batch 144*), not single-request low latency. No sign of active
  maintenance. Wrong shape for a one-shot critique.

### PowerInfer / PowerInfer-2 — niche
- Exploits **activation sparsity** (hot/cold neurons): hot on GPU, cold on CPU.
  Strong on ReLU-sparse / sparsified models; PowerInfer-2 targets smartphones.
  Most mainstream dense models aren't sparse enough to benefit without a
  sparsified variant. Not a general critique engine here.

## Cross-cutting Blackwell + Linux gotchas

- **CUDA 12.8**, not 13.1, for anything compiled for sm_120 (llama.cpp reports
  crashes/slowdowns on 13.1). Pair with g++-12.
- **PyTorch engines** (vLLM, KTransformers, DeepSpeed, AirLLM) need **cu128**
  wheels / a Blackwell-capable PyTorch — confirm the wheel targets sm_120.
- **MXFP4 quant** has a known sm_120 build failure in llama.cpp — prefer
  Q4_K/Q5_K/Q6_K for now.
- Several offload paths are **Linux-only / Windows-broken** (vLLM `--cpu-offload-gb`)
  — fine, S1 runs on Linux.

## S1 prototype plan

1. **Baseline:** `llama.cpp` `-ngl` partial offload of a 70B-Q4 and 30B-Q4; time
   prefill on a long brief + ~10-token critique output.
2. **Primary:** stand up **vLLM `--cpu-offload-gb`** (dense) and/or
   **KTransformers** (if the chosen critique model is MoE) behind their OpenAI
   servers; same measurement.
3. Compare total wall-clock for a realistic critique call; pick the engine GoFetch
   supervises for the streaming tier. Reuse the pinned/chunked DMA findings
   (docs/dma-and-swap.md) for the per-layer transfer where applicable.

## Sources

- KTransformers: https://github.com/kvcache-ai/ktransformers ·
  https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/api/server/server.md ·
  SOSP'25 paper https://dl.acm.org/doi/10.1145/3731569.3764843
- vLLM CPU offload: https://docs.vllm.ai/en/stable/api/vllm/config/offload/ ·
  https://docs.vllm.ai/en/v0.7.1/getting_started/examples/cpu_offload.html ·
  Windows bug https://github.com/vllm-project/vllm/issues/36279
- AirLLM: https://github.com/lyogavin/airllm
- FlexGen: https://arxiv.org/abs/2303.06865 · https://github.com/FMInference/FlexGen
- DeepSpeed ZeRO: https://deepspeed.readthedocs.io/en/latest/zero3.html ·
  https://www.deepspeed.ai/tutorials/zero-offload/
- PowerInfer: https://arxiv.org/abs/2312.12456 · https://github.com/SJTU-IPADS/PowerInfer
- llama.cpp Blackwell: https://github.com/ggml-org/llama.cpp/issues/19662 ·
  https://zenn.dev/toki_mwc/articles/rtx5090-blackwell-cuda-toolkit-trap-llama-cpp
