# Backends and accelerators

GoFetch should be **vendor-neutral**: NVIDIA and AMD ("ATI") now, Intel/Apple
later. Nothing in the daemon should hard-bind to CUDA. This reshapes the engine
plan, so it is written down here.

## The lever: llama.cpp already abstracts the GPU backend

Our chosen foundation already ships every backend we care about. The supervisor
engines (`llama-swap`, `direct`) drive llama.cpp, so they **inherit multi-vendor
support for free** — it becomes a build/deploy choice, not new code:

| Backend     | Vendor             | GoFetch use                          |
|-------------|--------------------|--------------------------------------|
| CUDA        | NVIDIA             | primary (the 5060)                   |
| HIP / ROCm  | AMD ("ATI")        | the AMD target                       |
| Vulkan      | NVIDIA/AMD/Intel   | universal fallback (widest, slower)  |
| SYCL        | Intel              | Intel GPUs                           |
| Metal       | Apple              | later                                |
| CPU         | all                | always available                     |

**Consequence:** prefer building *on* llama.cpp over a bespoke CUDA engine. A
hand-rolled CUDA engine would lock us to NVIDIA — directly against the goal.

## Implication for the in-process (M-deep) engine

The fast-DMA / persistent-context engine must be **backend-abstracted**, not
CUDA-only. Two routes, preferred first:

1. **Reuse ggml's per-backend pinned host buffers.** ggml/llama.cpp already has
   pinned host buffer types per backend (CUDA and HIP both expose page-locked
   host allocations). Driving those gives the pinned fast-DMA win on NVIDIA *and*
   AMD without vendor-specific GoFetch code. Strongly preferred.
2. **Abstract a `Backend` interface** (`Pin`, `Upload`, `Alloc`, `Free`) with
   CUDA + HIP + Vulkan implementations only if (1) is insufficient. HIP is a
   near-mechanical port of CUDA (`hipify`); Vulkan is the portable floor.

The pinned-vs-pageable *concept* is universal; only the API differs
(`cudaHostRegister` ↔ `hipHostRegister` ↔ Vulkan staging + `HOST_VISIBLE`
memory). The probe stays CUDA-first because the test GPU is the 5060; a HIP probe
is a `hipify` of it, a Vulkan probe comes later.

## Portability vs peak performance

Vulkan runs everywhere but is typically slower than the vendor-native backend.
So: **target the native backend per vendor** (CUDA on NVIDIA, HIP/ROCm on AMD,
SYCL on Intel) behind GoFetch's single `Engine` interface, with Vulkan as the
"runs anywhere" fallback. The daemon stays vendor-agnostic; the backend is chosen
at build/deploy time.

## Auxiliary accelerator: the NPU

The laptop's NPU (Intel "AI Boost" or AMD "XDNA" / Ryzen AI) is **not** useful
for big-LLM decode — too little memory bandwidth to system RAM, immature
software. But it is a great home for the **always-on small models** in the
pipeline, which frees the GPU + VRAM entirely for generation/critique:

| Candidate job        | Fit on NPU | Notes                                             |
|----------------------|-----------|---------------------------------------------------|
| RAG **embeddings**   | excellent | encoder-only, batchy, INT8 — the NPU's sweet spot |
| **Reranker**         | good      | cross-encoder; heavier but viable                 |
| **Router / classifier** (difficulty, intent, doc-type) | excellent | tiny, always-on; keeps the GPU free for real work |
| **Guards** (PII, privilege, safety) | good | small, always-on filters                          |
| Draft model for speculative decode | poor (for now) | sync overhead + thin software usually cancels the win |

**Confirmed hardware: Intel Core Ultra → "AI Boost" NPU.** Software stack is
**OpenVINO** (device `"NPU"`), or ONNX Runtime with the OpenVINO execution
provider. Models run as OpenVINO IR (convert from ONNX/HF) at INT8.

**Linux bring-up (for next week):** needs the `intel_vpu` kernel driver (in
kernel since ~6.7 for Meteor Lake; newer Core Ultra wants a recent kernel) plus
the user-space NPU stack — the Level-Zero NPU driver and firmware
(`intel-level-zero-npu`, `intel-driver-compiler-npu`, `intel-fw-npu`) — then
OpenVINO picks up the `NPU` device. Verify with `ls /dev/accel/` and OpenVINO's
`hello_query_device`. Practical constraints: the NPU plugin prefers **static
shapes** and INT8 and has limited op coverage, so fixed-sequence-length
embeddings/classifiers fit best; dynamic-shape generative models do not.

**Design fit:** model the NPU as a separate "auxiliary models" lane in GoFetch —
fixed-shape embeddings / router-classifier / guards routed there (good fits for
its constraints), GPU reserved for the hot-swap pool and the streaming critique
tier. Additive; comes after the GPU path works.
