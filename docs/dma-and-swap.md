# RAM→VRAM, DMA, and every swap-latency lever

This is the reference for *why* a model swap costs what it costs, and the full
list of places there is headroom. Run the probe (below) on the target GPU to
replace the estimates here with real numbers before building on them.

## The full path: GGUF on disk → tokens out

A swap walks weights through four phases. Only B is "DMA"; the others matter too.

```
[A] disk → host RAM      NVMe read into the page cache
[B] host RAM → VRAM      the PCIe DMA transfer            ← the "DMA stuff"
[C] on-device setup      buffers, graph, KV allocation
[D] inference            per-token compute (VRAM-bandwidth bound, no PCIe)
```

Key framing: **once a model is resident, PCIe is irrelevant** — decode (D) is
bound by VRAM bandwidth. So every DMA optimization only buys *swap latency*. Its
value scales with how often you swap; a model that stays resident and serves
many tokens amortizes its upload to ~zero.

## Phase B in detail — what DMA actually is

DMA = the GPU's **copy engine** reads host physical memory across PCIe and
writes VRAM, without the CPU copying byte-by-byte. The CPU only programs a
descriptor (source physical pages, destination, length); the copy engine does
the move. For that to be safe, the source's **physical address must be stable**
for the whole transfer. That single requirement is the entire pinned-vs-pageable
story.

### Pageable memory (what we have today)

Ordinary `malloc`/`mmap` memory is **pageable**: the OS may move or evict it, so
its physical address can change. The copy engine can't use an address that might
move. So `cudaMemcpy` from pageable memory does this *internally*:

1. allocate a small **pinned staging buffer** (a few MB, fixed address);
2. **CPU `memcpy`** a chunk of your data into it;
3. DMA the staging buffer → VRAM;
4. repeat in chunks.

Costs: an extra **CPU memcpy of every byte**, no overlap (the driver is babysitting
the staged copy), and you miss peak PCIe bandwidth. Real pageable H2D often lands
around **half** of pinned.

### Pinned (page-locked) memory — the win

`cudaMallocHost`/`cudaHostAlloc` allocate page-locked host memory; `cudaHostRegister`
page-locks an *existing* region (e.g. our `mlock`'d mmap of the GGUF). The OS
guarantees it won't move, so the copy engine DMAs **directly** from it — no
staging, no CPU memcpy, full PCIe bandwidth, and the copy can be **async**
(`cudaMemcpyAsync`) and overlap other work. Two flavors:

- **register-in-place** (`cudaHostRegister` on the mmap region): no extra copy;
  pairs exactly with the warm-pool `locked` tier. Cost: registration pins every
  page and builds mappings — a one-time cost paid at *warm* time, not per swap.
  Caveat: some drivers refuse to register file-backed read-only mappings — the
  probe tells us if ours does.
- **alloc-and-copy** (`cudaMallocHost` + read weights in): always works, hits the
  ceiling, but pays one RAM→RAM copy to fill the pinned buffer.

The probe measures both so we pick the right one on this machine.

### The hard ceiling: PCIe

No software beats the link. Verify the actual link with
`nvidia-smi -q | grep -A3 "Link"` or `sudo lspci -vv | grep -i lnksta`:

| Link            | Theoretical | ~Real pinned |
|-----------------|-------------|--------------|
| PCIe 4.0 x8     | 16 GB/s     | ~12–13 GB/s  |
| PCIe 4.0 x16    | 32 GB/s     | ~24 GB/s     |
| PCIe 5.0 x8     | 32 GB/s     | ~24 GB/s     |

Also confirm **Resizable BAR** is on in BIOS, and that the dGPU isn't training
down to fewer lanes on battery / via a MUX. A 5 GB model at 13 GB/s ≈ **0.4 s**;
at pageable ~7 GB/s ≈ **0.7 s**.

## Every lever, ranked

| # | Lever | Phase | Where | Status |
|---|-------|-------|-------|--------|
| 1 | **Don't swap** — keep small models (reranker/embedder) co-resident with the 7B; only swap big generators | — | BigLaw policy + GoFetch co-residency | design |
| 2 | **Skip the disk read** — page-cache warm tier | A | `warm` cache tier | ✅ built |
| 3 | **No process restart** — persistent CUDA context + in-place `llama_model_free`/`load`. CUDA context init alone is 0.3–1 s; this can beat the DMA win by itself | B/C | `embedded` engine | scaffolded |
| 4 | **Pinned DMA** — `cudaHostRegister` the `locked` tier (or alloc-and-copy) | B | `embedded` + `locked` tier | scaffolded |
| 5 | **Reuse VRAM** — pooled device allocator instead of `cudaFree`+`cudaMalloc` per swap (both are synchronous, can cost ms) | C | `embedded` | planned |
| 6 | **Overlap upload** — multi-stream `cudaMemcpyAsync` so transfer hides behind buffer/graph setup | B/C | `embedded` | planned |
| 7 | **Smaller weights** — quant choice (Q4 vs Q5/Q6, IQ quants). Fewer bytes uploaded *and* more fits resident | B/D | model selection | knob |
| 8 | **PCIe maximization** — x16 vs x8, gen4/5, ReBAR on, no lane downgrade | B | BIOS/hardware | verify |
| 9 | **KV quantization + flash-attention** — shrink KV cache VRAM, freeing room for weights / longer context | C/D | engine flags | knob |
| 10 | **Pre-warm prediction** — orchestrator pre-pins the next likely model (and pays registration) ahead of need | A/B | BigLaw, via `/v1/ram/warm` | primitive exists |

### Explicitly NOT worth it here

- **Unified memory / `cudaMallocManaged` / zero-copy mapped weights**: weights are
  touched every token, so on-demand page migration faults constantly mid-decode —
  bigger capacity, much slower decode. Wrong tool for weights.
- **Streaming-weights inference** (compute layer 0 while later layers upload):
  llama.cpp has no such mode; research-grade, out of scope.
- **GPUDirect Storage** (NVMe→VRAM, bypass RAM): helps only *cold* loads, and
  laptop support is effectively absent. Our win is RAM-warm, so N/A.

## The probe

Measures levers 4, 5, and 8 directly on the GPU. Build needs only the **CUDA
toolkit** (not llama.cpp):

```sh
make probe                                   # CGO_ENABLED=1 go build -tags cuda
./gofetch-probe -model /models/your-7b.gguf  # -iters N, -max-gb G to cap size
```

If CUDA isn't at `/usr/local/cuda`, point cgo at it:

```sh
export CGO_CFLAGS="-I/path/to/cuda/include"
export CGO_LDFLAGS="-L/path/to/cuda/lib64 -lcudart"
make probe
```

Sample output:

```
transfer size : 4.80 GB   (x7 iters, chunk 8 MB)

H2D bandwidth (mean / peak GB/s):
  pageable          :   6.90 /   7.10    (today, no surgery)
  pinned (register) :  12.80 /  13.00    (register 180 ms once at warm; no copy)
  pinned (alloc)    :  13.10 /  13.20    (+30 ms alloc +380 ms RAM-copy, every load)
  pinned (chunked)  :  11.90             (representative: many tensor-sized copies)

VRAM allocator (per swap if not pooled):
  cudaMalloc        :   40.0 ms
  cudaFree          :   12.0 ms

upload of 4.80 GB (realistic, chunked): 696 ms pageable -> 403 ms pinned (1.7x)
```

How to read it:
- **Use the chunked number as reality.** The monolithic mean/peak are an
  optimistic ceiling; the loader's many small tensor copies land near `chunked`.
- **Peak vs mean** — if mean ≪ peak, the GPU is throttling or the PCIe link is
  power-managing; plug in, set the power profile to performance, and re-run.
- **register vs alloc** — if `register` succeeded and ≈ `alloc`, prefer
  register-in-place: `alloc` pays the `RAM-copy` (`FillMS`) on *every* load.
- **malloc/free** — if these are several ms, lever 5 (a pooled VRAM allocator
  that avoids per-swap free/alloc) is worth building.

### Probe caveats (what it does NOT tell you)

- It is a **transport** benchmark, not a swap. A real swap also rebuilds the
  context/graph and allocates KV — measure that in the engine, not here.
- It does **not** measure transfer/compute **overlap** (lever 6); that only shows
  up against real decode work, so it belongs in the engine benchmark.
- **OS matters.** The `#cgo` paths assume native Linux + CUDA at
  `/usr/local/cuda`. Under **WSL2**, pinned-memory behavior and `cudaHostRegister`
  on mmap differ and can mislead; on **Windows** the build needs MSVC and
  different lib paths. Confirm the environment before trusting the numbers.
