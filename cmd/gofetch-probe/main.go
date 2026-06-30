// Command gofetch-probe measures the host->device (RAM->VRAM) transport for a
// GGUF on the local GPU, so swap-latency decisions rest on real hardware numbers
// instead of estimates. It is a TRANSPORT microbenchmark, not a swap-time
// measurement (a real swap also rebuilds the context/graph + KV). Requires a
// CUDA build:
//
//	make probe                       # builds ./gofetch-probe (-tags cuda)
//	./gofetch-probe -model M.gguf    # measure
//
// See docs/dma-and-swap.md for how to read the numbers.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/hordruma/gofetch/internal/cuda"
)

func main() {
	model := flag.String("model", "", "path to a GGUF to upload (required)")
	iters := flag.Int("iters", 7, "transfers per variant (mean + peak reported)")
	maxGB := flag.Float64("max-gb", 0, "cap transfer size in GB (0 = whole file)")
	chunkMB := flag.Int("chunk-mb", 8, "chunk size for the representative chunked variant")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "usage: gofetch-probe -model <path.gguf> [-iters N] [-max-gb G] [-chunk-mb M]")
		os.Exit(2)
	}

	var maxBytes int64
	if *maxGB > 0 {
		maxBytes = int64(*maxGB * 1e9)
	}
	chunkBytes := int64(*chunkMB) * 1024 * 1024

	r, err := cuda.Probe(*model, *iters, maxBytes, chunkBytes)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}

	gb := float64(r.Bytes) / 1e9
	fmt.Printf("transfer size : %.2f GB   (x%d iters, chunk %d MB)\n\n", gb, *iters, *chunkMB)

	fmt.Println("H2D bandwidth (mean / peak GB/s):")
	fmt.Printf("  pageable          : %6.2f / %6.2f    (today, no surgery)\n", r.PageableGBps, r.PageablePeakGBps)
	if r.RegOK {
		fmt.Printf("  pinned (register) : %6.2f / %6.2f    (register %.0f ms once at warm; no copy)\n",
			r.PinnedRegGBps, r.PinnedRegPeakGBps, r.RegisterMS)
	} else {
		fmt.Printf("  pinned (register) : unavailable      (driver refused cudaHostRegister on mmap)\n")
	}
	fmt.Printf("  pinned (alloc)    : %6.2f / %6.2f    (+%.0f ms alloc +%.0f ms RAM-copy, every load)\n",
		r.PinnedAllocGBps, r.PinnedAllocPeakGBps, r.HostAllocMS, r.FillMS)
	fmt.Printf("  pinned (chunked)  : %6.2f             (representative: many tensor-sized copies)\n",
		r.PinnedChunkedGBps)

	fmt.Println("\nVRAM allocator (per swap if not pooled):")
	fmt.Printf("  cudaMalloc        : %6.1f ms\n", r.MallocMS)
	fmt.Printf("  cudaFree          : %6.1f ms\n", r.FreeMS)

	// Use the chunked pinned number as the realistic upload rate.
	if r.PageableGBps > 0 && r.PinnedChunkedGBps > 0 {
		fmt.Printf("\nupload of %.2f GB (realistic, chunked): %.0f ms pageable  ->  %.0f ms pinned   (%.1fx)\n",
			gb, gb/r.PageableGBps*1e3, gb/r.PinnedChunkedGBps*1e3, r.PinnedChunkedGBps/r.PageableGBps)
		fmt.Println("note: monolithic numbers above are an optimistic ceiling; the loader's many")
		fmt.Println("small tensor copies land nearer the chunked figure.")
	}
}
