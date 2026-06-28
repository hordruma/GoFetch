// Command gofetch-probe measures the host->device (RAM->VRAM) transfer path for
// a GGUF on the local GPU, so the swap-latency optimizations are validated
// against real hardware numbers instead of estimates. Requires a CUDA build:
//
//	make probe                       # builds ./gofetch-probe (-tags cuda)
//	./gofetch-probe -model M.gguf    # measure
//
// See docs/dma-and-swap.md for what each number means and how to read it.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/hordruma/gofetch/internal/cuda"
)

func main() {
	model := flag.String("model", "", "path to a GGUF to upload (required)")
	iters := flag.Int("iters", 5, "transfers to average per variant")
	maxGB := flag.Float64("max-gb", 0, "cap transfer size in GB (0 = whole file)")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "usage: gofetch-probe -model <path.gguf> [-iters N] [-max-gb G]")
		os.Exit(2)
	}

	var maxBytes int64
	if *maxGB > 0 {
		maxBytes = int64(*maxGB * 1e9)
	}

	r, err := cuda.Probe(*model, *iters, maxBytes)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}

	gb := float64(r.Bytes) / 1e9
	fmt.Printf("transfer size      : %.2f GB  (x%d iters)\n", gb, *iters)
	fmt.Println("H2D bandwidth:")
	fmt.Printf("  pageable         : %6.2f GB/s   (today, no surgery)\n", r.PageableGBps)
	if r.RegOK {
		fmt.Printf("  pinned (register): %6.2f GB/s   (register cost %.1f ms, one-time at warm)\n", r.PinnedRegGBps, r.RegisterMS)
	} else {
		fmt.Printf("  pinned (register): unavailable  (driver refused cudaHostRegister on mmap)\n")
	}
	fmt.Printf("  pinned (alloc)   : %6.2f GB/s   (ceiling; +%.1f ms alloc + a RAM copy)\n", r.PinnedAllocGBps, r.HostAllocMS)
	fmt.Println("VRAM allocator:")
	fmt.Printf("  cudaMalloc       : %6.1f ms\n", r.MallocMS)
	fmt.Printf("  cudaFree         : %6.1f ms\n", r.FreeMS)

	if r.PageableGBps > 0 {
		best := r.PinnedAllocGBps
		if r.RegOK && r.PinnedRegGBps > best {
			best = r.PinnedRegGBps
		}
		fmt.Printf("\nupload of %.2f GB: %.0f ms pageable  ->  %.0f ms pinned   (%.1fx)\n",
			gb, gb/r.PageableGBps*1e3, gb/best*1e3, best/r.PageableGBps)
	}
}
