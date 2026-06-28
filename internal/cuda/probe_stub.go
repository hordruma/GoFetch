//go:build !cuda

package cuda

import "fmt"

// Probe is unavailable in the default build. Rebuild with `-tags cuda` on a
// machine with the CUDA toolkit installed (see docs/dma-and-swap.md).
func Probe(path string, iters int, maxBytes int64) (*Result, error) {
	return nil, fmt.Errorf("cuda probe requires a CUDA build: `make probe` / `go build -tags cuda` (CUDA toolkit required)")
}
