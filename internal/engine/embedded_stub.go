//go:build !cuda

package engine

import (
	"fmt"

	"github.com/hordruma/gofetch/internal/models"
	"github.com/hordruma/gofetch/internal/warm"
)

// NewEmbedded is unavailable in the default (non-CUDA) build. The embedded engine
// links llama.cpp in-process to do the fast DMA path (cudaHostRegister on the
// locked warm tier) and persistent in-place model swap, which require the CUDA
// toolchain and a vendored llama.cpp. Rebuild with `-tags cuda`.
func NewEmbedded(_ *models.Registry, _ *warm.Pool) (Engine, error) {
	return nil, fmt.Errorf("embedded engine requires a CUDA build: rebuild with `go build -tags cuda` against a vendored llama.cpp (see DESIGN.md, M-deep)")
}
