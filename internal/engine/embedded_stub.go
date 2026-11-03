//go:build !embedded

package engine

import (
	"fmt"

	"github.com/hordruma/gofetch/internal/models"
	"github.com/hordruma/gofetch/internal/warm"
)

// NewEmbedded is unavailable in the default build. The embedded engine links
// llama.cpp in-process (cgo) for a persistent context and no-restart in-place
// model swap; it needs the llama.cpp headers and a libllama to link against, so
// it lives behind a build tag to keep the default build pure-Go and
// dependency-free.
//
//	go build -tags embedded \
//	  CGO_CFLAGS=-I<llama headers> \
//	  CGO_LDFLAGS='-L<libllama dir> -lllama -lggml -lggml-base -Wl,-rpath,<libllama dir>'
//
// See internal/engine/embedded_llama.go and docs/verification.md.
func NewEmbedded(_ *models.Registry, _ *warm.Pool, _, _ string) (Engine, error) {
	return nil, fmt.Errorf("embedded engine requires an in-process build: rebuild with `go build -tags embedded` against a libllama (see docs/verification.md)")
}
