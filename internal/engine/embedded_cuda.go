//go:build cuda

// This file is the in-process ("embedded") engine: the home of the deeper
// surgery. It is a development skeleton — pure Go so a `-tags cuda` build
// compiles — with the cgo/CUDA/llama.cpp integration points marked as TODO. It
// is NOT yet functional; wiring it requires a vendored llama.cpp + CUDA toolkit
// and must be benchmarked on real hardware (see DESIGN.md, "M-deep").
//
// Why this engine exists: an external process supervisor (the llama-swap and
// direct engines) cannot reach the three wins below, because they live inside
// the inference process and its CUDA context:
//
//  1. Fast DMA. cudaHostRegister() the locked warm-tier region, then upload to
//     VRAM at the PCIe ceiling (~12-14 GB/s on x8 Gen4) instead of the driver's
//     pageable->staging->VRAM path (~half that).
//  2. No-restart swap. Keep one process + one CUDA context alive across swaps;
//     llama_model_free() the outgoing model and llama_model_load_from_file() the
//     incoming one, skipping process spawn, context init, and GGUF re-parse.
//  3. Overlapped upload. Stage weights over multiple CUDA streams so transfer
//     overlaps graph/buffer setup. (Pipelining upload with the first token's
//     compute is research-grade and intentionally out of scope; see DESIGN.md.)
package engine

import (
	"context"
	"fmt"
	"net/http"
	"sync"

	"github.com/hordruma/gofetch/internal/models"
	"github.com/hordruma/gofetch/internal/warm"
)

// Embedded owns a single persistent CUDA context and at most one resident model.
type Embedded struct {
	reg  *models.Registry
	pool *warm.Pool

	mu     sync.Mutex
	active string
	// model *C.llama_model  // TODO(M-deep): handle to the loaded llama.cpp model
	// ctx   *C.llama_context
}

// NewEmbedded constructs the in-process engine and initializes the CUDA backend.
func NewEmbedded(reg *models.Registry, pool *warm.Pool) (Engine, error) {
	// TODO(M-deep): llama_backend_init(); create a persistent CUDA context here
	// so it is reused across every swap.
	return &Embedded{reg: reg, pool: pool}, nil
}

// EnsureResident swaps the resident model in place, without a process restart.
func (e *Embedded) EnsureResident(ctx context.Context, modelID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.active == modelID {
		return nil
	}
	m, ok := e.reg.Get(modelID)
	if !ok {
		return fmt.Errorf("unknown model %q", modelID)
	}

	// Ensure the weights are page-locked so the upload can DMA from pinned host
	// memory. The locked warm tier is the buffer cudaHostRegister will pin.
	if err := e.pool.Warm(m.Path, warm.TierLocked); err != nil {
		return fmt.Errorf("pin weights: %w", err)
	}

	// TODO(M-deep):
	//   1. if e.model != nil { llama_free(e.ctx); llama_model_free(e.model) }
	//   2. cudaHostRegister(host_ptr, size, cudaHostRegisterReadOnly) on the
	//      locked mmap region for this GGUF (expose its base ptr from warm.Pool).
	//   3. llama_model_load_from_file(m.Path, params{ n_gpu_layers: m.NGL }),
	//      with the loader uploading from the pinned region over N streams.
	//   4. llama_init_from_model(...) to build the context (KV sized for m.Ctx).
	_ = m
	e.active = modelID
	return fmt.Errorf("embedded engine not yet implemented (skeleton)")
}

// Drop frees the resident model's VRAM but keeps the CUDA context alive.
func (e *Embedded) Drop(ctx context.Context, modelID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.active != modelID {
		return nil
	}
	// TODO(M-deep): cudaHostUnregister; llama_free; llama_model_free.
	e.active = ""
	return nil
}

func (e *Embedded) Active() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.active
}

// SaveKV is not available in-process yet: the embedded engine owns the KV cache
// directly, so snapshots will go through llama_state_seq_save_file rather than
// the server's slot endpoint.
func (e *Embedded) SaveKV(context.Context, string) (KVStats, error) {
	// TODO(M-deep): llama_state_seq_save_file(e.ctx, path, seq_id, ...).
	return KVStats{}, fmt.Errorf("embedded engine KV snapshots not yet implemented")
}

// LoadKV is the restore counterpart to SaveKV.
func (e *Embedded) LoadKV(context.Context, string) (KVStats, error) {
	// TODO(M-deep): llama_state_seq_load_file(e.ctx, path, seq_id, ...).
	return KVStats{}, fmt.Errorf("embedded engine KV snapshots not yet implemented")
}

// ServeHTTP runs an OpenAI-compatible request against the resident model.
func (e *Embedded) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// TODO(M-deep): translate the OpenAI request, run llama_decode in-process,
	// and stream tokens back. Until implemented, signal not-ready.
	http.Error(w, "embedded engine inference not yet implemented", http.StatusNotImplemented)
}
