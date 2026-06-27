// Package engine abstracts the inference backend. An Engine owns the single
// VRAM-resident model and serves OpenAI-compatible traffic for it. Two
// implementations are provided: LlamaSwap (drive an external llama-swap proxy)
// and Direct (GoFetch spawns llama-server itself).
package engine

import (
	"context"
	"net/http"
)

// Engine is the contract GoFetch's control plane drives.
type Engine interface {
	// EnsureResident makes modelID the VRAM-resident model, swapping out the
	// current one if necessary. It blocks until the model is ready to serve.
	EnsureResident(ctx context.Context, modelID string) error

	// Drop unloads modelID from VRAM if it is currently resident.
	Drop(ctx context.Context, modelID string) error

	// Active returns the currently resident model ID, or "" if none.
	Active() string

	// ServeHTTP proxies an OpenAI-compatible request to the active model.
	ServeHTTP(w http.ResponseWriter, r *http.Request)
}
