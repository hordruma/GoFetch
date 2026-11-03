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

	// SaveKV snapshots the resident model's prompt-cache slot to name, a bare
	// filename beneath the engine's slot-save directory. The snapshot is only
	// meaningful for the model that produced it; the caller enforces that.
	SaveKV(ctx context.Context, name string) (KVStats, error)

	// LoadKV restores a snapshot previously written by SaveKV into the slot.
	LoadKV(ctx context.Context, name string) (KVStats, error)

	// ServeHTTP proxies an OpenAI-compatible request to the active model.
	ServeHTTP(w http.ResponseWriter, r *http.Request)
}

// KVStats reports what a snapshot operation moved. Tokens is the prompt-cache
// depth restored or saved; Bytes is the snapshot size on the tmpfs.
type KVStats struct {
	Tokens  int     `json:"tokens"`
	Bytes   int64   `json:"bytes"`
	Millis  float64 `json:"ms"`
	Restore bool    `json:"-"`
}
