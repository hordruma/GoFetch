// Package models defines the static model registry. A Model is a GGUF on disk
// plus the parameters GoFetch needs to load it; the registry is an immutable
// lookup built once from config.
package models

// Model describes a single GGUF and how to load it.
type Model struct {
	ID   string `json:"id"`
	Path string `json:"path"`
	NGL  int    `json:"ngl"`  // GPU layers to offload (99 = all)
	Ctx  int    `json:"ctx"`  // context window
	Tier string `json:"tier"` // default warm tier: "cache" (default) | "locked"
}

// Registry is an immutable, concurrency-safe lookup of models by ID.
type Registry struct {
	byID  map[string]Model
	order []string
}

// NewRegistry builds a registry from a slice of models. Later duplicates of an
// ID overwrite earlier ones.
func NewRegistry(ms []Model) *Registry {
	r := &Registry{byID: make(map[string]Model, len(ms))}
	for _, m := range ms {
		if _, seen := r.byID[m.ID]; !seen {
			r.order = append(r.order, m.ID)
		}
		r.byID[m.ID] = m
	}
	return r
}

// Get returns the model with the given ID.
func (r *Registry) Get(id string) (Model, bool) {
	m, ok := r.byID[id]
	return m, ok
}

// All returns the registered models in registration order.
func (r *Registry) All() []Model {
	out := make([]Model, 0, len(r.order))
	for _, id := range r.order {
		out = append(out, r.byID[id])
	}
	return out
}
