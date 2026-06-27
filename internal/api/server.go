// Package api exposes GoFetch's HTTP control plane and the OpenAI-compatible
// inference passthrough. All policy lives in the caller; these handlers only
// execute mechanics and report state.
package api

import (
	"net/http"

	"github.com/hordruma/gofetch/internal/config"
	"github.com/hordruma/gofetch/internal/engine"
	"github.com/hordruma/gofetch/internal/kvcache"
	"github.com/hordruma/gofetch/internal/models"
	"github.com/hordruma/gofetch/internal/warm"
)

// Server wires the managers behind the HTTP API.
type Server struct {
	cfg  *config.Config
	reg  *models.Registry
	eng  engine.Engine
	warm *warm.Pool
	kv   *kvcache.Store
	mux  *http.ServeMux
}

// New builds the server and registers routes.
func New(cfg *config.Config, reg *models.Registry, eng engine.Engine, pool *warm.Pool, kv *kvcache.Store) *Server {
	s := &Server{cfg: cfg, reg: reg, eng: eng, warm: pool, kv: kv, mux: http.NewServeMux()}
	s.routes()
	return s
}

func (s *Server) routes() {
	s.mux.HandleFunc("GET /v1/status", s.handleStatus)
	s.mux.HandleFunc("POST /v1/vram/load", s.handleVRAMLoad)
	s.mux.HandleFunc("POST /v1/vram/drop", s.handleVRAMDrop)
	s.mux.HandleFunc("POST /v1/ram/warm", s.handleRAMWarm)
	s.mux.HandleFunc("POST /v1/ram/evict", s.handleRAMEvict)
	s.mux.HandleFunc("POST /v1/kv/save", s.handleKVSave)
	s.mux.HandleFunc("POST /v1/kv/load", s.handleKVLoad)
	s.mux.HandleFunc("POST /v1/kv/evict", s.handleKVEvict)

	// OpenAI-compatible passthrough to the active model.
	s.mux.HandleFunc("/v1/chat/completions", s.handleInference)
	s.mux.HandleFunc("/v1/completions", s.handleInference)
	s.mux.HandleFunc("/v1/embeddings", s.handleInference)
}

// ListenAndServe starts the HTTP server.
func (s *Server) ListenAndServe() error {
	return http.ListenAndServe(s.cfg.Listen, s.mux)
}
