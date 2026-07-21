package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"

	"github.com/hordruma/gofetch/internal/warm"
)

// modelReq is the body for endpoints addressing a single model. Tier is only
// honored by /v1/ram/warm; empty falls back to the model's configured tier.
type modelReq struct {
	Model string `json:"model"`
	Tier  string `json:"tier,omitempty"`
}

// kvReq is the body for KV snapshot endpoints.
type kvReq struct {
	Model string `json:"model"`
	Key   string `json:"key"`
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return false
	}
	return true
}

// resolvePath returns the GGUF path for a model ID, or "" + error response.
func (s *Server) resolvePath(w http.ResponseWriter, id string) (string, bool) {
	m, ok := s.reg.Get(id)
	if !ok {
		writeErr(w, http.StatusNotFound, "unknown model "+id)
		return "", false
	}
	return m.Path, true
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	kvUsed, kvBudget, kvItems := s.kv.Status()
	writeJSON(w, http.StatusOK, map[string]any{
		"vram": map[string]any{"active": s.eng.Active()},
		"ram":  s.warm.Snapshot(),
		"kv":   map[string]any{"used": kvUsed, "budget": kvBudget, "items": kvItems},
	})
}

func (s *Server) handleVRAMLoad(w http.ResponseWriter, r *http.Request) {
	var req modelReq
	if !decode(w, r, &req) {
		return
	}
	if _, ok := s.reg.Get(req.Model); !ok {
		writeErr(w, http.StatusNotFound, "unknown model "+req.Model)
		return
	}
	if err := s.eng.EnsureResident(r.Context(), req.Model); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"active": s.eng.Active()})
}

func (s *Server) handleVRAMDrop(w http.ResponseWriter, r *http.Request) {
	var req modelReq
	if !decode(w, r, &req) {
		return
	}
	if err := s.eng.Drop(r.Context(), req.Model); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"active": s.eng.Active()})
}

func (s *Server) handleRAMWarm(w http.ResponseWriter, r *http.Request) {
	var req modelReq
	if !decode(w, r, &req) {
		return
	}
	m, ok := s.reg.Get(req.Model)
	if !ok {
		writeErr(w, http.StatusNotFound, "unknown model "+req.Model)
		return
	}
	tierStr := req.Tier
	if tierStr == "" {
		tierStr = m.Tier // fall back to the model's configured default tier
	}
	tier, err := warm.ParseTier(tierStr)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := s.warm.Warm(m.Path, tier); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"warmed": req.Model, "tier": tier.String()})
}

func (s *Server) handleRAMEvict(w http.ResponseWriter, r *http.Request) {
	var req modelReq
	if !decode(w, r, &req) {
		return
	}
	path, ok := s.resolvePath(w, req.Model)
	if !ok {
		return
	}
	if err := s.warm.Evict(path); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"evicted": req.Model})
}

// handleKVSave tells the engine to snapshot the resident model's prompt cache,
// then accounts the resulting file against the KV budget.
//
// The snapshot is only valid for the model that produced it, so the model named
// here must be the resident one; loading it under a different model would feed
// another model's attention state into this one.
func (s *Server) handleKVSave(w http.ResponseWriter, r *http.Request) {
	var req kvReq
	if !decode(w, r, &req) {
		return
	}
	if !s.checkKVRequest(w, req) {
		return
	}
	stats, err := s.eng.SaveKV(r.Context(), s.kv.NameFor(req.Model, req.Key))
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "save slot: "+err.Error())
		return
	}
	// Account what the engine actually wrote. A snapshot the budget rejects is
	// deleted by the store, so the engine's file never outlives its accounting.
	if err := s.kv.Register(req.Model, req.Key); err != nil {
		writeErr(w, http.StatusInsufficientStorage, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"saved": req.Key, "file": s.kv.FileFor(req.Model, req.Key),
		"tokens": stats.Tokens, "bytes": stats.Bytes, "ms": stats.Millis,
	})
}

// handleKVLoad restores a previously saved snapshot into the resident model's
// prompt cache, so the next request over that document prefix skips prefill.
func (s *Server) handleKVLoad(w http.ResponseWriter, r *http.Request) {
	var req kvReq
	if !decode(w, r, &req) {
		return
	}
	if !s.checkKVRequest(w, req) {
		return
	}
	file, ok := s.kv.Touch(req.Model, req.Key)
	if !ok {
		writeErr(w, http.StatusNotFound, "no snapshot for "+req.Model+"/"+req.Key)
		return
	}
	stats, err := s.eng.LoadKV(r.Context(), s.kv.NameFor(req.Model, req.Key))
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "restore slot: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"file": file, "tokens": stats.Tokens, "bytes": stats.Bytes, "ms": stats.Millis,
	})
}

// checkKVRequest validates the request names a known, resident model and a key.
func (s *Server) checkKVRequest(w http.ResponseWriter, req kvReq) bool {
	model := req.Model
	if model == "" {
		writeErr(w, http.StatusBadRequest, "model required")
		return false
	}
	if req.Key == "" {
		writeErr(w, http.StatusBadRequest, "key required")
		return false
	}
	if _, ok := s.reg.Get(model); !ok {
		writeErr(w, http.StatusNotFound, "unknown model "+model)
		return false
	}
	if active := s.eng.Active(); active != model {
		writeErr(w, http.StatusConflict,
			"model "+model+" is not resident (active: "+quoteEmpty(active)+"); KV snapshots are bound to the model that produced them")
		return false
	}
	return true
}

func quoteEmpty(s string) string {
	if s == "" {
		return "none"
	}
	return s
}

func (s *Server) handleKVEvict(w http.ResponseWriter, r *http.Request) {
	var req kvReq
	if !decode(w, r, &req) {
		return
	}
	if err := s.kv.Evict(req.Model, req.Key); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"evicted": req.Key})
}

// handleInference proxies OpenAI-compatible traffic to the active model,
// auto-loading the model named in the request body if it is not already
// resident. This keeps the passthrough usable by a plain OpenAI client that
// knows nothing about GoFetch's control plane — the caller still decides which
// model to use, GoFetch only executes the swap it implies.
func (s *Server) handleInference(w http.ResponseWriter, r *http.Request) {
	// The body must be buffered: it is inspected for the model name here and
	// then replayed to the upstream engine.
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "read body: "+err.Error())
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))

	var req struct {
		Model string `json:"model"`
	}
	_ = json.Unmarshal(body, &req) // a body without a model just uses the resident one

	switch {
	case req.Model != "":
		if _, ok := s.reg.Get(req.Model); !ok {
			writeErr(w, http.StatusNotFound, "unknown model "+req.Model)
			return
		}
		if err := s.eng.EnsureResident(r.Context(), req.Model); err != nil {
			writeErr(w, http.StatusServiceUnavailable, "load "+req.Model+": "+err.Error())
			return
		}
	case s.eng.Active() == "":
		writeErr(w, http.StatusServiceUnavailable, "no model resident: name one in the request body or POST /v1/vram/load first")
		return
	}
	s.eng.ServeHTTP(w, r)
}
