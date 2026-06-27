package api

import (
	"encoding/json"
	"net/http"
)

// modelReq is the body for endpoints addressing a single model.
type modelReq struct {
	Model string `json:"model"`
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
	ramUsed, ramBudget, files := s.warm.Status()
	kvUsed, kvBudget, kvItems := s.kv.Status()
	writeJSON(w, http.StatusOK, map[string]any{
		"vram": map[string]any{"active": s.eng.Active()},
		"ram":  map[string]any{"used": ramUsed, "budget": ramBudget, "files": files},
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
	path, ok := s.resolvePath(w, req.Model)
	if !ok {
		return
	}
	if err := s.warm.Warm(path); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"warmed": req.Model})
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

// handleKVSave records a KV snapshot the engine has just written. The actual
// slot save is performed by llama.cpp into Store.FileFor(model, key); this
// endpoint accounts it against the KV budget.
//
// TODO(M3): trigger the llama.cpp slot save here rather than only accounting it.
func (s *Server) handleKVSave(w http.ResponseWriter, r *http.Request) {
	var req kvReq
	if !decode(w, r, &req) {
		return
	}
	if req.Key == "" {
		writeErr(w, http.StatusBadRequest, "key required")
		return
	}
	if err := s.kv.Register(req.Model, req.Key); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"saved": req.Key, "file": s.kv.FileFor(req.Model, req.Key)})
}

func (s *Server) handleKVLoad(w http.ResponseWriter, r *http.Request) {
	var req kvReq
	if !decode(w, r, &req) {
		return
	}
	file, ok := s.kv.Touch(req.Model, req.Key)
	if !ok {
		writeErr(w, http.StatusNotFound, "no snapshot for "+req.Model+"/"+req.Key)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"file": file})
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

// handleInference proxies OpenAI-compatible traffic to the active model.
func (s *Server) handleInference(w http.ResponseWriter, r *http.Request) {
	s.eng.ServeHTTP(w, r)
}
