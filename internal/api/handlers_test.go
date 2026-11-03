package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/hordruma/gofetch/internal/config"
	"github.com/hordruma/gofetch/internal/engine"
	"github.com/hordruma/gofetch/internal/kvcache"
	"github.com/hordruma/gofetch/internal/models"
	"github.com/hordruma/gofetch/internal/warm"
)

// fakeEngine records what the control plane asked it to do.
type fakeEngine struct {
	mu       sync.Mutex
	active   string
	loads    []string
	loadErr  error
	served   int
	lastBody string

	dir      string // slot save directory, shared with the kv store
	snapSize int
	kvErr    error
	saved    []string
	restored []string
}

func (f *fakeEngine) EnsureResident(_ context.Context, id string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.loads = append(f.loads, id)
	if f.loadErr != nil {
		return f.loadErr
	}
	f.active = id
	return nil
}

func (f *fakeEngine) Drop(_ context.Context, id string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.active == id {
		f.active = ""
	}
	return nil
}

func (f *fakeEngine) Active() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.active
}

func (f *fakeEngine) SaveKV(_ context.Context, name string) (engine.KVStats, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.saved = append(f.saved, name)
	if f.kvErr != nil {
		return engine.KVStats{}, f.kvErr
	}
	// Mimic llama.cpp: the snapshot file appears under the slot save directory.
	if err := os.WriteFile(filepath.Join(f.dir, name), make([]byte, f.snapSize), 0o644); err != nil {
		return engine.KVStats{}, err
	}
	return engine.KVStats{Tokens: 128, Bytes: int64(f.snapSize)}, nil
}

func (f *fakeEngine) LoadKV(_ context.Context, name string) (engine.KVStats, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.restored = append(f.restored, name)
	if f.kvErr != nil {
		return engine.KVStats{}, f.kvErr
	}
	return engine.KVStats{Tokens: 128, Restore: true}, nil
}

func (f *fakeEngine) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.served++
	buf := make([]byte, r.ContentLength)
	_, _ = r.Body.Read(buf)
	f.lastBody = string(buf)
	_, _ = fmt.Fprintf(w, `{"upstream":%q}`, f.active)
}

func testServer(t *testing.T) (*Server, *fakeEngine) {
	t.Helper()
	reg := models.NewRegistry([]models.Model{
		{ID: "a", Path: "/tmp/a.gguf", Tier: "cache"},
		{ID: "b", Path: "/tmp/b.gguf", Tier: "locked"},
	})
	dir := t.TempDir()
	kv, err := kvcache.New(dir, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	eng := &fakeEngine{dir: dir, snapSize: 4096}
	return New(&config.Config{}, reg, eng, warm.New(1<<20, 1<<20), kv), eng
}

func do(t *testing.T, s *Server, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(method, path, strings.NewReader(body))
	w := httptest.NewRecorder()
	s.mux.ServeHTTP(w, r)
	return w
}

// The documented contract is that the OpenAI passthrough auto-loads the model
// named in the request, so a plain OpenAI client works without touching the
// control plane.
func TestInferenceAutoLoadsNamedModel(t *testing.T) {
	s, eng := testServer(t)
	w := do(t, s, http.MethodPost, "/v1/chat/completions", `{"model":"b","messages":[]}`)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d (%s), want 200", w.Code, w.Body.String())
	}
	if len(eng.loads) != 1 || eng.loads[0] != "b" {
		t.Errorf("loads = %v, want exactly one load of %q", eng.loads, "b")
	}
	if eng.served != 1 {
		t.Errorf("served = %d, want the request proxied once", eng.served)
	}
	// The body is consumed to read the model name and must reach the engine intact.
	if !strings.Contains(eng.lastBody, `"messages"`) {
		t.Errorf("upstream body = %q, want the original request body replayed", eng.lastBody)
	}
}

func TestInferenceDoesNotReloadResidentModel(t *testing.T) {
	s, eng := testServer(t)
	do(t, s, http.MethodPost, "/v1/vram/load", `{"model":"a"}`)
	do(t, s, http.MethodPost, "/v1/chat/completions", `{"model":"a"}`)

	if len(eng.loads) != 2 {
		t.Errorf("loads = %v; the engine deduplicates, but the handler must still delegate", eng.loads)
	}
	if eng.Active() != "a" {
		t.Errorf("active = %q, want %q", eng.Active(), "a")
	}
}

// With nothing resident and no model named, the caller gets a clear 503 rather
// than a bare proxy error from an upstream that is not running.
func TestInferenceWithoutResidentModel(t *testing.T) {
	s, eng := testServer(t)
	w := do(t, s, http.MethodPost, "/v1/chat/completions", `{"messages":[]}`)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", w.Code)
	}
	if eng.served != 0 {
		t.Error("request was proxied with no model resident")
	}
}

func TestInferenceUnknownModel(t *testing.T) {
	s, eng := testServer(t)
	w := do(t, s, http.MethodPost, "/v1/chat/completions", `{"model":"ghost"}`)

	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
	if len(eng.loads) != 0 {
		t.Errorf("loads = %v, want no attempt to load an unregistered model", eng.loads)
	}
}

func TestInferenceLoadFailureIsReported(t *testing.T) {
	s, eng := testServer(t)
	eng.loadErr = fmt.Errorf("out of VRAM")
	w := do(t, s, http.MethodPost, "/v1/chat/completions", `{"model":"a"}`)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", w.Code)
	}
	if !strings.Contains(w.Body.String(), "out of VRAM") {
		t.Errorf("body = %s, want the underlying load error surfaced", w.Body)
	}
}

func TestVRAMLoadUnknownModel(t *testing.T) {
	s, _ := testServer(t)
	if w := do(t, s, http.MethodPost, "/v1/vram/load", `{"model":"ghost"}`); w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

func TestMalformedBodyRejected(t *testing.T) {
	s, _ := testServer(t)
	for _, path := range []string{"/v1/vram/load", "/v1/ram/warm", "/v1/kv/save"} {
		if w := do(t, s, http.MethodPost, path, "not json"); w.Code != http.StatusBadRequest {
			t.Errorf("%s: status = %d, want 400", path, w.Code)
		}
	}
}

func TestWarmRejectsUnknownTier(t *testing.T) {
	s, _ := testServer(t)
	if w := do(t, s, http.MethodPost, "/v1/ram/warm", `{"model":"a","tier":"bogus"}`); w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestStatusShape(t *testing.T) {
	s, _ := testServer(t)
	w := do(t, s, http.MethodGet, "/v1/status", "")
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d", w.Code)
	}
	var got map[string]json.RawMessage
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("status is not JSON: %v", err)
	}
	for _, k := range []string{"vram", "ram", "kv"} {
		if _, ok := got[k]; !ok {
			t.Errorf("status is missing %q: %s", k, w.Body)
		}
	}
}

func TestKVLoadMissingSnapshot(t *testing.T) {
	s, _ := testServer(t)
	do(t, s, http.MethodPost, "/v1/vram/load", `{"model":"a"}`)
	if w := do(t, s, http.MethodPost, "/v1/kv/load", `{"model":"a","key":"nope"}`); w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

func TestKVSaveRequiresKey(t *testing.T) {
	s, _ := testServer(t)
	if w := do(t, s, http.MethodPost, "/v1/kv/save", `{"model":"a"}`); w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

// Saving must actually drive the engine's slot save, not merely account a file
// somebody else was expected to have written.
func TestKVSaveDrivesEngineAndAccounts(t *testing.T) {
	s, eng := testServer(t)
	do(t, s, http.MethodPost, "/v1/vram/load", `{"model":"a"}`)

	w := do(t, s, http.MethodPost, "/v1/kv/save", `{"model":"a","key":"doc7"}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d (%s), want 200", w.Code, w.Body)
	}
	if len(eng.saved) != 1 {
		t.Fatalf("engine saves = %v, want exactly one", eng.saved)
	}
	// llama.cpp resolves the name beneath its own slot-save directory and
	// rejects separators, so the engine must receive a bare filename.
	if strings.ContainsAny(eng.saved[0], "/\\") {
		t.Errorf("engine got %q, want a bare filename", eng.saved[0])
	}
	var body map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["tokens"] != float64(128) {
		t.Errorf("tokens = %v, want the engine's report surfaced", body["tokens"])
	}
	if w2 := do(t, s, http.MethodGet, "/v1/status", ""); !strings.Contains(w2.Body.String(), `"used":4096`) {
		t.Errorf("status kv usage = %s, want the snapshot accounted", w2.Body)
	}
}

// A snapshot only means anything under the model that produced it, so the
// control plane must refuse to save or restore one against a different model.
func TestKVRefusesNonResidentModel(t *testing.T) {
	s, _ := testServer(t)
	do(t, s, http.MethodPost, "/v1/vram/load", `{"model":"a"}`)

	for _, path := range []string{"/v1/kv/save", "/v1/kv/load"} {
		w := do(t, s, http.MethodPost, path, `{"model":"b","key":"doc7"}`)
		if w.Code != http.StatusConflict {
			t.Errorf("%s with a non-resident model: status = %d, want 409", path, w.Code)
		}
	}
}

func TestKVLoadRestoresThroughEngine(t *testing.T) {
	s, eng := testServer(t)
	do(t, s, http.MethodPost, "/v1/vram/load", `{"model":"a"}`)
	do(t, s, http.MethodPost, "/v1/kv/save", `{"model":"a","key":"doc7"}`)

	w := do(t, s, http.MethodPost, "/v1/kv/load", `{"model":"a","key":"doc7"}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d (%s), want 200", w.Code, w.Body)
	}
	if len(eng.restored) != 1 || eng.restored[0] != eng.saved[0] {
		t.Errorf("restored %v, want the same name that was saved (%v)", eng.restored, eng.saved)
	}
}

// If the engine's snapshot cannot be accounted, it must not be left on the
// tmpfs: the store deletes it, and the caller is told storage was the problem.
func TestKVSaveOverBudgetIsRejectedAndCleaned(t *testing.T) {
	s, eng := testServer(t)
	eng.snapSize = 2 << 20 // larger than the store's 1 MiB budget
	do(t, s, http.MethodPost, "/v1/vram/load", `{"model":"a"}`)

	w := do(t, s, http.MethodPost, "/v1/kv/save", `{"model":"a","key":"huge"}`)
	if w.Code != http.StatusInsufficientStorage {
		t.Errorf("status = %d, want 507", w.Code)
	}
	left, _ := filepath.Glob(filepath.Join(eng.dir, "*.kv"))
	if len(left) != 0 {
		t.Errorf("%d snapshot(s) left on disk after a rejected save", len(left))
	}
}

func TestKVSaveEngineFailureReported(t *testing.T) {
	s, eng := testServer(t)
	eng.kvErr = fmt.Errorf("slot busy")
	do(t, s, http.MethodPost, "/v1/vram/load", `{"model":"a"}`)

	w := do(t, s, http.MethodPost, "/v1/kv/save", `{"model":"a","key":"doc7"}`)
	if w.Code != http.StatusInternalServerError || !strings.Contains(w.Body.String(), "slot busy") {
		t.Errorf("status = %d body = %s, want 500 with the engine error", w.Code, w.Body)
	}
}
