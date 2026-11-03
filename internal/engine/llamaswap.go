package engine

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"time"
)

// activeTTL bounds how stale LlamaSwap.Active() may be. Active() is on the
// request path and has no context to cancel with, so it serves a cached answer
// and refreshes it at most this often.
const activeTTL = time.Second

// LlamaSwap drives an external llama-swap instance: llama-swap owns the process
// swap and the OpenAI proxy, GoFetch adds the warm-pool, the KV store, and
// explicit load/unload control on top.
//
// Unlike the direct engine, GoFetch does not choose llama-server's flags here —
// llama-swap's own config does. In particular --slot-save-path must be set
// there for KV snapshots to work, and it should point at the same directory as
// GoFetch's kv.dir.
type LlamaSwap struct {
	upstream *url.URL
	proxy    *httputil.ReverseProxy
	client   *http.Client

	mu      sync.Mutex
	active  string
	checked time.Time
}

// NewLlamaSwap returns an engine that proxies to the llama-swap base URL.
func NewLlamaSwap(upstream string) (*LlamaSwap, error) {
	u, err := url.Parse(upstream)
	if err != nil {
		return nil, fmt.Errorf("parse upstream: %w", err)
	}
	if u.Host == "" {
		return nil, fmt.Errorf("upstream %q has no host", upstream)
	}
	return &LlamaSwap{
		upstream: u,
		proxy:    httputil.NewSingleHostReverseProxy(u),
		client:   &http.Client{Timeout: 10 * time.Minute},
	}, nil
}

// EnsureResident force-loads modelID. llama-swap starts a model when a request
// is routed to it, so GoFetch reaches through llama-swap's direct-access route
// and waits for that model's own health endpoint — which returns only once the
// model is loaded and ready to serve.
func (e *LlamaSwap) EnsureResident(ctx context.Context, modelID string) error {
	if modelID == "" {
		return fmt.Errorf("model required")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		e.modelBase(modelID)+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return fmt.Errorf("force-load %s: %w", modelID, err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("force-load %s: llama-swap %s: %s", modelID, resp.Status, strings.TrimSpace(string(body)))
	}

	e.mu.Lock()
	e.active, e.checked = modelID, time.Now()
	e.mu.Unlock()
	return nil
}

// Drop unloads modelID through llama-swap's management API, freeing its VRAM.
func (e *LlamaSwap) Drop(ctx context.Context, modelID string) error {
	if modelID == "" {
		return nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		e.upstream.JoinPath("api", "models", "unload", modelID).String(), nil)
	if err != nil {
		return err
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return fmt.Errorf("unload %s: %w", modelID, err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4<<10))
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unload %s: llama-swap %s", modelID, resp.Status)
	}

	e.mu.Lock()
	if e.active == modelID {
		e.active = ""
	}
	e.checked = time.Now()
	e.mu.Unlock()
	return nil
}

// Active reports which model llama-swap is actually running, not merely which
// one GoFetch last asked for: llama-swap may unload on its own TTL, or a
// request routed straight at it may have swapped the model behind GoFetch's
// back. The answer is cached briefly because Active() is on the request path.
func (e *LlamaSwap) Active() string {
	e.mu.Lock()
	if time.Since(e.checked) < activeTTL {
		defer e.mu.Unlock()
		return e.active
	}
	e.mu.Unlock()

	running, err := e.running()
	e.mu.Lock()
	defer e.mu.Unlock()
	if err != nil {
		// Upstream unreachable: keep reporting the last known state rather than
		// claiming nothing is loaded.
		return e.active
	}
	e.active, e.checked = running, time.Now()
	return e.active
}

// running asks llama-swap which model is loaded. GoFetch keeps a single VRAM
// slot, so if llama-swap reports several, the first ready one is authoritative.
func (e *LlamaSwap) running() (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, e.upstream.JoinPath("running").String(), nil)
	if err != nil {
		return "", err
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("llama-swap /running: %s", resp.Status)
	}
	var out struct {
		Running []struct {
			Model string `json:"model"`
			State string `json:"state"`
		} `json:"running"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&out); err != nil {
		return "", err
	}
	for _, m := range out.Running {
		if m.State == "ready" {
			return m.Model, nil
		}
	}
	return "", nil
}

// modelBase is llama-swap's direct-access route for one model's own server.
func (e *LlamaSwap) modelBase(modelID string) string {
	return e.upstream.JoinPath("upstream", modelID).String()
}

// SaveKV snapshots the resident model's slot. The request is addressed to that
// model's own llama-server through llama-swap's direct-access route, so it
// cannot be answered by whatever model happens to be routed by default.
//
// This requires --slot-save-path in llama-swap's config for the model; GoFetch
// does not control that backend's flags.
func (e *LlamaSwap) SaveKV(ctx context.Context, name string) (KVStats, error) {
	active := e.Active()
	if active == "" {
		return KVStats{}, fmt.Errorf("no model resident")
	}
	return slotAction(ctx, e.client, e.modelBase(active), "save", name)
}

// LoadKV restores a snapshot into the resident model's slot.
func (e *LlamaSwap) LoadKV(ctx context.Context, name string) (KVStats, error) {
	active := e.Active()
	if active == "" {
		return KVStats{}, fmt.Errorf("no model resident")
	}
	return slotAction(ctx, e.client, e.modelBase(active), "restore", name)
}

// ServeHTTP reverse-proxies OpenAI traffic to llama-swap, which routes it to the
// model named in the request body.
func (e *LlamaSwap) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	e.proxy.ServeHTTP(w, r)
}
