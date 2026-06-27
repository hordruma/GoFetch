package engine

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sync"
	"time"
)

// LlamaSwap drives an external llama-swap instance: it owns the process-swap and
// OpenAI proxy, GoFetch adds the warm-pool and KV store around it. All traffic
// is reverse-proxied to llama-swap, which performs the actual model swap when it
// sees a request naming a different model.
type LlamaSwap struct {
	upstream *url.URL
	proxy    *httputil.ReverseProxy
	client   *http.Client

	mu     sync.Mutex
	active string
}

// NewLlamaSwap returns an engine that proxies to the llama-swap base URL.
func NewLlamaSwap(upstream string) (*LlamaSwap, error) {
	u, err := url.Parse(upstream)
	if err != nil {
		return nil, fmt.Errorf("parse upstream: %w", err)
	}
	return &LlamaSwap{
		upstream: u,
		proxy:    httputil.NewSingleHostReverseProxy(u),
		client:   &http.Client{Timeout: 60 * time.Second},
	}, nil
}

// EnsureResident asks llama-swap to bring modelID online.
//
// TODO(M4): llama-swap loads a model when it first serves a request naming it.
// The robust force-load is its management API; until that is wired, this records
// intent and best-effort pings upstream health so callers can pre-warm. The swap
// still happens lazily on the first real inference request for modelID.
func (e *LlamaSwap) EnsureResident(ctx context.Context, modelID string) error {
	e.mu.Lock()
	e.active = modelID
	e.mu.Unlock()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, e.upstream.JoinPath("health").String(), nil)
	if err != nil {
		return err
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return fmt.Errorf("llama-swap health: %w", err)
	}
	_ = resp.Body.Close()
	return nil
}

// Drop requests llama-swap unload the model.
//
// TODO(M4): call llama-swap's unload endpoint. For now this clears local state.
func (e *LlamaSwap) Drop(ctx context.Context, modelID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.active == modelID {
		e.active = ""
	}
	return nil
}

// Active returns the model GoFetch last asked llama-swap to make resident.
func (e *LlamaSwap) Active() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.active
}

// ServeHTTP reverse-proxies to llama-swap.
func (e *LlamaSwap) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	e.proxy.ServeHTTP(w, r)
}
