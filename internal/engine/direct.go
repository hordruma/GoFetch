package engine

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os/exec"
	"strconv"
	"sync"
	"time"

	"github.com/hordruma/gofetch/internal/models"
)

// upstreamPort is the loopback port GoFetch starts each llama-server on. A single
// VRAM slot means a single port is sufficient.
const upstreamPort = 8099

// Direct spawns and supervises llama-server itself, enforcing the single
// VRAM-slot invariant: at most one model process runs at a time.
type Direct struct {
	bin string
	reg *models.Registry

	mu     sync.Mutex
	active string
	cmd    *exec.Cmd
	proxy  *httputil.ReverseProxy
}

// NewDirect returns an engine that manages llama-server processes.
func NewDirect(bin string, reg *models.Registry) (*Direct, error) {
	if bin == "" {
		return nil, fmt.Errorf("llama-server binary required")
	}
	u, _ := url.Parse("http://127.0.0.1:" + strconv.Itoa(upstreamPort))
	return &Direct{
		bin:   bin,
		reg:   reg,
		proxy: httputil.NewSingleHostReverseProxy(u),
	}, nil
}

// EnsureResident stops any other model and starts modelID, waiting for it to
// become ready.
func (d *Direct) EnsureResident(ctx context.Context, modelID string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.active == modelID && d.cmd != nil {
		return nil
	}
	m, ok := d.reg.Get(modelID)
	if !ok {
		return fmt.Errorf("unknown model %q", modelID)
	}
	d.stopLocked()

	// TODO(M1): tune flags; surface logs; add --slot-save-path for KV snapshots.
	cmd := exec.Command(d.bin,
		"--model", m.Path,
		"--port", strconv.Itoa(upstreamPort),
		"--host", "127.0.0.1",
		"-ngl", strconv.Itoa(m.NGL),
		"-c", strconv.Itoa(m.Ctx),
	)
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start llama-server: %w", err)
	}
	d.cmd = cmd
	d.active = modelID

	if err := d.waitReady(ctx); err != nil {
		d.stopLocked()
		return err
	}
	return nil
}

// Drop stops modelID if it is resident.
func (d *Direct) Drop(ctx context.Context, modelID string) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.active != modelID {
		return nil
	}
	d.stopLocked()
	return nil
}

// Active returns the resident model ID.
func (d *Direct) Active() string {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.active
}

// ServeHTTP proxies to the resident llama-server.
func (d *Direct) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	d.proxy.ServeHTTP(w, r)
}

// stopLocked terminates the running process. Caller holds d.mu.
func (d *Direct) stopLocked() {
	if d.cmd != nil && d.cmd.Process != nil {
		_ = d.cmd.Process.Kill()
		_ = d.cmd.Wait()
	}
	d.cmd = nil
	d.active = ""
}

// waitReady polls the server's /health until it responds or ctx is done.
func (d *Direct) waitReady(ctx context.Context) error {
	url := "http://127.0.0.1:" + strconv.Itoa(upstreamPort) + "/health"
	client := &http.Client{Timeout: 2 * time.Second}
	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("model not ready: %w", ctx.Err())
		default:
		}
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if resp, err := client.Do(req); err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
}
