package engine

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os/exec"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/hordruma/gofetch/internal/models"
)

// upstreamPort is the loopback port GoFetch starts each llama-server on. A single
// VRAM slot means a single port is sufficient.
const upstreamPort = 8099

// defaultStartupTimeout bounds how long EnsureResident waits for a freshly
// spawned llama-server to report healthy. Loading a large model off cold disk
// into VRAM is slow, so this is generous — but it must be finite: the engine
// mutex is held for the whole wait, so an unbounded wait wedges every other
// control-plane operation behind it.
const defaultStartupTimeout = 5 * time.Minute

// defaultShutdownGrace is how long a model gets to exit on SIGTERM before it is
// killed. llama-server frees VRAM on a clean exit; SIGKILL leaves that to the
// driver's process teardown.
const defaultShutdownGrace = 10 * time.Second

// Direct spawns and supervises llama-server itself, enforcing the single
// VRAM-slot invariant: at most one model process runs at a time.
type Direct struct {
	bin          string
	reg          *models.Registry
	slotSavePath string // llama.cpp --slot-save-path; empty disables KV snapshots
	port         int    // loopback port for the backend; a field so tests can isolate

	// startupTimeout and shutdownGrace are fields rather than constants so tests
	// can drive the timeout paths without waiting minutes.
	startupTimeout time.Duration
	shutdownGrace  time.Duration

	kvClient *http.Client // snapshot ops; separate from the streaming proxy

	mu     sync.Mutex
	active string
	cmd    *exec.Cmd
	exited chan error // closed-over cmd.Wait result; one per running process
	proxy  *httputil.ReverseProxy
}

// NewDirect returns an engine that manages llama-server processes. slotSavePath
// is the directory llama.cpp writes KV snapshots into (the kvcache store's
// directory); pass "" to run without KV snapshot support.
func NewDirect(bin string, reg *models.Registry, slotSavePath string) (*Direct, error) {
	if bin == "" {
		return nil, fmt.Errorf("llama-server binary required")
	}
	u, _ := url.Parse("http://127.0.0.1:" + strconv.Itoa(upstreamPort))
	return &Direct{
		bin:            bin,
		reg:            reg,
		slotSavePath:   slotSavePath,
		port:           upstreamPort,
		startupTimeout: defaultStartupTimeout,
		shutdownGrace:  defaultShutdownGrace,
		kvClient:       &http.Client{Timeout: 2 * time.Minute},
		proxy:          httputil.NewSingleHostReverseProxy(u),
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

	args := []string{
		"--model", m.Path,
		"--port", strconv.Itoa(d.port),
		"--host", "127.0.0.1",
		"-ngl", strconv.Itoa(m.NGL),
		"-c", strconv.Itoa(m.Ctx),
		// One slot, so "the prompt cache" of the resident model is unambiguous
		// and KV snapshots always address the same slot. GoFetch swaps models
		// sequentially; it does not multiplex one model across parallel slots.
		"-np", "1",
	}
	if d.slotSavePath != "" {
		// llama.cpp writes/reads per-slot KV snapshots under this directory; the
		// kvcache store owns the filenames beneath it.
		args = append(args, "--slot-save-path", d.slotSavePath)
	}
	cmd := exec.Command(d.bin, args...)
	// Without this the backend's diagnostics go nowhere, so a model that fails to
	// load (bad GGUF, out of VRAM, unsupported quant) fails completely silently.
	cmd.Stdout = &backendLog{model: modelID}
	cmd.Stderr = &backendLog{model: modelID}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start llama-server: %w", err)
	}
	// Buffered-then-closed: the first receiver (usually waitReady) gets the exit
	// error, and every later receive — stopLocked's — returns immediately rather
	// than waiting for a second notification that will never come.
	exited := make(chan error, 1)
	go func() {
		exited <- cmd.Wait()
		close(exited)
	}()
	d.cmd = cmd
	d.exited = exited
	d.active = modelID

	if err := d.waitReady(ctx, exited); err != nil {
		d.stopLocked()
		return err
	}
	slog.Info("model resident", "model", modelID, "pid", cmd.Process.Pid)
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

// SaveKV snapshots the resident model's slot to name under --slot-save-path.
func (d *Direct) SaveKV(ctx context.Context, name string) (KVStats, error) {
	if err := d.kvReady(); err != nil {
		return KVStats{}, err
	}
	return slotAction(ctx, d.kvClient, d.upstreamBase(), "save", name)
}

// LoadKV restores a snapshot into the resident model's slot.
func (d *Direct) LoadKV(ctx context.Context, name string) (KVStats, error) {
	if err := d.kvReady(); err != nil {
		return KVStats{}, err
	}
	return slotAction(ctx, d.kvClient, d.upstreamBase(), "restore", name)
}

// kvReady reports whether a snapshot operation can be attempted at all.
func (d *Direct) kvReady() error {
	if d.slotSavePath == "" {
		return fmt.Errorf("KV snapshots disabled: no slot save path configured")
	}
	if d.Active() == "" {
		return fmt.Errorf("no model resident")
	}
	return nil
}

func (d *Direct) upstreamBase() string {
	return "http://127.0.0.1:" + strconv.Itoa(d.port)
}

// stopLocked terminates the running process, giving it a chance to exit cleanly
// so it frees VRAM itself. Caller holds d.mu.
func (d *Direct) stopLocked() {
	if d.cmd != nil && d.cmd.Process != nil {
		_ = d.cmd.Process.Signal(syscall.SIGTERM)
		select {
		case <-d.exited:
		case <-time.After(d.shutdownGrace):
			_ = d.cmd.Process.Kill()
			<-d.exited // the reaper goroutine always reports, so this cannot hang
		}
	}
	d.cmd = nil
	d.exited = nil
	d.active = ""
}

// waitReady polls the server's /health until it reports ready, the process dies,
// the caller goes away, or the startup deadline passes. It never waits forever:
// the engine mutex is held across the whole call.
func (d *Direct) waitReady(ctx context.Context, exited <-chan error) error {
	ctx, cancel := context.WithTimeout(ctx, d.startupTimeout)
	defer cancel()

	url := "http://127.0.0.1:" + strconv.Itoa(d.port) + "/health"
	client := &http.Client{Timeout: 2 * time.Second}
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	for {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if resp, err := client.Do(req); err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		select {
		case err := <-exited:
			// Died during load — report that instead of polling a dead port until
			// the deadline. The backend's own logs carry the reason.
			return fmt.Errorf("llama-server exited during startup: %v", err)
		case <-ctx.Done():
			return fmt.Errorf("model not ready after %s: %w", d.startupTimeout, ctx.Err())
		case <-ticker.C:
		}
	}
}

// backendLog forwards a backend's output into GoFetch's structured log, one
// line per record, tagged with the model that produced it.
type backendLog struct {
	model string
	buf   []byte
}

func (l *backendLog) Write(p []byte) (int, error) {
	l.buf = append(l.buf, p...)
	for {
		i := bytes.IndexByte(l.buf, '\n')
		if i < 0 {
			break
		}
		if line := bytes.TrimRight(l.buf[:i], "\r"); len(line) > 0 {
			slog.Info("llama-server", "model", l.model, "out", string(line))
		}
		l.buf = l.buf[i+1:]
	}
	return len(p), nil
}
