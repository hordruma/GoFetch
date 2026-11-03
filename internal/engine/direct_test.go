package engine

import (
	"context"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/hordruma/gofetch/internal/models"
)

// fakeBin writes an executable shell script that ignores llama-server's flags.
func fakeBin(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "fake-llama-server")
	if err := os.WriteFile(p, []byte("#!/bin/sh\n"+body+"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

func testDirect(t *testing.T, body string) *Direct {
	t.Helper()
	reg := models.NewRegistry([]models.Model{{ID: "m", Path: "/tmp/m.gguf", NGL: 99, Ctx: 512}})
	d, err := NewDirect(fakeBin(t, body), reg, "")
	if err != nil {
		t.Fatal(err)
	}
	d.startupTimeout = 1500 * time.Millisecond
	d.shutdownGrace = time.Second
	// Never probe the real upstream port: a llama-server (or a leftover process)
	// listening there would answer /health and make these tests pass spuriously.
	d.port = freePort(t)
	return d
}

// freePort reserves a loopback port and releases it, so the backend under test
// gets a port nothing else is serving.
func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// A backend that never reports healthy must not block forever: EnsureResident
// holds the engine mutex, so an unbounded wait wedges the whole control plane.
func TestEnsureResidentStartupTimeout(t *testing.T) {
	d := testDirect(t, "exec sleep 30")

	start := time.Now()
	err := d.EnsureResident(context.Background(), "m")
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected a timeout error from a backend that never becomes healthy")
	}
	if !strings.Contains(err.Error(), "not ready") {
		t.Errorf("err = %v, want a not-ready timeout", err)
	}
	if elapsed > 10*time.Second {
		t.Errorf("waited %s, want the startup deadline to bound it", elapsed)
	}
	if d.Active() != "" {
		t.Errorf("active = %q, want empty after a failed load", d.Active())
	}
	if d.cmd != nil {
		t.Error("the timed-out process was not reaped")
	}
}

// A backend that dies during load should report that immediately rather than
// polling a dead port until the deadline.
func TestEnsureResidentDetectsBackendExit(t *testing.T) {
	d := testDirect(t, "exit 3")

	start := time.Now()
	err := d.EnsureResident(context.Background(), "m")
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected an error when the backend exits during startup")
	}
	if !strings.Contains(err.Error(), "exited during startup") {
		t.Errorf("err = %v, want an exit-during-startup error", err)
	}
	if elapsed >= d.startupTimeout {
		t.Errorf("took %s: the exit was not detected, it waited for the deadline", elapsed)
	}
	if d.Active() != "" {
		t.Errorf("active = %q, want empty after a failed load", d.Active())
	}
}

// A caller that goes away must release the engine, not leave it held.
func TestEnsureResidentHonorsCallerCancellation(t *testing.T) {
	d := testDirect(t, "exec sleep 30")
	d.startupTimeout = time.Minute

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	if err := d.EnsureResident(ctx, "m"); err == nil {
		t.Fatal("expected cancellation to abort the load")
	}

	done := make(chan struct{})
	go func() { d.mu.Lock(); d.mu.Unlock(); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("engine mutex still held after the caller was cancelled")
	}
}

func TestNewDirectRequiresBinary(t *testing.T) {
	if _, err := NewDirect("", models.NewRegistry(nil), ""); err == nil {
		t.Fatal("expected an error when no llama-server binary is configured")
	}
}

func TestEnsureResidentUnknownModel(t *testing.T) {
	d := testDirect(t, "exit 0")
	if err := d.EnsureResident(context.Background(), "nope"); err == nil {
		t.Fatal("expected an error for a model that is not in the registry")
	}
}
