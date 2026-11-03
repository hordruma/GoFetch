package kvcache

import (
	"os"
	"path/filepath"
	"testing"
)

func snapshot(t *testing.T, s *Store, model, key string, size int) {
	t.Helper()
	if err := os.WriteFile(s.FileFor(model, key), make([]byte, size), 0o644); err != nil {
		t.Fatalf("write snapshot: %v", err)
	}
}

func onDisk(t *testing.T, dir string) int64 {
	t.Helper()
	matches, err := filepath.Glob(filepath.Join(dir, "*.kv"))
	if err != nil {
		t.Fatal(err)
	}
	var total int64
	for _, m := range matches {
		fi, err := os.Stat(m)
		if err != nil {
			t.Fatal(err)
		}
		total += fi.Size()
	}
	return total
}

// A snapshot rejected for exceeding the budget must not stay on the tmpfs the
// budget exists to bound: it is unregistered, so nothing would ever remove it
// and the store's accounting would drift from reality.
func TestRegisterRemovesOversizeSnapshot(t *testing.T) {
	dir := t.TempDir()
	s, err := New(dir, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	snapshot(t, s, "m", "toobig", 4<<20)
	if err := s.Register("m", "toobig"); err == nil {
		t.Fatal("expected the oversize snapshot to be rejected")
	}
	used, _, _ := s.Status()
	if disk := onDisk(t, dir); disk != used {
		t.Errorf("store accounts %d bytes but %d bytes remain on disk (orphan leaked)", used, disk)
	}
}

// Snapshots from a previous process cannot be looked up again — the filename is
// a hash of (model, key) — so they must not be inherited as untracked usage.
func TestNewPurgesStaleSnapshots(t *testing.T) {
	dir := t.TempDir()
	s, err := New(dir, 1<<30)
	if err != nil {
		t.Fatal(err)
	}
	snapshot(t, s, "m", "doc", 1<<20)
	if err := s.Register("m", "doc"); err != nil {
		t.Fatal(err)
	}
	keep := filepath.Join(dir, "unrelated.txt")
	if err := os.WriteFile(keep, []byte("not ours"), 0o644); err != nil {
		t.Fatal(err)
	}

	s2, err := New(dir, 1<<30) // restart
	if err != nil {
		t.Fatal(err)
	}
	if disk := onDisk(t, dir); disk != 0 {
		t.Errorf("%d bytes of stale snapshots survived the restart", disk)
	}
	if used, _, items := s2.Status(); used != 0 || len(items) != 0 {
		t.Errorf("fresh store reports used=%d items=%d, want 0/0", used, len(items))
	}
	if _, err := os.Stat(keep); err != nil {
		t.Errorf("purge removed a non-snapshot file: %v", err)
	}
}
