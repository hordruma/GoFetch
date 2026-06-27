package warm

import (
	"os"
	"path/filepath"
	"testing"
)

func writeFile(t *testing.T, dir, name string, size int) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestWarmEvictsLRUUnderBudget(t *testing.T) {
	dir := t.TempDir()
	a := writeFile(t, dir, "a.gguf", 8192)
	b := writeFile(t, dir, "b.gguf", 8192)

	// Budget fits exactly one file.
	p := New(8192, false)

	if err := p.Warm(a); err != nil {
		t.Fatalf("warm a: %v", err)
	}
	if err := p.Warm(b); err != nil {
		t.Fatalf("warm b: %v", err)
	}

	used, budget, files := p.Status()
	if used > budget {
		t.Fatalf("used %d exceeds budget %d", used, budget)
	}
	if len(files) != 1 {
		t.Fatalf("want 1 warmed file after LRU eviction, got %d", len(files))
	}
	if files[0].Path != b {
		t.Fatalf("want b warmed (a evicted as LRU), got %s", files[0].Path)
	}
}

func TestWarmRejectsOversizeFile(t *testing.T) {
	dir := t.TempDir()
	big := writeFile(t, dir, "big.gguf", 16384)
	p := New(8192, false)
	if err := p.Warm(big); err == nil {
		t.Fatal("want error warming file larger than budget")
	}
}

func TestWarmIdempotent(t *testing.T) {
	dir := t.TempDir()
	a := writeFile(t, dir, "a.gguf", 4096)
	p := New(0, false) // unbounded
	if err := p.Warm(a); err != nil {
		t.Fatal(err)
	}
	if err := p.Warm(a); err != nil {
		t.Fatal(err)
	}
	if used, _, files := p.Status(); len(files) != 1 || used != 4096 {
		t.Fatalf("re-warm should not double-count: used=%d files=%d", used, len(files))
	}
}
