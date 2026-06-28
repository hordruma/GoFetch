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

func TestCacheTierEvictsLRUUnderBudget(t *testing.T) {
	dir := t.TempDir()
	a := writeFile(t, dir, "a.gguf", 8192)
	b := writeFile(t, dir, "b.gguf", 8192)

	p := New(8192, 0) // cache budget fits exactly one file

	if err := p.Warm(a, TierCache); err != nil {
		t.Fatalf("warm a: %v", err)
	}
	if err := p.Warm(b, TierCache); err != nil {
		t.Fatalf("warm b: %v", err)
	}

	st := p.Snapshot()
	if st.CacheUsed > st.CacheBudget {
		t.Fatalf("cache used %d exceeds budget %d", st.CacheUsed, st.CacheBudget)
	}
	if len(st.Files) != 1 || st.Files[0].Path != b {
		t.Fatalf("want only b warmed (a evicted as LRU), got %+v", st.Files)
	}
}

func TestTiersHaveIndependentBudgets(t *testing.T) {
	dir := t.TempDir()
	a := writeFile(t, dir, "a.gguf", 4096)
	b := writeFile(t, dir, "b.gguf", 4096)

	// Cache budget fits one; locked budget fits one. Warming a locked file must
	// NOT evict the cache file — separate pools.
	p := New(4096, 4096)
	if err := p.Warm(a, TierCache); err != nil {
		t.Fatalf("warm a cache: %v", err)
	}
	if err := p.Warm(b, TierLocked); err != nil {
		// mlock may be denied in restricted CI; skip rather than fail.
		t.Skipf("locked warm unavailable (mlock denied?): %v", err)
	}
	st := p.Snapshot()
	if len(st.Files) != 2 {
		t.Fatalf("want both tiers populated, got %+v", st.Files)
	}
	if st.CacheUsed != 4096 || st.LockUsed != 4096 {
		t.Fatalf("tier accounting wrong: %+v", st)
	}
}

func TestRetierPromotesInPlace(t *testing.T) {
	dir := t.TempDir()
	a := writeFile(t, dir, "a.gguf", 4096)
	p := New(0, 0) // both unbounded

	if err := p.Warm(a, TierCache); err != nil {
		t.Fatal(err)
	}
	if err := p.Warm(a, TierLocked); err != nil {
		t.Skipf("locked warm unavailable: %v", err)
	}
	st := p.Snapshot()
	if len(st.Files) != 1 {
		t.Fatalf("retier should not duplicate: %+v", st.Files)
	}
	if st.Files[0].Tier != "locked" {
		t.Fatalf("want promoted to locked, got %s", st.Files[0].Tier)
	}
	if st.CacheUsed != 0 {
		t.Fatalf("cache tier should be empty after promotion, used=%d", st.CacheUsed)
	}
}

func TestWarmRejectsOversizeFile(t *testing.T) {
	dir := t.TempDir()
	big := writeFile(t, dir, "big.gguf", 16384)
	p := New(8192, 0)
	if err := p.Warm(big, TierCache); err == nil {
		t.Fatal("want error warming file larger than cache budget")
	}
}
