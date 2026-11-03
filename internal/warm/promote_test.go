package warm

import "testing"

// A promotion the pool refuses must leave the file warm at its original tier.
// Releasing the old mapping before the new tier's budget check dropped the file
// from the pool entirely.
func TestWarmRejectedPromotionKeepsOriginalTier(t *testing.T) {
	dir := t.TempDir()
	path := writeFile(t, dir, "model.gguf", 4<<20)

	p := New(64<<20, 1<<20) // locked budget far too small for a 4 MiB file
	if err := p.Warm(path, TierCache); err != nil {
		t.Fatalf("initial warm: %v", err)
	}
	if err := p.Warm(path, TierLocked); err == nil {
		t.Fatal("promotion over the locked budget should fail")
	}

	st := p.Snapshot()
	if st.CacheUsed != 4<<20 {
		t.Errorf("cache_used = %d, want %d (file was dropped by a failed promotion)", st.CacheUsed, 4<<20)
	}
	if len(st.Files) != 1 || st.Files[0].Tier != "cache" {
		t.Errorf("files = %+v, want the file still warm at tier cache", st.Files)
	}
}

// A promotion that succeeds must move the file, not duplicate its accounting.
func TestWarmPromotionMovesAccounting(t *testing.T) {
	dir := t.TempDir()
	path := writeFile(t, dir, "model.gguf", 1<<20)

	p := New(64<<20, 64<<20)
	if err := p.Warm(path, TierCache); err != nil {
		t.Fatalf("warm cache: %v", err)
	}
	if err := p.Warm(path, TierLocked); err != nil {
		t.Skipf("mlock unavailable in this environment: %v", err)
	}
	st := p.Snapshot()
	if st.CacheUsed != 0 || st.LockUsed != 1<<20 {
		t.Errorf("cache_used=%d locked_used=%d, want 0 and %d", st.CacheUsed, st.LockUsed, 1<<20)
	}
	if len(st.Files) != 1 {
		t.Errorf("files = %+v, want exactly one entry", st.Files)
	}
}
