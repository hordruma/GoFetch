package kvcache

import (
	"os"
	"testing"
)

func writeSnapshot(t *testing.T, s *Store, model, key string, size int) {
	t.Helper()
	if err := os.WriteFile(s.FileFor(model, key), make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := s.Register(model, key); err != nil {
		t.Fatalf("register %s/%s: %v", model, key, err)
	}
}

func TestKVEvictsLRUUnderBudget(t *testing.T) {
	s, err := New(t.TempDir(), 1024)
	if err != nil {
		t.Fatal(err)
	}
	writeSnapshot(t, s, "m", "doc-a", 1024)
	writeSnapshot(t, s, "m", "doc-b", 1024) // forces doc-a out

	used, budget, items := s.Status()
	if used > budget {
		t.Fatalf("used %d exceeds budget %d", used, budget)
	}
	if len(items) != 1 || items[0].Key != "doc-b" {
		t.Fatalf("want only doc-b retained, got %+v", items)
	}
	if _, ok := s.Touch("m", "doc-a"); ok {
		t.Fatal("doc-a should have been evicted")
	}
}

func TestKVModelBinding(t *testing.T) {
	s, err := New(t.TempDir(), 0)
	if err != nil {
		t.Fatal(err)
	}
	// Same key under different models must map to different files.
	if s.FileFor("model-a", "doc") == s.FileFor("model-b", "doc") {
		t.Fatal("snapshots for different models must not collide")
	}
}
