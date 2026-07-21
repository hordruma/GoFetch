package models

import "testing"

func TestRegistryLookup(t *testing.T) {
	r := NewRegistry([]Model{{ID: "a", Path: "/a.gguf"}, {ID: "b", Path: "/b.gguf"}})

	m, ok := r.Get("b")
	if !ok || m.Path != "/b.gguf" {
		t.Errorf("Get(b) = %+v, %v; want the registered model", m, ok)
	}
	if _, ok := r.Get("ghost"); ok {
		t.Error("Get on an unregistered ID reported success")
	}
}

// Duplicate IDs are last-wins, but must not appear twice in the listing.
func TestRegistryDuplicatesKeepOrder(t *testing.T) {
	r := NewRegistry([]Model{
		{ID: "a", Path: "/first.gguf"},
		{ID: "b", Path: "/b.gguf"},
		{ID: "a", Path: "/second.gguf"},
	})

	all := r.All()
	if len(all) != 2 {
		t.Fatalf("All() = %+v, want 2 entries", all)
	}
	if all[0].ID != "a" || all[1].ID != "b" {
		t.Errorf("All() order = %q,%q; want registration order a,b", all[0].ID, all[1].ID)
	}
	if all[0].Path != "/second.gguf" {
		t.Errorf("duplicate ID resolved to %q, want the last definition", all[0].Path)
	}
}

func TestEmptyRegistry(t *testing.T) {
	r := NewRegistry(nil)
	if len(r.All()) != 0 {
		t.Error("empty registry is not empty")
	}
	if _, ok := r.Get(""); ok {
		t.Error("empty ID resolved")
	}
}
