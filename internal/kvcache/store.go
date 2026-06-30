// Package kvcache manages document-level KV snapshots. llama.cpp's server writes
// per-slot KV state to files under --slot-save-path; GoFetch points that path at
// a tmpfs (e.g. /dev/shm) so snapshots live in RAM, and tracks them under a byte
// budget with LRU eviction.
//
// A snapshot is keyed by (model, key). It is ONLY valid for the exact model that
// produced it and the exact token positions it occupied — callers must never
// load a snapshot under a different model. GoFetch enforces the model binding by
// encoding it into the filename; position validity is the caller's contract.
package kvcache

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

type meta struct {
	model string
	key   string
	file  string
	size  int64
	last  time.Time
}

// Store is a concurrency-safe, budget-bounded set of KV snapshots on disk
// (intended to be tmpfs).
type Store struct {
	mu      sync.Mutex
	dir     string
	budget  int64
	used    int64
	entries map[string]*meta // by snapshot id
	now     func() time.Time
}

// New creates the snapshot directory and returns a store.
func New(dir string, budget int64) (*Store, error) {
	if dir == "" {
		return nil, fmt.Errorf("kv dir required")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("mkdir kv dir: %w", err)
	}
	return &Store{
		dir:     dir,
		budget:  budget,
		entries: make(map[string]*meta),
		now:     time.Now,
	}, nil
}

// id is a stable, collision-resistant identifier for a (model, key) pair.
func id(model, key string) string {
	sum := sha256.Sum256([]byte(model + "\x00" + key))
	return hex.EncodeToString(sum[:16])
}

// FileFor returns the absolute snapshot path for (model, key). This is the path
// GoFetch passes to llama.cpp as the slot save/restore filename.
func (s *Store) FileFor(model, key string) string {
	return filepath.Join(s.dir, id(model, key)+".kv")
}

// Register records a snapshot that has just been written to FileFor(model, key),
// accounting its size and evicting LRU snapshots to stay within budget.
func (s *Store) Register(model, key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	file := s.FileFor(model, key)
	fi, err := os.Stat(file)
	if err != nil {
		return fmt.Errorf("stat snapshot: %w", err)
	}
	sid := id(model, key)
	if old, ok := s.entries[sid]; ok {
		s.used -= old.size
		delete(s.entries, sid)
	}
	if s.budget > 0 && fi.Size() > s.budget {
		return fmt.Errorf("snapshot %d bytes exceeds KV budget %d", fi.Size(), s.budget)
	}
	if err := s.evictToFitLocked(sid, fi.Size()); err != nil {
		return err
	}
	s.entries[sid] = &meta{model: model, key: key, file: file, size: fi.Size(), last: s.now()}
	s.used += fi.Size()
	return nil
}

// Touch refreshes the LRU timestamp on load. Returns the file path and whether
// the snapshot is known.
func (s *Store) Touch(model, key string) (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	m, ok := s.entries[id(model, key)]
	if !ok {
		return "", false
	}
	m.last = s.now()
	return m.file, true
}

// Evict removes a snapshot and its file. No-op if unknown.
func (s *Store) Evict(model, key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.removeLocked(id(model, key))
}

// Info is a status snapshot of one KV entry.
type Info struct {
	Model string    `json:"model"`
	Key   string    `json:"key"`
	Size  int64     `json:"size"`
	Last  time.Time `json:"last"`
}

// Status returns usage and the retained snapshots, newest first.
func (s *Store) Status() (used, budget int64, items []Info) {
	s.mu.Lock()
	defer s.mu.Unlock()
	items = make([]Info, 0, len(s.entries))
	for _, m := range s.entries {
		items = append(items, Info{Model: m.model, Key: m.key, Size: m.size, Last: m.last})
	}
	sort.Slice(items, func(i, j int) bool { return items[i].Last.After(items[j].Last) })
	return s.used, s.budget, items
}

func (s *Store) evictToFitLocked(except string, need int64) error {
	if s.budget <= 0 {
		return nil
	}
	for s.used+need > s.budget {
		victim := s.lruLocked(except)
		if victim == "" {
			return fmt.Errorf("cannot free %d bytes under KV budget %d", need, s.budget)
		}
		if err := s.removeLocked(victim); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) lruLocked(except string) string {
	var oldestID string
	var oldest *meta
	for sid, m := range s.entries {
		if sid == except {
			continue
		}
		if oldest == nil || m.last.Before(oldest.last) {
			oldest, oldestID = m, sid
		}
	}
	return oldestID
}

func (s *Store) removeLocked(sid string) error {
	m, ok := s.entries[sid]
	if !ok {
		return nil
	}
	err := os.Remove(m.file)
	delete(s.entries, sid)
	s.used -= m.size
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove snapshot: %w", err)
	}
	return nil
}
