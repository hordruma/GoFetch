// Package warm manages the RAM warm-pool: GGUF files held hot in the OS page
// cache so a subsequent llama-server start loads at RAM speed instead of disk
// speed. Warming is done by mmap-ing the file, faulting every page in, and
// optionally mlock-ing it. Eviction is LRU under a byte budget.
//
// Linux only (uses mmap/mlock/madvise via the syscall package).
package warm

import (
	"fmt"
	"os"
	"sort"
	"sync"
	"syscall"
	"time"
)

const pageSize = 4096

type entry struct {
	path   string
	data   []byte
	size   int64
	last   time.Time
	locked bool
}

// Pool is a concurrency-safe set of warmed files bounded by a byte budget.
type Pool struct {
	mu      sync.Mutex
	entries map[string]*entry
	used    int64
	budget  int64
	lock    bool
	now     func() time.Time
}

// New constructs a warm-pool. If lock is true, warmed files are mlock-ed (this
// requires sufficient RLIMIT_MEMLOCK).
func New(budget int64, lock bool) *Pool {
	return &Pool{
		entries: make(map[string]*entry),
		budget:  budget,
		lock:    lock,
		now:     time.Now,
	}
}

// Info is a status snapshot of one warmed file.
type Info struct {
	Path   string    `json:"path"`
	Size   int64     `json:"size"`
	Locked bool      `json:"locked"`
	Last   time.Time `json:"last"`
}

// Warm pulls a file into the page cache, evicting LRU entries first if needed
// to stay within budget. Re-warming an already-warm file just refreshes its LRU
// timestamp.
func (p *Pool) Warm(path string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if e, ok := p.entries[path]; ok {
		e.last = p.now()
		return nil
	}

	fi, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("stat %s: %w", path, err)
	}
	size := fi.Size()
	if size <= 0 {
		return fmt.Errorf("empty file %s", path)
	}
	if p.budget > 0 && size > p.budget {
		return fmt.Errorf("file %s (%d bytes) exceeds RAM budget (%d)", path, size, p.budget)
	}
	if err := p.evictToFitLocked(size); err != nil {
		return err
	}

	data, locked, err := mapAndFault(path, size, p.lock)
	if err != nil {
		return err
	}
	p.entries[path] = &entry{path: path, data: data, size: size, last: p.now(), locked: locked}
	p.used += size
	return nil
}

// Evict releases a warmed file. It is a no-op if the file is not warm.
func (p *Pool) Evict(path string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	e, ok := p.entries[path]
	if !ok {
		return nil
	}
	return p.releaseLocked(e)
}

// Status returns a snapshot of the warm-pool and its byte usage.
func (p *Pool) Status() (used, budget int64, files []Info) {
	p.mu.Lock()
	defer p.mu.Unlock()
	files = make([]Info, 0, len(p.entries))
	for _, e := range p.entries {
		files = append(files, Info{Path: e.path, Size: e.size, Locked: e.locked, Last: e.last})
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Last.After(files[j].Last) })
	return p.used, p.budget, files
}

// evictToFitLocked frees LRU entries until need bytes fit under budget. Caller
// holds p.mu.
func (p *Pool) evictToFitLocked(need int64) error {
	if p.budget <= 0 {
		return nil
	}
	for p.used+need > p.budget {
		victim := p.lruLocked()
		if victim == nil {
			return fmt.Errorf("cannot free %d bytes under budget %d", need, p.budget)
		}
		if err := p.releaseLocked(victim); err != nil {
			return err
		}
	}
	return nil
}

func (p *Pool) lruLocked() *entry {
	var oldest *entry
	for _, e := range p.entries {
		if oldest == nil || e.last.Before(oldest.last) {
			oldest = e
		}
	}
	return oldest
}

// releaseLocked unmaps (and unlocks) a warmed file. Caller holds p.mu.
func (p *Pool) releaseLocked(e *entry) error {
	if e.locked {
		_ = syscall.Munlock(e.data)
	}
	err := syscall.Munmap(e.data)
	delete(p.entries, e.path)
	p.used -= e.size
	if err != nil {
		return fmt.Errorf("munmap %s: %w", e.path, err)
	}
	return nil
}

// mapAndFault mmaps the file read-only, advises WILLNEED, touches every page to
// fault it into the page cache, and optionally mlocks it.
func mapAndFault(path string, size int64, lock bool) ([]byte, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, false, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close() // fd may be closed once mapped

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, false, fmt.Errorf("mmap %s: %w", path, err)
	}
	_ = syscall.Madvise(data, syscall.MADV_WILLNEED)

	// Touch one byte per page to force the pages resident.
	var sink byte
	for i := 0; i < len(data); i += pageSize {
		sink ^= data[i]
	}
	runtimeKeepByte(sink)

	locked := false
	if lock {
		if err := syscall.Mlock(data); err != nil {
			// Non-fatal: pages are still warm, just not pinned.
			locked = false
		} else {
			locked = true
		}
	}
	return data, locked, nil
}

// runtimeKeepByte prevents the page-faulting loop from being optimized away.
//
//go:noinline
func runtimeKeepByte(b byte) { sink = b }

var sink byte
