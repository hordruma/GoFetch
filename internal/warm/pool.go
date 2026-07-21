// Package warm manages the RAM warm-pool: GGUF files held hot in the OS page
// cache so a subsequent model load reaches VRAM at RAM speed instead of disk
// speed. It supports two tiers:
//
//   - TierCache:  mmap + page-fault. Cheap, evictable, holds many models. The
//     weights are resident in the page cache but remain pageable, so a CUDA
//     upload from them takes the driver's pageable→staging→VRAM path.
//   - TierLocked: mmap + page-fault + mlock. Page-locked and non-evictable
//     (separate, smaller budget). This is the FOUNDATION for the fast DMA path:
//     an in-process CUDA engine can cudaHostRegister a locked region and then
//     DMA it to VRAM at the PCIe ceiling. mlock alone does not give fast DMA —
//     it guarantees residency and removes page-fault stalls.
//
// Linux only (mmap/mlock/madvise via the syscall package, stdlib only).
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

// Tier selects how aggressively a file is held in RAM.
type Tier int

const (
	TierCache  Tier = iota // page-cache resident, evictable
	TierLocked             // page-locked (mlock), non-evictable, separate budget
)

func (t Tier) String() string {
	if t == TierLocked {
		return "locked"
	}
	return "cache"
}

// ParseTier maps a config/request string to a Tier. Empty defaults to cache.
func ParseTier(s string) (Tier, error) {
	switch s {
	case "", "cache":
		return TierCache, nil
	case "locked", "pinned":
		return TierLocked, nil
	default:
		return TierCache, fmt.Errorf("unknown warm tier %q (want cache|locked)", s)
	}
}

type entry struct {
	path string
	data []byte
	size int64
	tier Tier
	last time.Time
}

// Pool is a concurrency-safe set of warmed files. Each tier has its own byte
// budget and its own LRU set, so promoting a hot model to the locked tier never
// evicts the large page-cache tier and vice-versa.
type Pool struct {
	mu      sync.Mutex
	entries map[string]*entry
	used    [2]int64 // indexed by Tier
	budget  [2]int64 // indexed by Tier
	now     func() time.Time
}

// New constructs a warm-pool. cacheBudget bounds TierCache; lockedBudget bounds
// TierLocked. A budget of 0 means unbounded for that tier.
func New(cacheBudget, lockedBudget int64) *Pool {
	p := &Pool{entries: make(map[string]*entry), now: time.Now}
	p.budget[TierCache] = cacheBudget
	p.budget[TierLocked] = lockedBudget
	return p
}

// Info is a status snapshot of one warmed file.
type Info struct {
	Path string    `json:"path"`
	Size int64     `json:"size"`
	Tier string    `json:"tier"`
	Last time.Time `json:"last"`
}

// Status reports per-tier usage and the warmed files, newest first.
type Status struct {
	CacheUsed   int64  `json:"cache_used"`
	CacheBudget int64  `json:"cache_budget"`
	LockUsed    int64  `json:"locked_used"`
	LockBudget  int64  `json:"locked_budget"`
	Files       []Info `json:"files"`
}

// Warm pulls a file into RAM at the given tier, evicting LRU entries of that
// same tier first if needed. Re-warming at the same tier refreshes LRU; warming
// at a different tier promotes/demotes the file.
func (p *Pool) Warm(path string, tier Tier) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// A tier change re-maps the file. The old mapping is held until the new one
	// is fully established so that a rejected promotion (over budget, mlock
	// refused) leaves the file warm at its original tier rather than dropping it.
	old, promoting := p.entries[path]
	if promoting && old.tier == tier {
		old.last = p.now()
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
	if b := p.budget[tier]; b > 0 && size > b {
		return fmt.Errorf("file %s (%d bytes) exceeds %s budget (%d)", path, size, tier, b)
	}
	if err := p.evictToFitLocked(tier, size); err != nil {
		return err
	}

	data, locked, err := mapAndFault(path, size, tier == TierLocked)
	if err != nil {
		return err
	}
	if tier == TierLocked && !locked {
		_ = syscall.Munmap(data)
		return fmt.Errorf("mlock %s failed (raise RLIMIT_MEMLOCK or lower locked budget)", path)
	}
	if promoting { // new mapping is live; the old tier's copy can go
		if err := p.releaseLocked(old); err != nil {
			_ = syscall.Munmap(data)
			return err
		}
	}
	p.entries[path] = &entry{path: path, data: data, size: size, tier: tier, last: p.now()}
	p.used[tier] += size
	return nil
}

// Evict releases a warmed file regardless of tier. No-op if not warm.
func (p *Pool) Evict(path string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	e, ok := p.entries[path]
	if !ok {
		return nil
	}
	return p.releaseLocked(e)
}

// Snapshot returns the current pool status.
func (p *Pool) Snapshot() Status {
	p.mu.Lock()
	defer p.mu.Unlock()
	files := make([]Info, 0, len(p.entries))
	for _, e := range p.entries {
		files = append(files, Info{Path: e.path, Size: e.size, Tier: e.tier.String(), Last: e.last})
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Last.After(files[j].Last) })
	return Status{
		CacheUsed:   p.used[TierCache],
		CacheBudget: p.budget[TierCache],
		LockUsed:    p.used[TierLocked],
		LockBudget:  p.budget[TierLocked],
		Files:       files,
	}
}

// evictToFitLocked frees LRU entries of the given tier until need bytes fit.
func (p *Pool) evictToFitLocked(tier Tier, need int64) error {
	budget := p.budget[tier]
	if budget <= 0 {
		return nil
	}
	for p.used[tier]+need > budget {
		victim := p.lruLocked(tier)
		if victim == nil {
			return fmt.Errorf("cannot free %d bytes under %s budget %d", need, tier, budget)
		}
		if err := p.releaseLocked(victim); err != nil {
			return err
		}
	}
	return nil
}

func (p *Pool) lruLocked(tier Tier) *entry {
	var oldest *entry
	for _, e := range p.entries {
		if e.tier != tier {
			continue
		}
		if oldest == nil || e.last.Before(oldest.last) {
			oldest = e
		}
	}
	return oldest
}

func (p *Pool) releaseLocked(e *entry) error {
	if e.tier == TierLocked {
		_ = syscall.Munlock(e.data)
	}
	err := syscall.Munmap(e.data)
	delete(p.entries, e.path)
	p.used[e.tier] -= e.size
	if err != nil {
		return fmt.Errorf("munmap %s: %w", e.path, err)
	}
	return nil
}

// mapAndFault mmaps the file read-only, advises WILLNEED, touches every page to
// fault it into the page cache, and optionally mlocks it.
func mapAndFault(path string, size int64, lock bool) (data []byte, locked bool, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, false, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close() // fd may close once mapped

	data, err = syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, false, fmt.Errorf("mmap %s: %w", path, err)
	}
	_ = syscall.Madvise(data, syscall.MADV_WILLNEED)

	// Touch one byte per page to force the pages resident.
	var s byte
	for i := 0; i < len(data); i += pageSize {
		s ^= data[i]
	}
	sink = s

	if lock {
		if err := syscall.Mlock(data); err == nil {
			locked = true
		}
	}
	return data, locked, nil
}

// sink defeats dead-code elimination of the page-faulting loop.
var sink byte
