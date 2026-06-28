// Package cuda holds the cgo bridge to the CUDA runtime. It is only functional
// in a `-tags cuda` build; the default build compiles a stub that errors with a
// rebuild hint. Its first job is a DMA bandwidth probe — measure the
// host->device transport on real hardware before building the engine on
// assumptions about it.
//
// Scope: this is a TRANSPORT microbenchmark ("how fast is the pipe"), not a
// swap-time predictor. A real swap also rebuilds the context/graph and allocates
// KV, which the probe does not measure.
package cuda

// Result reports the host->device (H2D) transfer characteristics measured by
// Probe. Bandwidths are decimal GB/s; durations are milliseconds. Each variant
// reports both the mean over iters and the peak (fastest single run) — peak is
// the truer hardware ceiling; a mean well below peak signals throttling or PCIe
// power-management (ASPM) noise.
type Result struct {
	Bytes      int64 `json:"bytes"`       // transfer size used
	ChunkBytes int64 `json:"chunk_bytes"` // chunk size for the chunked variant

	// (1) pageable: H2D straight from the mmap'd, still-pageable region — what
	// we get today (driver stages through a pinned buffer + CPU memcpy).
	PageableGBps     float64 `json:"pageable_gbps"`
	PageablePeakGBps float64 `json:"pageable_peak_gbps"`

	// (2) pinned register-in-place: cudaHostRegister on the mmap region, no extra
	// copy. RegOK reports whether the driver allowed registering a file-backed
	// read-only mapping; RegisterMS is the one-time cost paid at warm.
	PinnedRegGBps     float64 `json:"pinned_reg_gbps"`
	PinnedRegPeakGBps float64 `json:"pinned_reg_peak_gbps"`
	RegisterMS        float64 `json:"register_ms"`
	RegOK             bool    `json:"reg_ok"`

	// (3) pinned alloc-and-copy: a dedicated cudaMallocHost buffer (the upload
	// ceiling). HostAllocMS is the alloc cost; FillMS is the RAM->RAM copy paid
	// to fill it on every load (the cost register-in-place avoids).
	PinnedAllocGBps     float64 `json:"pinned_alloc_gbps"`
	PinnedAllocPeakGBps float64 `json:"pinned_alloc_peak_gbps"`
	HostAllocMS         float64 `json:"host_alloc_ms"`
	FillMS              float64 `json:"fill_ms"`

	// (4) pinned chunked: H2D split into ChunkBytes pieces, approximating the
	// real loader's many tensor-sized copies — more representative than the
	// monolithic numbers above (which are an optimistic upper bound).
	PinnedChunkedGBps float64 `json:"pinned_chunked_gbps"`

	// VRAM allocator costs — relevant to swap latency, since a naive swap does
	// cudaFree(old)+cudaMalloc(new) each time.
	MallocMS float64 `json:"malloc_ms"`
	FreeMS   float64 `json:"free_ms"`
}
