// Package cuda holds the cgo bridge to the CUDA runtime. It is only functional
// in a `-tags cuda` build; the default build compiles a stub that errors with a
// rebuild hint. The first thing it provides is a DMA bandwidth probe — measure
// the host->device transfer path on real hardware before building the engine on
// top of assumptions about it.
package cuda

// Result reports the host->device (H2D) transfer characteristics measured by
// Probe. Bandwidths are in decimal GB/s; durations in milliseconds.
type Result struct {
	Bytes int64 `json:"bytes"` // transfer size used for the measurement

	// PageableGBps is H2D from ordinary pageable memory (what we get today with
	// no surgery: driver stages through an internal pinned buffer + CPU memcpy).
	PageableGBps float64 `json:"pageable_gbps"`

	// PinnedRegGBps is H2D after cudaHostRegister on the mmap'd region in place
	// (no extra copy). RegOK reports whether registering a file-backed read-only
	// mapping was permitted on this driver; RegisterMS is the one-time cost.
	PinnedRegGBps float64 `json:"pinned_reg_gbps"`
	RegisterMS    float64 `json:"register_ms"`
	RegOK         bool    `json:"reg_ok"`

	// PinnedAllocGBps is H2D from a dedicated cudaMallocHost buffer the weights
	// were copied into (the upload ceiling, at the cost of one RAM->RAM copy).
	// HostAllocMS is the cudaMallocHost cost.
	PinnedAllocGBps float64 `json:"pinned_alloc_gbps"`
	HostAllocMS     float64 `json:"host_alloc_ms"`

	// VRAM allocator costs — relevant to swap latency, since a naive swap does
	// cudaFree(old)+cudaMalloc(new) each time.
	MallocMS float64 `json:"malloc_ms"`
	FreeMS   float64 `json:"free_ms"`
}
