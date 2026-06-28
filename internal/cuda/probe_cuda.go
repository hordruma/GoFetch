//go:build cuda

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart

#include <cuda_runtime.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

typedef struct {
    double pageable_gbps;
    double pinned_reg_gbps;
    double pinned_alloc_gbps;
    double register_ms;
    double host_alloc_ms;
    double malloc_ms;
    double free_ms;
    int    reg_ok;
    int    err;       // non-zero => fatal cudaError_t
    char   errmsg[256];
} dma_result;

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static void seterr(dma_result* o, cudaError_t e) {
    o->err = (int)e;
    strncpy(o->errmsg, cudaGetErrorString(e), 255);
}

// gofetch_dma_probe measures the H2D path for an n-byte host buffer. cudaMemcpy
// on the default stream is synchronous, so wall-clock timing around it is exact.
// Host-side allocator calls (cudaMalloc/cudaHostRegister/...) are likewise
// synchronous, so CPU timing is correct throughout (no CUDA events needed).
int gofetch_dma_probe(void* host, size_t n, int iters, dma_result* out) {
    memset(out, 0, sizeof(*out));
    if (iters < 1) iters = 1;
    void* dev = NULL;
    double t0, t1;
    cudaError_t e;

    t0 = now_s(); e = cudaMalloc(&dev, n); t1 = now_s();
    if (e != cudaSuccess) { seterr(out, e); return out->err; }
    out->malloc_ms = (t1 - t0) * 1e3;

    // (1) pageable: upload straight from the mmap'd, still-pageable region.
    t0 = now_s();
    for (int i = 0; i < iters; i++) {
        e = cudaMemcpy(dev, host, n, cudaMemcpyHostToDevice);
        if (e != cudaSuccess) { seterr(out, e); cudaFree(dev); return out->err; }
    }
    t1 = now_s();
    out->pageable_gbps = (double)n * iters / (t1 - t0) / 1e9;

    // (2) register-in-place: page-lock the existing mapping, no extra copy.
    // cudaHostRegisterDefault for broad compatibility; cudaHostRegisterReadOnly
    // (CUDA >= 11.1) can lower overhead for read-only sources.
    t0 = now_s(); e = cudaHostRegister(host, n, cudaHostRegisterDefault); t1 = now_s();
    if (e == cudaSuccess) {
        out->reg_ok = 1;
        out->register_ms = (t1 - t0) * 1e3;
        t0 = now_s();
        for (int i = 0; i < iters; i++) cudaMemcpy(dev, host, n, cudaMemcpyHostToDevice);
        t1 = now_s();
        out->pinned_reg_gbps = (double)n * iters / (t1 - t0) / 1e9;
        cudaHostUnregister(host);
    } else {
        // Non-fatal: some drivers refuse to register file-backed mappings.
        strncpy(out->errmsg, cudaGetErrorString(e), 255);
        cudaGetLastError();
    }

    // (3) dedicated pinned buffer + copy-in: the upload ceiling, paying one
    // host->host memcpy to fill it.
    void* pinned = NULL;
    t0 = now_s(); e = cudaMallocHost(&pinned, n); t1 = now_s();
    if (e == cudaSuccess) {
        out->host_alloc_ms = (t1 - t0) * 1e3;
        memcpy(pinned, host, n);
        t0 = now_s();
        for (int i = 0; i < iters; i++) cudaMemcpy(dev, pinned, n, cudaMemcpyHostToDevice);
        t1 = now_s();
        out->pinned_alloc_gbps = (double)n * iters / (t1 - t0) / 1e9;
        cudaFreeHost(pinned);
    }

    t0 = now_s(); cudaFree(dev); t1 = now_s();
    out->free_ms = (t1 - t0) * 1e3;
    return 0;
}
*/
import "C"

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

// Probe mmaps the file at path (capped to maxBytes if > 0), faults it resident,
// and measures the host->device transfer path iters times per variant.
func Probe(path string, iters int, maxBytes int64) (*Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	size := fi.Size()
	if size <= 0 {
		f.Close()
		return nil, fmt.Errorf("empty file %s", path)
	}
	if maxBytes > 0 && size > maxBytes {
		size = maxBytes
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	f.Close()
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}
	defer syscall.Munmap(data)

	// Fault every page in so we measure transfer, not page-fault latency.
	var s byte
	for i := 0; i < len(data); i += 4096 {
		s ^= data[i]
	}
	sink = s

	var res C.dma_result
	C.gofetch_dma_probe(unsafe.Pointer(&data[0]), C.size_t(len(data)), C.int(iters), &res)
	if res.err != 0 {
		return nil, fmt.Errorf("cuda: %s", C.GoString(&res.errmsg[0]))
	}

	return &Result{
		Bytes:           int64(len(data)),
		PageableGBps:    float64(res.pageable_gbps),
		PinnedRegGBps:   float64(res.pinned_reg_gbps),
		RegisterMS:      float64(res.register_ms),
		RegOK:           res.reg_ok != 0,
		PinnedAllocGBps: float64(res.pinned_alloc_gbps),
		HostAllocMS:     float64(res.host_alloc_ms),
		MallocMS:        float64(res.malloc_ms),
		FreeMS:          float64(res.free_ms),
	}, nil
}

var sink byte
