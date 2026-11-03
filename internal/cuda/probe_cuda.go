//go:build cuda

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart

#include <cuda_runtime.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    double pageable_gbps,     pageable_peak_gbps;
    double pinned_reg_gbps,   pinned_reg_peak_gbps;
    double pinned_alloc_gbps, pinned_alloc_peak_gbps;
    double pinned_chunked_gbps;
    double register_ms, host_alloc_ms, fill_ms, malloc_ms, free_ms;
    int    reg_ok;
    int    err;        // non-zero => fatal cudaError_t
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

// bench_h2d times iters host->device copies of n bytes (in `chunk`-sized pieces
// if chunk>0, else one monolithic copy). cudaMemcpy on the default stream is
// synchronous, so wall-clock timing is exact. Returns mean and peak GB/s via
// out params (peak = fastest single run = truest hardware ceiling).
static void bench_h2d(void* dev, void* src, size_t n, int iters, size_t chunk,
                      double* mean_gbps, double* peak_gbps) {
    double total = 0.0, best = 1e30;
    for (int it = 0; it < iters; it++) {
        double t0 = now_s();
        if (chunk == 0 || chunk >= n) {
            cudaMemcpy(dev, src, n, cudaMemcpyHostToDevice);
        } else {
            for (size_t off = 0; off < n; off += chunk) {
                size_t len = (off + chunk <= n) ? chunk : (n - off);
                cudaMemcpy((char*)dev + off, (char*)src + off, len, cudaMemcpyHostToDevice);
            }
        }
        cudaDeviceSynchronize();
        double dt = now_s() - t0;
        total += dt;
        if (dt < best) best = dt;
    }
    *mean_gbps = (double)n / (total / iters) / 1e9;
    *peak_gbps = (double)n / best / 1e9;
}

int gofetch_dma_probe(void* host, size_t n, int iters, size_t chunk, dma_result* out) {
    memset(out, 0, sizeof(*out));
    if (iters < 1) iters = 1;
    if (chunk == 0) chunk = 8 * 1024 * 1024;
    void* dev = NULL;
    double t0, t1;
    cudaError_t e;

    cudaFree(0); // force lazy context init so it doesn't contaminate timings

    t0 = now_s(); e = cudaMalloc(&dev, n); t1 = now_s();
    if (e != cudaSuccess) { seterr(out, e); return out->err; }
    out->malloc_ms = (t1 - t0) * 1e3;

    // Warmup transfer (discarded): wakes the PCIe link from any low-power state.
    cudaMemcpy(dev, host, n, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // (1) pageable, monolithic
    bench_h2d(dev, host, n, iters, 0, &out->pageable_gbps, &out->pageable_peak_gbps);

    // (2) register-in-place -> pinned, monolithic. cudaHostRegisterReadOnly
    // suits read-only weights and is what the engine would use; the driver
    // accepts it only on a writable VMA, which is why Probe maps the file
    // copy-on-write (see the mmap in probe_cuda.go's Go side).
    t0 = now_s(); e = cudaHostRegister(host, n, cudaHostRegisterReadOnly); t1 = now_s();
    if (e != cudaSuccess) { cudaGetLastError(); t0 = now_s(); e = cudaHostRegister(host, n, cudaHostRegisterDefault); t1 = now_s(); }
    if (e == cudaSuccess) {
        out->reg_ok = 1;
        out->register_ms = (t1 - t0) * 1e3;
        bench_h2d(dev, host, n, iters, 0, &out->pinned_reg_gbps, &out->pinned_reg_peak_gbps);
        cudaHostUnregister(host);
    } else {
        strncpy(out->errmsg, cudaGetErrorString(e), 255); // non-fatal
        cudaGetLastError();
    }

    // (3) dedicated pinned buffer + (4) chunked, both off the same allocation
    void* pinned = NULL;
    t0 = now_s(); e = cudaMallocHost(&pinned, n); t1 = now_s();
    if (e == cudaSuccess) {
        out->host_alloc_ms = (t1 - t0) * 1e3;
        t0 = now_s(); memcpy(pinned, host, n); t1 = now_s();
        out->fill_ms = (t1 - t0) * 1e3; // RAM->RAM cost paid every load by this path
        bench_h2d(dev, pinned, n, iters, 0, &out->pinned_alloc_gbps, &out->pinned_alloc_peak_gbps);
        double dummy;
        bench_h2d(dev, pinned, n, iters, chunk, &out->pinned_chunked_gbps, &dummy);
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
// and measures the H2D path iters times per variant. chunkBytes sets the chunk
// size for the representative chunked variant (0 => 8 MiB default).
func Probe(path string, iters int, maxBytes, chunkBytes int64) (*Result, error) {
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

	// Map copy-on-write writable, mirroring warm.TierLocked: this is the mapping
	// cudaHostRegister accepts (a read-only or MAP_SHARED file mapping is refused
	// with "invalid argument"), so the register-in-place variant measures the
	// real warm-tier path rather than always reporting "unavailable".
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size),
		syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE)
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
	C.gofetch_dma_probe(unsafe.Pointer(&data[0]), C.size_t(len(data)), C.int(iters), C.size_t(chunkBytes), &res)
	if res.err != 0 {
		return nil, fmt.Errorf("cuda: %s", C.GoString(&res.errmsg[0]))
	}

	chunk := chunkBytes
	if chunk == 0 {
		chunk = 8 * 1024 * 1024
	}
	return &Result{
		Bytes:               int64(len(data)),
		ChunkBytes:          chunk,
		PageableGBps:        float64(res.pageable_gbps),
		PageablePeakGBps:    float64(res.pageable_peak_gbps),
		PinnedRegGBps:       float64(res.pinned_reg_gbps),
		PinnedRegPeakGBps:   float64(res.pinned_reg_peak_gbps),
		RegisterMS:          float64(res.register_ms),
		RegOK:               res.reg_ok != 0,
		PinnedAllocGBps:     float64(res.pinned_alloc_gbps),
		PinnedAllocPeakGBps: float64(res.pinned_alloc_peak_gbps),
		HostAllocMS:         float64(res.host_alloc_ms),
		FillMS:              float64(res.fill_ms),
		PinnedChunkedGBps:   float64(res.pinned_chunked_gbps),
		MallocMS:            float64(res.malloc_ms),
		FreeMS:              float64(res.free_ms),
	}, nil
}

var sink byte
