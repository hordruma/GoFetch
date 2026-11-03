.PHONY: build run test vet tidy probe build-cuda build-embedded

build:
	go build -o gofetch ./cmd/gofetch

# In-process llama.cpp engine (engine.type "embedded"). Point these at the
# llama.cpp headers and a libllama build (any backend: CUDA, Vulkan, ROCm, CPU).
# LLAMA_HDR: dir with llama.h + ggml headers. LLAMA_LIB: dir with libllama.so.
# See docs/verification.md for a worked example (prebuilt release + fetched headers).
LLAMA_HDR ?= /usr/local/include
LLAMA_LIB ?= /usr/local/lib
build-embedded:
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I$(LLAMA_HDR)" \
	CGO_LDFLAGS="-L$(LLAMA_LIB) -lllama -lggml -lggml-base -Wl,-rpath,$(LLAMA_LIB)" \
	go build -tags embedded -o gofetch ./cmd/gofetch

# Requires the CUDA toolkit. Override include/lib via CGO_CFLAGS / CGO_LDFLAGS
# if CUDA is not at /usr/local/cuda. See docs/dma-and-swap.md.
probe:
	CGO_ENABLED=1 go build -tags cuda -o gofetch-probe ./cmd/gofetch-probe

build-cuda:
	CGO_ENABLED=1 go build -tags cuda -o gofetch ./cmd/gofetch

run: build
	./gofetch -config config.json

test:
	go test ./...

vet:
	go vet ./...

tidy:
	go mod tidy
