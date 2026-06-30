.PHONY: build run test vet tidy probe build-cuda

build:
	go build -o gofetch ./cmd/gofetch

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
