.PHONY: build run test vet tidy

build:
	go build -o gofetch ./cmd/gofetch

run: build
	./gofetch -config config.json

test:
	go test ./...

vet:
	go vet ./...

tidy:
	go mod tidy
