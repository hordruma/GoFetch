// Command gofetch is the mechanical memory-plane daemon: it loads, warms, drops,
// and swaps local models behind an HTTP control plane, and owns a tmpfs-backed
// KV snapshot store. Policy is the caller's; this process executes mechanics.
package main

import (
	"flag"
	"log/slog"
	"os"

	"github.com/hordruma/gofetch/internal/api"
	"github.com/hordruma/gofetch/internal/config"
	"github.com/hordruma/gofetch/internal/engine"
	"github.com/hordruma/gofetch/internal/kvcache"
	"github.com/hordruma/gofetch/internal/models"
	"github.com/hordruma/gofetch/internal/warm"
)

func main() {
	cfgPath := flag.String("config", "config.json", "path to JSON config file")
	flag.Parse()

	cfg, err := config.Load(*cfgPath)
	if err != nil {
		slog.Error("load config", "err", err)
		os.Exit(1)
	}

	reg := models.NewRegistry(cfg.Models)
	pool := warm.New(cfg.RAM.BudgetBytes, cfg.RAM.LockedBudgetBytes)

	// The KV store owns the snapshot directory, so it is created before the
	// engine that must be told to write snapshots into it.
	kv, err := kvcache.New(cfg.KV.Dir, cfg.KV.BudgetBytes)
	if err != nil {
		slog.Error("init kv store", "err", err)
		os.Exit(1)
	}

	var eng engine.Engine
	switch cfg.Engine.Type {
	case "llama-swap":
		eng, err = engine.NewLlamaSwap(cfg.Engine.Upstream)
	case "direct":
		eng, err = engine.NewDirect(cfg.Engine.LlamaServerBin, reg, kv.Dir())
	case "embedded":
		eng, err = engine.NewEmbedded(reg, pool)
	default:
		slog.Error("unknown engine type", "type", cfg.Engine.Type)
		os.Exit(1)
	}
	if err != nil {
		slog.Error("init engine", "err", err)
		os.Exit(1)
	}

	srv := api.New(cfg, reg, eng, pool, kv)
	slog.Info("gofetch listening", "addr", cfg.Listen, "engine", cfg.Engine.Type, "models", len(cfg.Models))
	if err := srv.ListenAndServe(); err != nil {
		slog.Error("server stopped", "err", err)
		os.Exit(1)
	}
}
