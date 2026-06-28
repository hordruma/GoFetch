// Package config loads GoFetch configuration from a JSON file. Stdlib only so
// the daemon builds offline with no module downloads.
package config

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/hordruma/gofetch/internal/models"
)

// Config is the top-level daemon configuration.
type Config struct {
	Listen string         `json:"listen"`
	Engine EngineConfig   `json:"engine"`
	RAM    RAMConfig      `json:"ram"`
	KV     KVConfig       `json:"kv"`
	Models []models.Model `json:"models"`
}

// EngineConfig selects and configures the inference engine.
type EngineConfig struct {
	Type           string `json:"type"`             // "llama-swap" | "direct" | "embedded"
	Upstream       string `json:"upstream"`         // llama-swap base URL
	LlamaServerBin string `json:"llama_server_bin"` // direct: llama-server binary
}

// RAMConfig bounds the two-tier warm-pool.
type RAMConfig struct {
	BudgetBytes       int64 `json:"budget_bytes"`        // TierCache budget (page-cache warm)
	LockedBudgetBytes int64 `json:"locked_budget_bytes"` // TierLocked budget (mlock-pinned)
}

// KVConfig bounds the KV snapshot store.
type KVConfig struct {
	Dir         string `json:"dir"`          // tmpfs directory for KV snapshots
	BudgetBytes int64  `json:"budget_bytes"` // max bytes of KV snapshots retained
}

// Load reads and validates a config file.
func Load(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	if c.Listen == "" {
		c.Listen = ":8080"
	}
	if c.Engine.Type == "" {
		c.Engine.Type = "llama-swap"
	}
	if err := c.validate(); err != nil {
		return nil, err
	}
	return &c, nil
}

func (c *Config) validate() error {
	switch c.Engine.Type {
	case "llama-swap":
		if c.Engine.Upstream == "" {
			return fmt.Errorf("engine.upstream required for llama-swap engine")
		}
	case "direct":
		if c.Engine.LlamaServerBin == "" {
			return fmt.Errorf("engine.llama_server_bin required for direct engine")
		}
	case "embedded":
		// In-process CUDA engine; requires a -tags cuda build (validated at init).
	default:
		return fmt.Errorf("unknown engine.type %q", c.Engine.Type)
	}
	return nil
}
