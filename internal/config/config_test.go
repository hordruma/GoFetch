package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadDefaults(t *testing.T) {
	c, err := Load(writeConfig(t, `{"engine":{"type":"direct","llama_server_bin":"llama-server"}}`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if c.Listen != ":8080" {
		t.Errorf("listen = %q, want the default :8080", c.Listen)
	}
}

func TestLoadDefaultEngineType(t *testing.T) {
	c, err := Load(writeConfig(t, `{"engine":{"upstream":"http://127.0.0.1:8081"}}`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if c.Engine.Type != "llama-swap" {
		t.Errorf("engine.type = %q, want the default llama-swap", c.Engine.Type)
	}
}

func TestLoadRejectsInvalid(t *testing.T) {
	cases := []struct {
		name, body, want string
	}{
		{"unknown engine", `{"engine":{"type":"bogus"}}`, "unknown engine.type"},
		{"direct without binary", `{"engine":{"type":"direct"}}`, "llama_server_bin required"},
		{"llama-swap without upstream", `{"engine":{"type":"llama-swap"}}`, "upstream required"},
		{"malformed JSON", `{`, "parse config"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Load(writeConfig(t, tc.body))
			if err == nil {
				t.Fatal("expected an error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("err = %v, want it to mention %q", err, tc.want)
			}
		})
	}
}

func TestLoadMissingFile(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "absent.json")); err == nil {
		t.Fatal("expected an error for a missing config file")
	}
}

func TestLoadModels(t *testing.T) {
	c, err := Load(writeConfig(t, `{
		"listen": "127.0.0.1:9000",
		"engine": {"type":"direct","llama_server_bin":"llama-server"},
		"ram": {"budget_bytes": 100, "locked_budget_bytes": 50},
		"kv": {"dir":"/dev/shm/x","budget_bytes": 10},
		"models": [{"id":"m","path":"/m.gguf","ngl":99,"ctx":4096,"tier":"locked"}]
	}`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(c.Models) != 1 || c.Models[0].ID != "m" || c.Models[0].Ctx != 4096 || c.Models[0].Tier != "locked" {
		t.Errorf("models = %+v, want the configured model parsed in full", c.Models)
	}
	if c.RAM.LockedBudgetBytes != 50 || c.KV.BudgetBytes != 10 {
		t.Errorf("budgets = %+v / %+v, want them parsed", c.RAM, c.KV)
	}
}
