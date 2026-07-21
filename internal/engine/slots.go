package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// kvSlot is the llama.cpp server slot GoFetch drives. The backend is started
// with a single slot (`-np 1`) so that the slot holding a document prefix is
// unambiguous: GoFetch's model is one resident model with one prompt cache, and
// a snapshot addressed to "the slot" must mean exactly one thing.
const kvSlot = 0

// slotResponse is llama.cpp's reply to a slot save/restore action.
type slotResponse struct {
	IDSlot    int    `json:"id_slot"`
	Filename  string `json:"filename"`
	NSaved    int    `json:"n_saved"`
	NRestored int    `json:"n_restored"`
	NWritten  int64  `json:"n_written"`
	NRead     int64  `json:"n_read"`
	Timings   struct {
		SaveMS    float64 `json:"save_ms"`
		RestoreMS float64 `json:"restore_ms"`
	} `json:"timings"`
	Error json.RawMessage `json:"error"`
}

// slotAction performs a llama.cpp slot save/restore against a server base URL.
// llama.cpp takes the filename in the request body and resolves it beneath its
// own --slot-save-path, refusing anything that looks like a path traversal, so
// name must be a bare filename.
func slotAction(ctx context.Context, client *http.Client, base, action, name string) (KVStats, error) {
	if name == "" {
		return KVStats{}, fmt.Errorf("snapshot name required")
	}
	body, _ := json.Marshal(map[string]string{"filename": name})
	url := fmt.Sprintf("%s/slots/%d?action=%s", base, kvSlot, action)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return KVStats{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return KVStats{}, fmt.Errorf("slot %s: %w", action, err)
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
	if resp.StatusCode != http.StatusOK {
		return KVStats{}, fmt.Errorf("slot %s: upstream %s: %s", action, resp.Status, bytes.TrimSpace(raw))
	}
	var sr slotResponse
	if err := json.Unmarshal(raw, &sr); err != nil {
		return KVStats{}, fmt.Errorf("slot %s: decode reply: %w", action, err)
	}
	if len(sr.Error) > 0 {
		return KVStats{}, fmt.Errorf("slot %s: %s", action, sr.Error)
	}
	if action == "restore" {
		return KVStats{Tokens: sr.NRestored, Bytes: sr.NRead, Millis: sr.Timings.RestoreMS, Restore: true}, nil
	}
	return KVStats{Tokens: sr.NSaved, Bytes: sr.NWritten, Millis: sr.Timings.SaveMS}, nil
}
