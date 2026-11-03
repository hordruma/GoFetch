//go:build embedded

// Package engine's in-process ("embedded") implementation links llama.cpp
// directly via cgo. Its reason to exist is the two wins a process supervisor
// (the direct / llama-swap engines) structurally cannot reach:
//
//  1. No-restart swap. The llama.cpp backend and its GPU context are initialized
//     ONCE, in NewEmbedded, and kept alive for the daemon's lifetime. A model
//     swap is llama_model_free() + llama_model_load_from_file() in place — no
//     process spawn, no context re-init, no reload of the backend .so. On the
//     measured rig the weight transfer is ~6% of a swap; this eliminates most of
//     the other ~94%.
//  2. In-process KV. Prompt-prefix reuse across requests and KV snapshots go
//     through the llama.cpp state/memory API directly, with no HTTP hop.
//
// Build: this file compiles only under `-tags embedded`, and needs the llama.cpp
// headers and a libllama to link against, supplied via CGO_CFLAGS/CGO_LDFLAGS:
//
//	CGO_CFLAGS="-I<llama.cpp headers>" \
//	CGO_LDFLAGS="-L<lib dir> -lllama -lggml -lggml-base -Wl,-rpath,<lib dir>" \
//	  go build -tags embedded ./cmd/gofetch
//
// Concurrency: one llama_context is not safe for concurrent decode, and GoFetch
// holds a single VRAM slot, so every swap and every generation is serialized on
// one mutex. That matches the design — one resident model, one prompt cache.
package engine

/*
#include <stdlib.h>
#include <string.h>
#include "llama.h"
#include "ggml-backend.h"

// gofetch_apply_template renders chat messages through the model's own chat
// template. roles[i]/contents[i] is the i-th message. Returns llama.cpp's code
// (>=0 the written length, which may exceed buflen -> grow and retry; <0 error).
static int gofetch_apply_template(const struct llama_model* model,
                                  const char** roles, const char** contents,
                                  int n, char* buf, int buflen) {
    const char* tmpl = llama_model_chat_template(model, NULL);
    if (tmpl == NULL) return -1000; // model ships no template; caller falls back
    struct llama_chat_message* msgs =
        (struct llama_chat_message*)malloc((size_t)n * sizeof(struct llama_chat_message));
    for (int i = 0; i < n; i++) { msgs[i].role = roles[i]; msgs[i].content = contents[i]; }
    int r = llama_chat_apply_template(tmpl, msgs, (size_t)n, true, buf, buflen);
    free(msgs);
    return r;
}
*/
import "C"

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/hordruma/gofetch/internal/models"
	"github.com/hordruma/gofetch/internal/warm"
)

// Embedded is the in-process llama.cpp engine.
type Embedded struct {
	reg   *models.Registry
	pool  *warm.Pool
	kvDir string

	mu     sync.Mutex
	active string
	model  *C.struct_llama_model
	ctx    *C.struct_llama_context
	vocab  *C.struct_llama_vocab
	nCtx   int
	// embedMode is whether the current context was built for pooled embeddings
	// (vs generation). Switching modes on a resident model rebuilds the context.
	embedMode bool
	// tokens currently committed to KV sequence 0, positions 0..len-1. Tracked
	// so a follow-up request can reuse the matching prefix instead of
	// re-prefilling, and so KV snapshots record their token stream.
	tokens []C.llama_token

	reqSeq atomic.Uint64
}

// NewEmbedded initializes the llama.cpp backend once and returns the engine. The
// backend .so files load from libDir (empty => rely on the loader path).
func NewEmbedded(reg *models.Registry, pool *warm.Pool, libDir, kvDir string) (Engine, error) {
	if libDir != "" {
		cdir := C.CString(libDir)
		C.ggml_backend_load_all_from_path(cdir)
		C.free(unsafe.Pointer(cdir))
	} else {
		C.ggml_backend_load_all()
	}
	C.llama_backend_init()
	slog.Info("embedded engine: llama.cpp backend initialized", "lib_dir", libDir)
	return &Embedded{reg: reg, pool: pool, kvDir: kvDir}, nil
}

// EnsureResident makes modelID the resident model (in generation mode),
// swapping in place without tearing down the backend or GPU context.
func (e *Embedded) EnsureResident(_ context.Context, modelID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.ensureLocked(modelID, false)
}

// ensureLocked makes modelID resident and its context configured for the
// requested mode (embed=false: generation/logits; embed=true: pooled
// embeddings). Switching a resident model between modes rebuilds only the
// context — the model stays loaded, so it is cheap. Caller holds e.mu.
func (e *Embedded) ensureLocked(modelID string, embed bool) error {
	m, ok := e.reg.Get(modelID)
	if !ok {
		return fmt.Errorf("unknown model %q", modelID)
	}

	// (Re)load the model only when it actually changes.
	if e.active != modelID || e.model == nil {
		e.freeLocked() // free the outgoing model; backend stays inited

		// NB: no pool.Warm here — llama.cpp mmaps and faults the GGUF itself
		// during load, so pre-faulting on the load path just reads it twice.
		cpath := C.CString(m.Path)
		mp := C.llama_model_default_params()
		mp.n_gpu_layers = C.int32_t(m.NGL)
		t0 := time.Now()
		model := C.llama_model_load_from_file(cpath, mp)
		C.free(unsafe.Pointer(cpath))
		if model == nil {
			return fmt.Errorf("load model %q: llama_model_load_from_file failed", modelID)
		}
		nctx := m.Ctx
		if nctx <= 0 {
			nctx = 4096
		}
		e.model, e.vocab, e.active, e.nCtx = model, C.llama_model_get_vocab(model), modelID, nctx
		e.ctx = nil // force a fresh context below
		slog.Info("embedded: model loaded", "model", modelID, "ms", time.Since(t0).Milliseconds())
	}

	// (Re)build the context when missing or when the mode changes.
	if e.ctx == nil || e.embedMode != embed {
		if e.ctx != nil {
			C.llama_free(e.ctx)
			e.ctx = nil
		}
		cp := C.llama_context_default_params()
		cp.n_ctx = C.uint32_t(e.nCtx)
		if embed {
			// Pooled sequence embeddings. MEAN is a sane default that also works
			// on generative models; an embedding model with its own pooling still
			// pools per sequence, which is what llama_get_embeddings_seq returns.
			cp.embeddings = C.bool(true)
			cp.pooling_type = C.LLAMA_POOLING_TYPE_MEAN
		}
		ctx := C.llama_init_from_model(e.model, cp)
		if ctx == nil {
			return fmt.Errorf("init context for %q (embed=%v) failed", modelID, embed)
		}
		e.ctx, e.embedMode, e.tokens = ctx, embed, e.tokens[:0]
	}
	return nil
}

// freeLocked releases the resident model and context. The backend stays inited —
// that is the whole point of the no-restart swap. Caller holds e.mu.
func (e *Embedded) freeLocked() {
	if e.ctx != nil {
		C.llama_free(e.ctx)
		e.ctx = nil
	}
	if e.model != nil {
		C.llama_model_free(e.model)
		e.model = nil
	}
	e.vocab, e.active, e.tokens = nil, "", e.tokens[:0]
}

// Drop frees the resident model's VRAM, keeping the backend alive.
func (e *Embedded) Drop(_ context.Context, modelID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.active == modelID {
		e.freeLocked()
	}
	return nil
}

// Active returns the resident model ID.
func (e *Embedded) Active() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.active
}

// tokenize converts text to tokens. addSpecial adds BOS etc.; parseSpecial
// interprets control tokens already present in the text (chat template output).
func (e *Embedded) tokenize(text string, addSpecial, parseSpecial bool) ([]C.llama_token, error) {
	if text == "" {
		return nil, nil
	}
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	cap := C.int32_t(len(text) + 8)
	buf := make([]C.llama_token, cap)
	n := C.llama_tokenize(e.vocab, ctext, C.int32_t(len(text)), &buf[0], cap,
		C.bool(addSpecial), C.bool(parseSpecial))
	if n < 0 { // buffer too small: -n is the required size
		cap = -n
		buf = make([]C.llama_token, cap)
		n = C.llama_tokenize(e.vocab, ctext, C.int32_t(len(text)), &buf[0], cap,
			C.bool(addSpecial), C.bool(parseSpecial))
		if n < 0 {
			return nil, fmt.Errorf("tokenize failed")
		}
	}
	return buf[:n], nil
}

func (e *Embedded) piece(tok C.llama_token) string {
	buf := make([]C.char, 256)
	n := C.llama_token_to_piece(e.vocab, tok, &buf[0], 256, 0, C.bool(false))
	if n <= 0 {
		return ""
	}
	return C.GoStringN(&buf[0], n)
}

// commonPrefix returns how many leading tokens of a match those already in KV.
func commonPrefix(a, b []C.llama_token) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// genResult reports what a generation did.
type genResult struct {
	text       string
	nPrompt    int // prompt tokens
	nCached    int // prompt tokens served from the resident KV prefix (skipped prefill)
	nGenerated int // tokens sampled
}

// generate decodes promptToks (reusing any KV prefix already resident) and
// samples up to maxNew tokens. emit, if non-nil, is called with each token's
// text as it is produced (the streaming path); returning an error from emit —
// e.g. the client disconnected — stops generation cleanly. Caller holds e.mu.
func (e *Embedded) generate(promptToks []C.llama_token, maxNew int, temp float64, seed uint32, emit func(string) error) (genResult, error) {
	res := genResult{nPrompt: len(promptToks)}
	if e.ctx == nil {
		return res, fmt.Errorf("no model resident")
	}
	if len(promptToks) >= e.nCtx {
		return res, fmt.Errorf("prompt %d tokens exceeds context %d", len(promptToks), e.nCtx)
	}

	// Reuse the matching prefix; drop KV cells past it and re-decode the rest.
	// Cap the reuse at all-but-one prompt token: the last prompt token must be
	// decoded this call so the logits sampling reads are fresh. Without this, a
	// request whose prompt is entirely cached (e.g. an identical repeat, or a
	// prompt fully covered by a restored KV snapshot) would sample from the
	// logits left by the PREVIOUS request's final decode — producing wrong or
	// empty output.
	reuse := commonPrefix(promptToks, e.tokens)
	if reuse >= len(promptToks) {
		reuse = len(promptToks) - 1
	}
	res.nCached = reuse
	mem := C.llama_get_memory(e.ctx)
	if reuse < len(e.tokens) {
		C.llama_memory_seq_rm(mem, 0, C.llama_pos(reuse), -1)
		e.tokens = e.tokens[:reuse]
	}

	// Decode the uncached prompt tail.
	if tail := promptToks[reuse:]; len(tail) > 0 {
		if C.llama_decode(e.ctx, C.llama_batch_get_one(&tail[0], C.int32_t(len(tail)))) != 0 {
			return res, fmt.Errorf("decode prompt failed")
		}
		e.tokens = append(e.tokens, tail...)
	}

	smpl := newSampler(temp, seed)
	defer C.llama_sampler_free(smpl)

	var b strings.Builder
	for res.nGenerated < maxNew && len(e.tokens) < e.nCtx {
		tok := C.llama_sampler_sample(smpl, e.ctx, -1)
		if C.llama_vocab_is_eog(e.vocab, tok) {
			break
		}
		piece := e.piece(tok)
		b.WriteString(piece)
		e.tokens = append(e.tokens, tok)
		res.nGenerated++
		if emit != nil {
			if err := emit(piece); err != nil {
				res.text = b.String()
				return res, err
			}
		}
		one := []C.llama_token{tok}
		if C.llama_decode(e.ctx, C.llama_batch_get_one(&one[0], 1)) != 0 {
			res.text = b.String()
			return res, fmt.Errorf("decode step failed")
		}
	}
	res.text = b.String()
	return res, nil
}

// newSampler builds a per-request sampler chain. temp<=0 is greedy (used by
// tests for determinism); otherwise a standard top-k/top-p/temp/dist chain.
func newSampler(temp float64, seed uint32) *C.struct_llama_sampler {
	chain := C.llama_sampler_chain_init(C.llama_sampler_chain_default_params())
	if temp <= 0 {
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_greedy())
		return chain
	}
	C.llama_sampler_chain_add(chain, C.llama_sampler_init_top_k(40))
	C.llama_sampler_chain_add(chain, C.llama_sampler_init_top_p(0.95, 1))
	C.llama_sampler_chain_add(chain, C.llama_sampler_init_temp(C.float(temp)))
	C.llama_sampler_chain_add(chain, C.llama_sampler_init_dist(C.uint32_t(seed)))
	return chain
}

// applyChatTemplate renders messages via the model's own chat template.
func (e *Embedded) applyChatTemplate(msgs []chatMessage) (string, bool) {
	n := len(msgs)
	if n == 0 {
		return "", false
	}
	roles := make([]*C.char, n)
	conts := make([]*C.char, n)
	for i, m := range msgs {
		roles[i] = C.CString(m.Role)
		conts[i] = C.CString(m.Content)
	}
	defer func() {
		for i := range roles {
			C.free(unsafe.Pointer(roles[i]))
			C.free(unsafe.Pointer(conts[i]))
		}
	}()

	buflen := C.int(8192)
	for {
		buf := make([]C.char, buflen)
		r := C.gofetch_apply_template(e.model,
			(**C.char)(unsafe.Pointer(&roles[0])), (**C.char)(unsafe.Pointer(&conts[0])),
			C.int(n), &buf[0], buflen)
		switch {
		case r == -1000:
			return "", false // no template; caller falls back
		case r < 0:
			return "", false
		case r <= buflen:
			return C.GoStringN(&buf[0], r), true
		default:
			buflen = r // grow and retry
		}
	}
}

// --- OpenAI-compatible serving ---

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type oaiRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Prompt      string        `json:"prompt"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature *float64      `json:"temperature"`
	Seed        *uint32       `json:"seed"`
	Stream      bool          `json:"stream"`
}

// ServeHTTP runs an OpenAI-compatible request in-process. It supports
// /v1/chat/completions and /v1/completions (streaming via "stream": true) and
// /v1/embeddings.
func (e *Embedded) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 8<<20))
	if err != nil {
		httpErr(w, http.StatusBadRequest, "read body: "+err.Error())
		return
	}
	if strings.Contains(r.URL.Path, "embeddings") {
		e.serveEmbeddings(w, body)
		return
	}
	var req oaiRequest
	if err := json.Unmarshal(body, &req); err != nil {
		httpErr(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	chat := strings.Contains(r.URL.Path, "chat")
	if chat && len(req.Messages) == 0 {
		httpErr(w, http.StatusBadRequest, "messages required")
		return
	}
	if !chat && req.Prompt == "" && len(req.Messages) == 0 {
		httpErr(w, http.StatusBadRequest, "prompt required")
		return
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if req.Model != "" {
		if err := e.ensureLocked(req.Model, false); err != nil {
			httpErr(w, http.StatusServiceUnavailable, err.Error())
			return
		}
	}
	if e.ctx == nil {
		httpErr(w, http.StatusServiceUnavailable, "no model resident")
		return
	}
	if e.embedMode { // resident context is in embed mode; rebuild for generation
		if err := e.ensureLocked(e.active, false); err != nil {
			httpErr(w, http.StatusServiceUnavailable, err.Error())
			return
		}
	}

	// Build the prompt: chat template for chat requests, raw prompt otherwise.
	var promptToks []C.llama_token
	if chat {
		text, ok := e.applyChatTemplate(req.Messages)
		if !ok { // model without a template: fall back to a plain join
			var sb strings.Builder
			for _, m := range req.Messages {
				sb.WriteString(m.Role)
				sb.WriteString(": ")
				sb.WriteString(m.Content)
				sb.WriteByte('\n')
			}
			sb.WriteString("assistant:")
			text = sb.String()
		}
		promptToks, err = e.tokenize(text, false, true)
	} else {
		promptToks, err = e.tokenize(req.Prompt, true, false)
	}
	if err != nil {
		httpErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	maxNew := req.MaxTokens
	if maxNew <= 0 {
		maxNew = 256
	}
	temp := 0.0
	if req.Temperature != nil {
		temp = *req.Temperature
	}
	var seed uint32 = 0xC0FFEE
	if req.Seed != nil {
		seed = *req.Seed
	}

	id := fmt.Sprintf("gofetch-%d", e.reqSeq.Add(1))

	if req.Stream {
		e.streamCompletion(w, id, req.Model, chat, promptToks, maxNew, temp, seed)
		return
	}

	res, err := e.generate(promptToks, maxNew, temp, seed, nil)
	if err != nil {
		httpErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	slog.Info("embedded: completion", "model", req.Model, "prompt", res.nPrompt,
		"cached", res.nCached, "generated", res.nGenerated)
	e.writeCompletion(w, id, req.Model, chat, res.text, res.nPrompt, res.nGenerated)
}

// streamCompletion generates and streams tokens as OpenAI SSE chunks. The
// caller holds e.mu (generation is serialized on the single context).
func (e *Embedded) streamCompletion(w http.ResponseWriter, id, model string, chat bool, promptToks []C.llama_token, maxNew int, temp float64, seed uint32) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		httpErr(w, http.StatusInternalServerError, "streaming unsupported by the server")
		return
	}
	object := "text_completion"
	if chat {
		object = "chat.completion.chunk"
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	send := func(choice map[string]any) error {
		chunk := map[string]any{"id": id, "object": object, "model": model, "choices": []any{choice}}
		b, _ := json.Marshal(chunk)
		if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}

	// Chat streams open with a role delta, then content deltas.
	if chat {
		_ = send(map[string]any{"index": 0, "delta": map[string]string{"role": "assistant"}})
	}
	emit := func(piece string) error {
		if chat {
			return send(map[string]any{"index": 0, "delta": map[string]string{"content": piece}})
		}
		return send(map[string]any{"index": 0, "text": piece})
	}

	res, err := e.generate(promptToks, maxNew, temp, seed, emit)
	if err != nil {
		// Mid-stream failures (including client disconnect) can only be signaled
		// in-band; the status line is already sent. Log and stop.
		slog.Warn("embedded: stream ended early", "model", model, "err", err,
			"generated", res.nGenerated)
		return
	}
	// Terminal chunk (finish_reason) then the OpenAI end-of-stream sentinel.
	if chat {
		_ = send(map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "stop"})
	} else {
		_ = send(map[string]any{"index": 0, "text": "", "finish_reason": "stop"})
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
	slog.Info("embedded: stream complete", "model", model, "prompt", res.nPrompt,
		"cached", res.nCached, "generated", res.nGenerated)
}

func (e *Embedded) writeCompletion(w http.ResponseWriter, id, model string, chat bool, text string, nPrompt, nGen int) {
	usage := map[string]int{
		"prompt_tokens":     nPrompt,
		"completion_tokens": nGen,
		"total_tokens":      nPrompt + nGen,
	}
	var choice map[string]any
	object := "text_completion"
	if chat {
		object = "chat.completion"
		choice = map[string]any{"index": 0, "finish_reason": "stop",
			"message": map[string]string{"role": "assistant", "content": text}}
	} else {
		choice = map[string]any{"index": 0, "finish_reason": "stop", "text": text}
	}
	writeJSONResp(w, map[string]any{
		"id": id, "object": object, "model": model,
		"choices": []any{choice}, "usage": usage,
	})
}

// --- embeddings ---

type embedRequest struct {
	Model string          `json:"model"`
	Input json.RawMessage `json:"input"` // string OR []string
}

// serveEmbeddings computes pooled embeddings for one or many inputs, OpenAI
// /v1/embeddings shape. The resident context is put in embedding mode (rebuilt
// if it was in generation mode).
func (e *Embedded) serveEmbeddings(w http.ResponseWriter, body []byte) {
	var req embedRequest
	if err := json.Unmarshal(body, &req); err != nil {
		httpErr(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	inputs, err := parseEmbedInput(req.Input)
	if err != nil {
		httpErr(w, http.StatusBadRequest, err.Error())
		return
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if req.Model != "" {
		if err := e.ensureLocked(req.Model, true); err != nil {
			httpErr(w, http.StatusServiceUnavailable, err.Error())
			return
		}
	}
	if e.ctx == nil {
		httpErr(w, http.StatusServiceUnavailable, "no model resident")
		return
	}
	if !e.embedMode {
		if err := e.ensureLocked(e.active, true); err != nil {
			httpErr(w, http.StatusServiceUnavailable, err.Error())
			return
		}
	}

	nEmbd := int(C.llama_model_n_embd(e.model))
	data := make([]any, len(inputs))
	totalToks := 0
	for i, in := range inputs {
		vec, nTok, err := e.embed(in, nEmbd)
		if err != nil {
			httpErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		totalToks += nTok
		data[i] = map[string]any{"object": "embedding", "index": i, "embedding": vec}
	}
	writeJSONResp(w, map[string]any{
		"object": "list", "model": req.Model, "data": data,
		"usage": map[string]int{"prompt_tokens": totalToks, "total_tokens": totalToks},
	})
}

// embed computes one L2-normalized pooled embedding. Caller holds e.mu and the
// context is in embed mode.
func (e *Embedded) embed(text string, nEmbd int) ([]float32, int, error) {
	toks, err := e.tokenize(text, true, false)
	if err != nil {
		return nil, 0, err
	}
	if len(toks) == 0 {
		return make([]float32, nEmbd), 0, nil
	}
	if len(toks) > e.nCtx {
		return nil, 0, fmt.Errorf("input %d tokens exceeds context %d", len(toks), e.nCtx)
	}

	// Each input is independent: clear the KV so pooling covers only this text.
	C.llama_memory_clear(C.llama_get_memory(e.ctx), C.bool(true))

	batch := C.llama_batch_get_one(&toks[0], C.int32_t(len(toks)))
	// Encoder models (BERT-style embedders) need llama_encode; decoder models
	// use llama_decode. Pick by what the model advertises.
	var rc C.int32_t
	if bool(C.llama_model_has_encoder(e.model)) {
		rc = C.llama_encode(e.ctx, batch)
	} else {
		rc = C.llama_decode(e.ctx, batch)
	}
	if rc != 0 {
		return nil, 0, fmt.Errorf("embed decode failed")
	}

	ptr := C.llama_get_embeddings_seq(e.ctx, 0)
	if ptr == nil {
		return nil, 0, fmt.Errorf("no pooled embedding (model produced none)")
	}
	raw := unsafe.Slice((*C.float)(ptr), nEmbd)
	vec := make([]float32, nEmbd)
	var norm float64
	for i := 0; i < nEmbd; i++ {
		v := float32(raw[i])
		vec[i] = v
		norm += float64(v) * float64(v)
	}
	if norm > 0 { // L2-normalize so cosine similarity is a dot product
		inv := float32(1 / math.Sqrt(norm))
		for i := range vec {
			vec[i] *= inv
		}
	}
	return vec, len(toks), nil
}

// parseEmbedInput accepts OpenAI's string-or-array "input".
func parseEmbedInput(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("input required")
	}
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		return []string{one}, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err == nil {
		if len(many) == 0 {
			return nil, fmt.Errorf("input array is empty")
		}
		return many, nil
	}
	return nil, fmt.Errorf("input must be a string or array of strings")
}

// --- in-process KV snapshots ---

// SaveKV writes the resident context's whole state (KV + its token stream) to
// name under the KV directory. name is a bare filename; the store accounts it.
func (e *Embedded) SaveKV(_ context.Context, name string) (KVStats, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.ctx == nil {
		return KVStats{}, fmt.Errorf("no model resident")
	}
	path := filepath.Join(e.kvDir, name)
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	t0 := time.Now()
	var tokPtr *C.llama_token
	if len(e.tokens) > 0 {
		tokPtr = &e.tokens[0]
	}
	// Returns bool; the size is whatever landed on the tmpfs.
	if !bool(C.llama_state_save_file(e.ctx, cpath, tokPtr, C.size_t(len(e.tokens)))) {
		return KVStats{}, fmt.Errorf("state save failed")
	}
	ms := float64(time.Since(t0).Microseconds()) / 1000
	var bytes int64
	if fi, err := osStat(path); err == nil {
		bytes = fi
	}
	return KVStats{Tokens: len(e.tokens), Bytes: bytes, Millis: ms}, nil
}

// LoadKV restores a snapshot into the resident context and adopts its token
// stream, so a following request over the same prefix skips prefill.
func (e *Embedded) LoadKV(_ context.Context, name string) (KVStats, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.ctx == nil {
		return KVStats{}, fmt.Errorf("no model resident")
	}
	path := filepath.Join(e.kvDir, name)
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	buf := make([]C.llama_token, e.nCtx)
	var nOut C.size_t
	t0 := time.Now()
	ok := C.llama_state_load_file(e.ctx, cpath, &buf[0], C.size_t(len(buf)), &nOut)
	if !bool(ok) {
		return KVStats{}, fmt.Errorf("restore state failed (model/context mismatch?)")
	}
	e.tokens = append(e.tokens[:0], buf[:nOut]...)
	return KVStats{Tokens: int(nOut), Restore: true, Millis: float64(time.Since(t0).Microseconds()) / 1000}, nil
}

func httpErr(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func writeJSONResp(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

// osStat returns a file's size, isolated so the cgo file needs only one os import.
func osStat(path string) (int64, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	return fi.Size(), nil
}
