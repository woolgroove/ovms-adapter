package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/daulet/tokenizers"
)

var (
	OVMSBaseURL   = "http://127.0.0.1:9001"
	ModelName     = "bge-base-zh-int8"
	APIModelID    = "bge-base-zh-int8"
	MaxLength     = 512
	MaxBatch      = 1 // 模型固化 shape=[1,512]，单条推理；多条请求并发发起
	MaxParallel   = 4 // 并发调用 OVMS 的最大 goroutine 数，应 <= config.json 里的 nireq
	ListenAddr    = "0.0.0.0:8000"
	TokenizerPath = `E:\R\AI\ovms\models\bge-base-zh-int8\1\tokenizer.json`
	Device        = "NPU"
)

// httpClient 复用 TCP 连接，避免每次 request 都握手；并发场景下必开。
var httpClient = &http.Client{
	Timeout: 30 * time.Second,
	Transport: &http.Transport{
		MaxIdleConns:        64,
		MaxIdleConnsPerHost: 64,
		IdleConnTimeout:     90 * time.Second,
	},
}

type EmbeddingRequest struct {
	Input interface{} `json:"input"`
	Model string      `json:"model,omitempty"`
}

type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

type Usage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  Usage           `json:"usage"`
}

// TFS columnar inputs: {"inputs": {"<name>": [[...]] , ...}}
// 每个字段值是 [batch][seq] int64 二维数组，OVMS 不再在外面加 batch 维。
type OVMSRequest struct {
	Inputs map[string][][]int64 `json:"inputs"`
}

// OVMS TFS 响应：单输出场景就是 map[输出名] -> [batch][seq][dim] float32。
// 静态类型 unmarshal 避免 map[string]interface{} 的 39 万次装箱。
type ovmsResponse struct {
	Outputs map[string][][][]float32 `json:"outputs"`
}

var tk *tokenizers.Tokenizer

// zeroTypeIDs 是所有请求共享的 token_type_ids（BGE 单句场景永远全 0）。
// 只读，goroutine 安全；JSON 序列化不会修改它。
var zeroTypeIDs []int64

func meanPooling(hidden [][][]float32, mask [][]int64) [][]float32 {
	batch := len(hidden)
	if batch == 0 {
		return nil
	}
	seq := len(hidden[0])
	dim := len(hidden[0][0])
	result := make([][]float32, batch)
	for b := 0; b < batch; b++ {
		result[b] = make([]float32, dim)
		var count float32 = 0
		mb := mask[b]
		hb := hidden[b]
		// mask 布局固定为前 K 个 1 后面全 0，遇到 0 直接跳出，避免扫完 512。
		for s := 0; s < seq; s++ {
			if s >= len(mb) || mb[s] == 0 {
				break
			}
			count++
			hs := hb[s]
			for d := 0; d < dim; d++ {
				result[b][d] += hs[d]
			}
		}
		if count > 0 {
			inv := 1.0 / count
			for d := 0; d < dim; d++ {
				result[b][d] *= inv
			}
		}
	}
	return result
}

func l2Normalize(vectors [][]float32) [][]float64 {
	result := make([][]float64, len(vectors))
	for i, vec := range vectors {
		var norm float64 = 0
		result[i] = make([]float64, len(vec))
		for j, v := range vec {
			result[i][j] = float64(v)
			norm += result[i][j] * result[i][j]
		}
		norm = math.Sqrt(norm)
		if norm > 1e-12 {
			for j := range result[i] {
				result[i][j] /= norm
			}
		}
	}
	return result
}

func callOVMS(inputIDs, attentionMask, tokenTypeIDs [][]int64) ([][][]float32, error) {
	if len(inputIDs) == 0 {
		return nil, fmt.Errorf("batch is 0")
	}

	payload := OVMSRequest{
		Inputs: map[string][][]int64{
			"input_ids":      inputIDs,
			"attention_mask": attentionMask,
			"token_type_ids": tokenTypeIDs,
		},
	}

	body, _ := json.Marshal(payload)
	url := fmt.Sprintf("%s/v1/models/%s:predict", OVMSBaseURL, ModelName)

	resp, err := httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("OVMS call failed: %v", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		snippet := string(respBody)
		if len(snippet) > 512 {
			snippet = snippet[:512] + "...(truncated)"
		}
		return nil, fmt.Errorf("OVMS %s returned HTTP %d: %s", url, resp.StatusCode, snippet)
	}

	var parsed ovmsResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		snippet := string(respBody)
		if len(snippet) > 512 {
			snippet = snippet[:512] + "...(truncated)"
		}
		return nil, fmt.Errorf("OVMS response not JSON: %v; body=%s", err, snippet)
	}
	if len(parsed.Outputs) == 0 {
		return nil, fmt.Errorf("OVMS response has empty outputs")
	}
	// 单输出模型：优先取 last_hidden_state，兜底取第一个键。
	if h, ok := parsed.Outputs["last_hidden_state"]; ok {
		return h, nil
	}
	for _, h := range parsed.Outputs {
		return h, nil
	}
	return nil, fmt.Errorf("unreachable")
}

func handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid request"}`, http.StatusBadRequest)
		return
	}

	var texts []string
	switch v := req.Input.(type) {
	case string:
		texts = []string{v}
	case []interface{}:
		for _, s := range v {
			if str, ok := s.(string); ok {
				texts = append(texts, str)
			}
		}
	}

	if len(texts) == 0 {
		http.Error(w, `{"error":"input empty"}`, http.StatusBadRequest)
		return
	}

	N := len(texts)
	t0 := time.Now()

	// 模型固化 shape=[MaxBatch, MaxLength]，MaxBatch=1；多条时并发，最多 MaxParallel。
	data := make([]EmbeddingData, N)
	tokenCounts := make([]int, N)
	errs := make([]error, N)

	sem := make(chan struct{}, MaxParallel)
	var wg sync.WaitGroup
	for i := 0; i < N; i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, text string) {
			defer wg.Done()
			defer func() { <-sem }()

			ids, _, err := tk.EncodeErr(text, true)
			if err != nil {
				errs[idx] = fmt.Errorf("tokenize failed: %v", err)
				return
			}
			if len(ids) > MaxLength {
				ids = ids[:MaxLength]
			}

			inputIDs := [][]int64{make([]int64, MaxLength)}
			attentionMask := [][]int64{make([]int64, MaxLength)}
			tokenTypeIDs := [][]int64{zeroTypeIDs} // 只读共享，节省 512×int64 分配
			for j, id := range ids {
				inputIDs[0][j] = int64(id)
				attentionMask[0][j] = 1
			}
			tokenCounts[idx] = len(ids)

			lastHidden, err := callOVMS(inputIDs, attentionMask, tokenTypeIDs)
			if err != nil {
				errs[idx] = err
				return
			}

			pooled := meanPooling(lastHidden, attentionMask)
			emb := l2Normalize(pooled)[0]
			data[idx] = EmbeddingData{Object: "embedding", Embedding: emb, Index: idx}
		}(i, texts[i])
	}
	wg.Wait()

	for i, e := range errs {
		if e != nil {
			http.Error(w, fmt.Sprintf(`{"error":"item %d: %s"}`, i, e.Error()), 500)
			return
		}
	}

	totalTokens := 0
	for _, c := range tokenCounts {
		totalTokens += c
	}
	elapsed := time.Since(t0).Milliseconds()
	log.Printf("N=%d tokens=%d time=%dms parallel=%d", N, totalTokens, elapsed, MaxParallel)

	resp := EmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  APIModelID,
		Usage:  Usage{PromptTokens: totalTokens, TotalTokens: totalTokens},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "ok",
		"device": Device,
		"model":  APIModelID,
	})
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data": []map[string]interface{}{
			{"id": APIModelID, "object": "model", "owned_by": "local-ovms"},
		},
	})
}

func main() {
	log.Println("Loading tokenizer ...")
	var err error
	tk, err = tokenizers.FromFile(TokenizerPath)
	if err != nil {
		log.Fatalf("Failed: %v", err)
	}
	defer tk.Close()
	zeroTypeIDs = make([]int64, MaxLength)
	log.Println("Tokenizer ready.")

	http.HandleFunc("/v1/embeddings", handleEmbeddings)
	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/v1/models", handleModels)

	log.Printf("Server on %s", ListenAddr)
	log.Fatal(http.ListenAndServe(ListenAddr, nil))
}
