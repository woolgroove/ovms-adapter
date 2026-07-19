package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"time"

	"github.com/daulet/tokenizers"
)

var (
	OVMSBaseURL   = "http://127.0.0.1:9001"
	ModelName     = "bge-base-zh-int8"
	APIModelID    = "bge-base-zh-int8"
	MaxLength     = 512
	MaxBatch      = 8
	ListenAddr    = "0.0.0.0:8000"
	TokenizerPath = `E:\R\AI\ovms\models\bge-base-zh-int8\1\tokenizer.json`
	Device        = "NPU"
)

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

var tk *tokenizers.Tokenizer

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
		for s := 0; s < seq; s++ {
			if s >= len(mask[b]) || mask[b][s] == 0 {
				continue
			}
			count++
			for d := 0; d < dim; d++ {
				result[b][d] += hidden[b][s][d]
			}
		}
		if count > 0 {
			for d := 0; d < dim; d++ {
				result[b][d] /= count
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

	resp, err := http.Post(url, "application/json", bytes.NewReader(body))
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
	var result map[string]interface{}
	if err := json.Unmarshal(respBody, &result); err != nil {
		snippet := string(respBody)
		if len(snippet) > 512 {
			snippet = snippet[:512] + "...(truncated)"
		}
		return nil, fmt.Errorf("OVMS response not JSON: %v; body=%s", err, snippet)
	}

	// TFS `inputs`-style 请求，返回是 {"outputs": {"<name>": [[[...]]]}}。
	// 若模型只有单输出，OVMS 有时会直接返回 {"outputs": [[[...]]]}（三维数组）。
	// 两种都兼容：先尝试按 map 取 last_hidden_state，失败再按裸数组解析。
	rawOutputs, ok := result["outputs"]
	if !ok {
		snippet := string(respBody)
		if len(snippet) > 512 {
			snippet = snippet[:512] + "...(truncated)"
		}
		return nil, fmt.Errorf("cannot parse OVMS response (no outputs key); body=%s", snippet)
	}

	var hidden3d []interface{}
	switch v := rawOutputs.(type) {
	case map[string]interface{}:
		named, ok := v["last_hidden_state"]
		if !ok {
			// 若模型输出名不同，取第一个 value
			for _, val := range v {
				named = val
				break
			}
		}
		arr, ok := named.([]interface{})
		if !ok {
			return nil, fmt.Errorf("outputs.last_hidden_state is not an array")
		}
		hidden3d = arr
	case []interface{}:
		hidden3d = v
	default:
		return nil, fmt.Errorf("outputs is neither map nor array (%T)", v)
	}

	return parseNestedHidden(hidden3d)
}

// parseNestedHidden 把 [][][]float64（JSON any 表示）转成 [batch][seq][dim]float32。
func parseNestedHidden(raw []interface{}) ([][][]float32, error) {
	batch := len(raw)
	result := make([][][]float32, batch)
	for b := 0; b < batch; b++ {
		seqArr, ok := raw[b].([]interface{})
		if !ok {
			return nil, fmt.Errorf("hidden[%d] is not array", b)
		}
		seq := len(seqArr)
		result[b] = make([][]float32, seq)
		for s := 0; s < seq; s++ {
			dimArr, ok := seqArr[s].([]interface{})
			if !ok {
				return nil, fmt.Errorf("hidden[%d][%d] is not array", b, s)
			}
			dim := len(dimArr)
			result[b][s] = make([]float32, dim)
			for d := 0; d < dim; d++ {
				f, ok := dimArr[d].(float64)
				if !ok {
					return nil, fmt.Errorf("hidden[%d][%d][%d] is not number", b, s, d)
				}
				result[b][s][d] = float32(f)
			}
		}
	}
	return result, nil
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
	B := MaxBatch
	if N > B {
		http.Error(w, fmt.Sprintf(`{"error":"max batch %d"}`, B), http.StatusBadRequest)
		return
	}

	t0 := time.Now()

	// 模型编译期固化 shape=[MaxBatch, MaxLength]，输入必须补齐到该维度。
	// 逐条编码
	encoded := make([][]uint32, N)
	for i, text := range texts {
		ids, _, err := tk.EncodeErr(text, true)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":"tokenize failed: %s"}`, err.Error()), 500)
			return
		}
		if len(ids) > MaxLength {
			ids = ids[:MaxLength]
		}
		encoded[i] = ids
	}

	// 全部按固定 [B, MaxLength] 打包；未填充位置 mask=0，pooling 时会跳过。
	inputIDs := make([][]int64, B)
	attentionMask := make([][]int64, B)
	tokenTypeIDs := make([][]int64, B)
	for i := 0; i < B; i++ {
		inputIDs[i] = make([]int64, MaxLength)
		attentionMask[i] = make([]int64, MaxLength)
		tokenTypeIDs[i] = make([]int64, MaxLength)
		if i < N {
			for j := 0; j < len(encoded[i]); j++ {
				inputIDs[i][j] = int64(encoded[i][j])
				attentionMask[i][j] = 1
			}
		}
	}

	lastHidden, err := callOVMS(inputIDs, attentionMask, tokenTypeIDs)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err.Error()), 500)
		return
	}

	pooled := meanPooling(lastHidden[:N], attentionMask[:N])
	embeddings := l2Normalize(pooled)

	data := make([]EmbeddingData, N)
	totalTokens := 0
	for i, emb := range embeddings {
		// 真实 token 数 = attention_mask 里的 1 的个数
		for _, m := range attentionMask[i] {
			if m == 1 {
				totalTokens++
			}
		}
		data[i] = EmbeddingData{Object: "embedding", Embedding: emb, Index: i}
	}

	elapsed := time.Since(t0).Milliseconds()
	log.Printf("batch=%d/%d tokens=%d time=%dms", N, B, totalTokens, elapsed)

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
	log.Println("Tokenizer ready.")

	http.HandleFunc("/v1/embeddings", handleEmbeddings)
	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/v1/models", handleModels)

	log.Printf("Server on %s", ListenAddr)
	log.Fatal(http.ListenAndServe(ListenAddr, nil))
}
