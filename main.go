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

type OVMSRequest struct {
	Inputs []OVMSTensor `json:"inputs"`
}

type OVMSTensor struct {
	Name     string  `json:"name"`
	Shape    []int   `json:"shape"`
	Datatype string  `json:"datatype"`
	Data     []int64 `json:"data"`
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
	batch := len(inputIDs)
	if batch == 0 {
		return nil, fmt.Errorf("batch is 0")
	}
	seq := len(inputIDs[0])

	flatInput := make([]int64, 0, batch*seq)
	flatMask := make([]int64, 0, batch*seq)
	flatType := make([]int64, 0, batch*seq)
	for b := 0; b < batch; b++ {
		flatInput = append(flatInput, inputIDs[b]...)
		flatMask = append(flatMask, attentionMask[b]...)
		flatType = append(flatType, tokenTypeIDs[b]...)
	}

	payload := OVMSRequest{
		Inputs: []OVMSTensor{
			{Name: "input_ids", Shape: []int{batch, seq}, Datatype: "INT64", Data: flatInput},
			{Name: "attention_mask", Shape: []int{batch, seq}, Datatype: "INT64", Data: flatMask},
			{Name: "token_type_ids", Shape: []int{batch, seq}, Datatype: "INT64", Data: flatType},
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
	var result map[string]interface{}
	json.Unmarshal(respBody, &result)

	outputs, ok := result["outputs"].([]interface{})
	if !ok || len(outputs) == 0 {
		outputs, ok = result["predictions"].([]interface{})
		if !ok || len(outputs) == 0 {
			return nil, fmt.Errorf("cannot parse OVMS response")
		}
	}

	out0 := outputs[0].(map[string]interface{})
	dataRaw := out0["data"].([]interface{})

	shapeRaw, ok := out0["shape"].([]interface{})
	if !ok {
		return parseFlatData(dataRaw, batch, 1, 768)
	}

	shape := make([]int, len(shapeRaw))
	for i, v := range shapeRaw {
		shape[i] = int(v.(float64))
	}

	if len(shape) == 3 {
		return parseFlatData(dataRaw, shape[0], shape[1], shape[2])
	}
	return parseFlatData(dataRaw, batch, 1, 768)
}

func parseFlatData(dataRaw []interface{}, batch, seq, dim int) ([][][]float32, error) {
	result := make([][][]float32, batch)
	for b := 0; b < batch; b++ {
		result[b] = make([][]float32, seq)
		for s := 0; s < seq; s++ {
			result[b][s] = make([]float32, dim)
			for d := 0; d < dim; d++ {
				idx := b*seq*dim + s*dim + d
				if idx >= len(dataRaw) {
					continue
				}
				result[b][s][d] = float32(dataRaw[idx].(float64))
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

	// 逐条编码
	encoded := make([][]uint32, N)
	maxSeq := 0
	for i, text := range texts {
		// 使用 EncodeErr 获取错误返回值
		ids, _, err := tk.EncodeErr(text, true)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":"tokenize failed: %s"}`, err.Error()), 500)
			return
		}
		if len(ids) > MaxLength {
			ids = ids[:MaxLength]
		}
		encoded[i] = ids
		if len(ids) > maxSeq {
			maxSeq = len(ids)
		}
	}
	if maxSeq == 0 {
		maxSeq = 1
	}

	inputIDs := make([][]int64, N)
	attentionMask := make([][]int64, N)
	tokenTypeIDs := make([][]int64, N)

	for i, ids := range encoded {
		inputIDs[i] = make([]int64, maxSeq)
		attentionMask[i] = make([]int64, maxSeq)
		tokenTypeIDs[i] = make([]int64, maxSeq)
		for j := 0; j < len(ids); j++ {
			inputIDs[i][j] = int64(ids[j])
			attentionMask[i][j] = 1
		}
	}

	padCount := B - N
	if padCount > 0 {
		for p := 0; p < padCount; p++ {
			padIDs := make([]int64, maxSeq)
			padMask := make([]int64, maxSeq)
			padType := make([]int64, maxSeq)
			copy(padIDs, inputIDs[0])
			inputIDs = append(inputIDs, padIDs)
			attentionMask = append(attentionMask, padMask)
			tokenTypeIDs = append(tokenTypeIDs, padType)
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
		totalTokens += len(inputIDs[i])
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
