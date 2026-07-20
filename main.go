package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"sync"
	"time"

	grpc_client "ovms-adapter/grpc-client"

	"github.com/daulet/tokenizers"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var (
	OVMSGRPCAddr  = "127.0.0.1:9000"
	ModelName     = "bge-base-zh-int8"
	APIModelID    = "bge-base-zh-int8"
	MaxLength     = 512
	MaxBatch      = 1 // 模型固化 shape=[1,512]，单条推理；多条请求并发发起
	MaxParallel   = 4 // 适配器全局最大并发，应 <= config.json 里的 nireq
	ListenAddr    = "0.0.0.0:8000"
	TokenizerPath = `E:\R\AI\ovms\models\bge-base-zh-int8\1\tokenizer.json`
	Device        = "NPU"
)

var (
	tk         *tokenizers.Tokenizer
	grpcConn   *grpc.ClientConn
	grpcClient grpc_client.GRPCInferenceServiceClient
	inferSem   chan struct{}
)

// zeroTypeIDs 是所有请求共享的 token_type_ids（BGE 单句场景永远全 0）。
// 只读，goroutine 安全；protobuf 序列化不会修改它。
var zeroTypeIDs []int64

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

func meanPoolRaw(raw []byte, mask []int64, seq, dim int) ([]float32, error) {
	want := seq * dim * 4
	if len(raw) != want {
		return nil, fmt.Errorf("unexpected OVMS output size: got %d, want %d", len(raw), want)
	}
	result := make([]float32, dim)
	count := 0
	for s := 0; s < seq && s < len(mask); s++ {
		if mask[s] == 0 {
			break
		}
		count++
		row := raw[s*dim*4:]
		for d := 0; d < dim; d++ {
			bits := binary.LittleEndian.Uint32(row[d*4 : d*4+4])
			result[d] += math.Float32frombits(bits)
		}
	}
	if count == 0 {
		return result, nil
	}
	inv := float32(1) / float32(count)
	for d := range result {
		result[d] *= inv
	}
	return result, nil
}

func callOVMS(parent context.Context, inputIDs, attentionMask, tokenTypeIDs [][]int64) ([]float32, error) {
	if len(inputIDs) != MaxBatch {
		return nil, fmt.Errorf("invalid batch: got %d, want %d", len(inputIDs), MaxBatch)
	}
	flat := func(values [][]int64) []int64 {
		result := make([]int64, 0, MaxBatch*MaxLength)
		for _, row := range values {
			result = append(result, row...)
		}
		return result
	}

	request := &grpc_client.ModelInferRequest{
		ModelName: ModelName,
		Inputs: []*grpc_client.ModelInferRequest_InferInputTensor{
			{
				Name: "input_ids", Datatype: "INT64", Shape: []int64{MaxBatch, MaxLength},
				Contents: &grpc_client.InferTensorContents{Int64Contents: flat(inputIDs)},
			},
			{
				Name: "attention_mask", Datatype: "INT64", Shape: []int64{MaxBatch, MaxLength},
				Contents: &grpc_client.InferTensorContents{Int64Contents: flat(attentionMask)},
			},
			{
				Name: "token_type_ids", Datatype: "INT64", Shape: []int64{MaxBatch, MaxLength},
				Contents: &grpc_client.InferTensorContents{Int64Contents: flat(tokenTypeIDs)},
			},
		},
	}

	ctx, cancel := context.WithTimeout(parent, 30*time.Second)
	defer cancel()
	response, err := grpcClient.ModelInfer(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("OVMS gRPC call failed: %v", err)
	}
	if len(response.Outputs) == 0 {
		return nil, fmt.Errorf("OVMS gRPC response has no outputs")
	}
	output := response.Outputs[0]
	if output.Name != "last_hidden_state" {
		return nil, fmt.Errorf("unexpected OVMS output: %s", output.Name)
	}
	if len(output.Shape) != 3 || output.Shape[0] != int64(MaxBatch) || output.Shape[1] != int64(MaxLength) {
		return nil, fmt.Errorf("unexpected OVMS output shape: %v", output.Shape)
	}
	dim := int(output.Shape[2])
	if len(response.RawOutputContents) > 0 {
		return meanPoolRaw(response.RawOutputContents[0], attentionMask[0], MaxLength, dim)
	}
	if output.Contents != nil && len(output.Contents.Fp32Contents) > 0 {
		values := output.Contents.Fp32Contents
		if len(values) != MaxBatch*MaxLength*dim {
			return nil, fmt.Errorf("unexpected FP32 output count: %d", len(values))
		}
		result := make([]float32, dim)
		count := 0
		for s := 0; s < MaxLength && attentionMask[0][s] != 0; s++ {
			count++
			for d := 0; d < dim; d++ {
				result[d] += values[s*dim+d]
			}
		}
		for d := range result {
			result[d] /= float32(count)
		}
		return result, nil
	}
	return nil, fmt.Errorf("OVMS gRPC response has no raw or FP32 output contents")
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

	var wg sync.WaitGroup
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(idx int, text string) {
			defer wg.Done()

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

			select {
			case inferSem <- struct{}{}:
			case <-r.Context().Done():
				errs[idx] = r.Context().Err()
				return
			}
			pooled, err := callOVMS(r.Context(), inputIDs, attentionMask, tokenTypeIDs)
			<-inferSem
			if err != nil {
				errs[idx] = err
				return
			}

			emb := l2Normalize([][]float32{pooled})[0]
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

func modelReady(ctx context.Context) error {
	response, err := grpcClient.ModelReady(ctx, &grpc_client.ModelReadyRequest{Name: ModelName})
	if err != nil {
		return err
	}
	if !response.Ready {
		return fmt.Errorf("model %s is not ready", ModelName)
	}
	return nil
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	if err := modelReady(ctx); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "unavailable",
			"device": Device,
			"model":  APIModelID,
			"error":  err.Error(),
		})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "ok",
		"device":  Device,
		"model":   APIModelID,
		"backend": "grpc",
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
	inferSem = make(chan struct{}, MaxParallel)
	grpcConn, err = grpc.Dial(OVMSGRPCAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect OVMS gRPC at %s: %v", OVMSGRPCAddr, err)
	}
	defer grpcConn.Close()
	grpcClient = grpc_client.NewGRPCInferenceServiceClient(grpcConn)
	readyCtx, readyCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer readyCancel()
	if err := modelReady(readyCtx); err != nil {
		log.Fatalf("OVMS gRPC model readiness failed: %v", err)
	}
	log.Printf("Tokenizer ready; OVMS gRPC=%s; parallel=%d", OVMSGRPCAddr, MaxParallel)

	http.HandleFunc("/v1/embeddings", handleEmbeddings)
	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/v1/models", handleModels)

	log.Printf("Server on %s", ListenAddr)
	log.Fatal(http.ListenAndServe(ListenAddr, nil))
}
