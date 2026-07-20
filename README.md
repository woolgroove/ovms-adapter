# ovms-adapter

Windows CGO Go 适配器：将 OpenAI 兼容 `/v1/embeddings` 请求转换为 OVMS KServe v2 gRPC 推理请求。

- 对外：HTTP `0.0.0.0:8010`
- 对内：OVMS gRPC `127.0.0.1:9000`
- 模型固定输入：`[1, 512]`
- 全局最大推理并发：`4`（与 OVMS `nireq` 对齐）
- 输出：在二进制 FP32 张量上直接 Mean Pooling + L2 归一化

构建走 GitHub Actions（`.github/workflows/ovms-adapter.yml`），手动触发 `workflow_dispatch`。
