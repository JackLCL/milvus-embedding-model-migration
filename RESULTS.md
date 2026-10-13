# 实验结果

运行：20261001T022531Z
Collection：scifact_switch_20261001t022531z
用时：1457.56 秒（含完整模型推理、评测及验证）

| 版本 | Recall@10 | nDCG@10 |
|---|---:|---:|
| BGE-small-en-v1.5 · 384 维 | 0.8362 | 0.7127 |
| BGE-base-en-v1.5 · 768 维 | 0.8742 | 0.7427 |

在本次评测中，BGE-base-en-v1.5 的 Recall@10 比 BGE-small-en-v1.5 高 3.80 个百分点。结果基于完整 SciFact 语料、固定文本处理方式和索引参数；上线前仍需通过灰度流量评估检索质量与延迟。

## 检查

- server_is_3_0_2: {"passed": true}
- full_scifact_test: {"passed": true, "documents": 5183, "queries": 300}
- old_and_new_dimensions: {"passed": true}
- NULL_B_rows_excluded: {"passed": true}
- all_documents_backfilled: {"passed": true}
- all_texts_and_vectors_audited: {"passed": true, "rows": 5183, "max_error": {"BGE-small-en-v1.5": 0.0, "BGE-base-en-v1.5": 0.0}}
- wrong_dimension_rejected: {"passed": true, "message": "<MilvusException: (code=2000, message=failed to search: failed to search/query delegator 1 for channel by-dev-rootcoord-dml_13_469271701364567204v0: fail to search on QueryNode 1: worker(1) query failed: parser searchRequest failed:  => vector dimension mismatch, expected vector size(byte) 3072, actual 1536. at ../internal/core/src/query/Plan.cpp:183\n: segcore error[segcoreCode=2001])>"}
- canary_routing: {"passed": true, "B_requests": 6}
- A_latest_update_searchable: {"passed": true}
- B_latest_update_searchable: {"passed": true}
- probe_deleted: {"passed": true}
- old_field_and_index_removed: {"passed": true}
- B_only_write: {"passed": true}
- final_count: {"passed": true}
- cleanup_preserves_retrieval: {"passed": true}
- continuous_real_inference_HTTP: {"passed": true, "requests": 3754, "failures": 0}

## 持续 HTTP 检索（包含模型推理）

| 阶段 | 请求数 | 失败 | P95 ms |
|---|---:|---:|---:|
| baseline_A | 19 | 0 | 26.14 |
| add_B | 23 | 0 | 28.28 |
| backfill_B_from_milvus_text | 3533 | 0 | 50.94 |
| audit_and_evaluate_B | 101 | 0 | 34.99 |
| canary_AB | 4 | 0 | 30.92 |
| cutover_B | 17 | 0 | 62.25 |
| rollback_A | 18 | 0 | 31.66 |
| final_B | 20 | 0 | 71.17 |
| after_cleanup | 19 | 0 | 546.18 |

延迟包含原文查询的模型推理、排队、HTTP 和 Milvus 检索，不是单独的 Milvus 延迟。两个模型的回填与在线请求共享 CPU 推理资源。本次采用 4 个推理 worker，每个 session 2 线程。

生产部署需采用请求计数与排空机制，确认旧版本的在途请求已完成，并通过分布式配置发布机制协调各服务实例的版本切换。
