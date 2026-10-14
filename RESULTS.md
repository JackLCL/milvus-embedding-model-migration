# Recorded Experiment Results

Run: `20261014T071236Z`

Collection: `scifact_switch_20261014t071236z`

Duration: 1141.72 seconds, including model loading, inference, evaluation, and validation.

Inference: `pymilvus.model` 0.3.4, Sentence Transformers 6.1.0, PyTorch 2.14.1+cpu; original BAAI weights, FP32 on CPU.

| Model | Recall@10 | nDCG@10 |
| --- | ---: | ---: |
| BGE-small-en-v1.5, 384 dimensions | 0.8362 | 0.7127 |
| BGE-base-en-v1.5, 768 dimensions | 0.8742 | 0.7404 |

Recall@10 changes by +3.80 percentage points and nDCG@10 by +0.0277. Results use the complete SciFact corpus with fixed preprocessing and index parameters. Production rollout still requires canary evaluation of quality and latency.

## Validation checks

- `server_is_3_0_2`: {"passed": true}
- `full_scifact_test`: {"passed": true, "documents": 5183, "queries": 300}
- `old_and_new_dimensions`: {"passed": true}
- `NULL_B_rows_excluded`: {"passed": true}
- `all_documents_backfilled`: {"passed": true}
- `all_texts_and_vectors_audited`: {"passed": true, "rows": 5183, "max_error": {"A": 0.0, "B": 0.0}}
- `wrong_dimension_rejected`: {"passed": true, "message": "<MilvusException: (code=2000, message=failed to search: failed to search/query delegator 1 for channel by-dev-rootcoord-dml_2_469271701374605664v0: fail to search on QueryNode 1: worker(1) query failed: parser searchRequest failed:  => vector dimension mismatch, expected vector size(byte) 3072, actual 1536. at ../internal/core/src/query/Plan.cpp:183\n: segcore error[segcoreCode=2001])>"}
- `canary_routing`: {"passed": true, "B_requests": 6}
- `A_latest_update_searchable`: {"passed": true}
- `B_latest_update_searchable`: {"passed": true}
- `probe_deleted`: {"passed": true}
- `old_field_and_index_removed`: {"passed": true}
- `B_only_write`: {"passed": true}
- `final_count`: {"passed": true}
- `cleanup_preserves_retrieval`: {"passed": true}
- `continuous_real_inference_HTTP`: {"passed": true, "requests": 2812, "failures": 0}

## Continuous HTTP search, including model inference

| Phase | Requests | Failures | P95 ms |
| --- | ---: | ---: | ---: |
| baseline_A | 18 | 0 | 43.81 |
| add_B | 31 | 0 | 44.26 |
| backfill_B_from_milvus_text | 2600 | 0 | 78.57 |
| audit_and_evaluate_B | 91 | 0 | 103.72 |
| canary_AB | 5 | 0 | 78.79 |
| cutover_B | 16 | 0 | 147.91 |
| rollback_A | 16 | 0 | 51.71 |
| final_B | 18 | 0 | 115.46 |
| after_cleanup | 17 | 0 | 707.49 |

Latency includes embedding inference, queueing, HTTP communication, and Milvus search. Backfill and online queries share CPU inference resources. The run uses four inference workers with PyTorch configured for two CPU threads per operation.

Production deployments need explicit request draining and coordinated configuration rollout. The experiment uses a fixed observation interval under low concurrency.
