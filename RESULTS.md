# Recorded Experiment Results

Run: 20261001T022531Z
Collection: scifact_switch_20261001t022531z
Duration: 1457.56 seconds, including model inference, evaluation, and validation.

| Model | Recall@10 | nDCG@10 |
|---|---:|---:|
| BGE-small-en-v1.5 · 384 dimensions | 0.8362 | 0.7127 |
| BGE-base-en-v1.5 · 768 dimensions | 0.8742 | 0.7427 |

BGE-base-en-v1.5 improves Recall@10 by 3.80 percentage points over BGE-small-en-v1.5 in this evaluation. Results use the complete SciFact corpus with fixed preprocessing and index parameters. Production rollout still requires canary evaluation of retrieval quality and latency.

## Validation checks

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

## Continuous HTTP search, including model inference

| Phase | Requests | Failures | P95 ms |
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

Latency includes query embedding inference, queueing, HTTP communication, and Milvus search, rather than Milvus search alone. Backfill and online requests share CPU inference resources. This run uses four inference workers with two threads per session.

Production deployments need request accounting and draining to confirm that old-version requests have completed, plus distributed configuration rollout to coordinate version changes across service instances.
