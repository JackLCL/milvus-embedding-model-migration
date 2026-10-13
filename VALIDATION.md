# Execution validation

The published experiment was revalidated on a fresh public checkout against Milvus 3.0.2 with PyMilvus 3.0.2. Run ID: `20261013T092650Z`.

## Results

- Ingested and backfilled all 5,183 SciFact documents; evaluated all 300 BEIR test queries.
- Passed all 16 workflow checks, including backfill integrity, both retrieval versions, cutover, rollback, updates during the rollback window, field/index deletion, and writes after cleanup.
- Completed 3,798 continuous HTTP searches, including model inference, with 0 recorded failures, plus 30 mixed-version requests.
- Independently recomputed Recall@10 and nDCG@10 against the official relevance judgments.
- Executed all 10 Python snippets from `ARTICLE.md` in sequence on a separate collection using 64 real SciFact documents and 20 queries. The added index-readiness loop completed successfully.
- Installed a separate fresh Python environment from `requirements.lock.txt` and verified dependency versions and imports.

The full run started without an `artifacts/` directory and downloaded the dataset. It reused ONNX model files whose SHA-256 hashes matched the original run. Its Python environment matched every pinned dependency; the fresh-environment installation was verified separately.

## First-run initialization

The script now creates parent directories for its run outputs. This fixes a startup failure when a new checkout has no `artifacts/` directory. The environment setup commands and runtime workflow are otherwise unchanged.

## Scope

This is a single-process, low-concurrency validation against an existing Milvus instance. It does not establish production availability, cluster-failure tolerance, or request-draining guarantees. The original measurements in `RESULTS.md` remain the original experiment's results; this verification is recorded separately in [`recorded-results/reverification.json`](recorded-results/reverification.json).

Validated runtime script SHA-256:

```text
fe2c97c520be07e2a52d1d40073fca4f9232116d6dc70ad50288c29ccf300973
```
