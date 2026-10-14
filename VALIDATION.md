# Execution validation

The experiment was rerun with `pymilvus.model` 0.3.4 against Milvus 3.0.2 and PyMilvus 3.0.2. Run ID: `20261014T071236Z`.

## Results

- Ingested and backfilled all 5,183 SciFact documents; evaluated all 300 BEIR test queries.
- Passed all 16 workflow checks, including data preservation, both retrieval versions, cutover, rollback, dual-vector updates, field/index deletion, and writes after cleanup.
- Completed 2,812 continuous HTTP searches with 0 recorded failures, plus 30 mixed-version requests.
- Independently recomputed Recall@10 and nDCG@10 against official relevance judgments.
- Confirmed that the complete per-query B evaluation results are identical before and after old-field cleanup.
- Executed all 10 Python snippets from `ARTICLE.md` on a separate collection with 64 real SciFact documents and 20 queries, using a separate fresh environment installed from the published CPU setup commands and lock file.
- Verified output dimensions, FP32 precision, normalized vectors, a 512-token maximum input length, and exactly one application of the query instruction for both models.

## Reproducibility

The run used original BAAI model weights at pinned Hugging Face revisions. File hashes, pooling settings, library versions, and encoding settings are recorded in [`recorded-results/model-provenance.json`](recorded-results/model-provenance.json). Runtime output directories are created automatically, including on a fresh checkout.

## Scope

This is a single-process, low-concurrency validation against an existing Milvus instance. It does not establish production availability, cluster-failure tolerance, or request-draining guarantees. All current metrics and charts describe this run. Earlier experiment records remain available in Git history.

Machine-readable verification: [`recorded-results/reverification.json`](recorded-results/reverification.json).

Validated runtime script SHA-256:

```text
be312ec619749e9ab7b2d3b4696f565a9ddc5e75febbbf25ed400ac52e0a2bfa
```
