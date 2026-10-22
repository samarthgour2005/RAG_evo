# Reranker comparison: ColBERT vs cross-encoder

Measured over **53 labelled questions**, top-5 returned from 20 first-stage candidates, 3 timed pass(es) per configuration after warm-up. Quality comes from the first pass; latency pools every pass.

Regenerate with `python scripts/compare_rerankers.py`. This copy is from the
[Reranker benchmark run](https://github.com/samarthgour2005/RAG_evo/actions/runs/39847841262)
on commit a94dbe3, with ColBERT passage embeddings cached.

## Quality and latency

| Retriever | Reranker | P@5 | Recall@5 | MRR | Hit rate | Rerank p50 | Rerank p95 | Retrieval p50 | Retrieval p95 |
|---|---|---|---|---|---|---|---|---|---|
| dense | colbert | 0.200 | 0.972 | 0.916 | 0.981 | 66.5 ms | 68.3 ms | 85.4 ms | 88.4 ms |
| dense | cross-encoder | 0.204 | 0.981 | 0.914 | 0.981 | 206.7 ms | 350.6 ms | 225.3 ms | 369.8 ms |
| dense | none | 0.200 | 0.962 | 0.878 | 0.962 | — | — | 18.8 ms | 21.2 ms |
| hybrid | colbert | 0.192 | 0.934 | 0.901 | 0.943 | 65.8 ms | 67.1 ms | 85.2 ms | 87.5 ms |
| hybrid | cross-encoder | 0.192 | 0.934 | 0.888 | 0.943 | 54.6 ms | 67.0 ms | 74.9 ms | 91.4 ms |
| hybrid | none | 0.192 | 0.934 | 0.847 | 0.943 | — | — | 19.7 ms | 23.0 ms |
| sparse | colbert | 0.185 | 0.906 | 0.887 | 0.925 | 67.0 ms | 68.6 ms | 67.6 ms | 69.3 ms |
| sparse | cross-encoder | 0.189 | 0.925 | 0.888 | 0.943 | 220.3 ms | 254.0 ms | 221.0 ms | 255.0 ms |
| sparse | none | 0.166 | 0.811 | 0.715 | 0.830 | — | — | 0.3 ms | 0.4 ms |

Setup time (model load, plus ColBERT's one-off corpus encoding): cross-encoder 4.3 s, colbert 46.5 s.

Hybrid is the exception: fusion hands the reranker only the top-5 fused chunks,
not 20, so the cross-encoder scores 5 pairs there and is slightly faster than
ColBERT's fixed query-encoding cost. With 20 candidates (dense, sparse) ColBERT
is ~3× faster, and its latency barely moves with candidate count.

## Headline (dense first stage)

- Rerank latency p50: cross-encoder 206.7 ms → ColBERT 66.5 ms (68% faster, 3.11× the cross-encoder's speed).
- Rerank latency p95: 350.6 ms → 68.3 ms (81% faster).
- MRR: cross-encoder 0.914, ColBERT 0.916 (100.2% retained).

## Before caching passage embeddings

The first version re-encoded every candidate per query through RAGatouille's
index-free `rerank()` ([run](https://github.com/samarthgour2005/RAG_evo/actions/runs/39844563379)):
identical quality, but dense rerank p50 was 1466.8 ms against the cross-encoder's
240.7 ms. Caching the passage side is what makes late interaction pay.

## Resume bullets (generated from the numbers above)

- Replaced a cross-encoder reranker (ms-marco-MiniLM-L-6) with ColBERTv2 late-interaction reranking via RAGatouille, cutting median reranking latency 68% (207 ms → 67 ms, 3.1× faster) on CPU while retaining 100% of its MRR (0.916 vs 0.914).
- Built a reproducible reranker benchmark over 53 labelled questions (top-5 from 20 candidates, dense/sparse/hybrid first stages), reporting Precision@5, Recall@5, MRR, hit rate and p50/p95 latency per stage.
- Reached Recall@5 0.972 and hit rate 0.981 with ColBERT reranking (cross-encoder: 0.981 / 0.981), lifting MRR +4.3% over first-stage dense retrieval (0.878 → 0.916).

## Machine

- python: 3.12.15
- platform: Linux-6.17.0-1022-azure-x86_64-with-glibc2.39 (GitHub Actions ubuntu-latest)
- cpu_count: 4
- torch: 2.14.1+cu130 (CPU)
- torch_threads: 2

Latency is a property of the machine; quality is a property of the index. Re-run on the deployment host before quoting latency for it.
