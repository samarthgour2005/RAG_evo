# ADR 009 — ColBERT replaces the cross-encoder reranker

**Status:** Accepted  
**Date:** 2026-10

---

## Context

The second-stage reranker was `cross-encoder/ms-marco-MiniLM-L-6-v2`. It was the
largest quality lever in the benchmark (it lifted sparse MRR by 0.173) and also
the largest latency cost outside generation, which is why it stayed opt-in.

A cross-encoder concatenates the query onto every candidate and runs one full
forward pass per pair, with attention across both. At top-5 from 20 candidates
that is 20 joint passes per query, none of which can be shared.

## Decision

Rerank with ColBERTv2 (`colbert-ir/colbertv2.0`) through RAGatouille, as the
default. Keep the cross-encoder as a selectable backend.

- `retrieval/reranker.py` — `ColBERTReranker` beside `CrossEncoderReranker`,
  both behind `BaseReranker`; `build_reranker()` picks one.
- `RERANKER_BACKEND` (`colbert` | `cross-encoder`) selects it per deployment;
  `/config` and `/settings` report which one is live.
- Passage token embeddings are computed once and cached by text: the corpus is
  encoded when the reranker is first loaded, and chunks indexed later on first
  use. A query then encodes only the question and runs MaxSim against the
  cached matrices. The FAISS index is untouched, and no ColBERT search index is
  built — ColBERT reranks, it does not retrieve.

### Revision: why the cache exists

The first version called RAGatouille's index-free `rerank()`, which re-encodes
all 20 candidate passages on every query. Measured on a 4-core CI runner over
the 53 questions (dense first stage), quality matched the cross-encoder (MRR
0.916 vs 0.914) but the rerank stage was ~6× slower: p50 1467 ms against
241 ms. ColBERTv2 is BERT-base (110 M parameters) against MiniLM-L-6's 22 M, so
re-encoding passages per query forfeits exactly the advantage late interaction
exists for. Caching the passage side restores it; scores are identical to
`rerank()`'s.

## Why ColBERT

ColBERT encodes query and passage independently into one vector per token and
scores them by MaxSim — each query token takes its best match among the
passage's tokens, and the maxima are summed. That keeps the token-level
matching that makes a cross-encoder precise, without cross-attention over every
pair.

## Measuring it

The claim is "similar quality, lower latency", and both halves are measured
rather than asserted: `scripts/compare_rerankers.py` runs no reranker, the
cross-encoder and ColBERT over the same first-stage candidates on the 53
labelled questions, and reports quality next to the reranker stage's own
p50/p95 latency (from the pipeline trace, so first-stage retrieval — identical
across backends — does not dilute the difference). Results are written to
`docs/reranker_comparison.md`.

If a run on the deployment host shows ColBERT is *not* faster there, switch
`RERANKER_BACKEND` back; nothing else changes.

## Consequences

- **Dependencies.** RAGatouille brings colbert-ai and its langchain integration.
  Two pins keep it importable: `langchain<1` (RAGatouille 0.0.9 imports
  `langchain.retrievers`, removed in 1.0) and `transformers<5` (colbert-ai's
  `HF_ColBERT` fails to load under 5.x).
- **A C++ compiler at runtime.** colbert-ai JIT-compiles a MaxSim kernel on
  first load. The Docker runtime stage installs `g++` and compiles the kernel at
  build time.
- **Weights.** ColBERTv2 is ~440 MB against the cross-encoder's ~90 MB. Loaded
  lazily, on the first reranked request, as the cross-encoder was.
- **Score scale changes.** ColBERT scores are MaxSim sums (typically 10–30),
  not cross-encoder logits. Nothing compares reranker scores across backends;
  the retrieval table shows them as-is.
