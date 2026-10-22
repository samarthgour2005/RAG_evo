"""Second-stage re-ranking: ColBERT late interaction, with the cross-encoder kept
as the baseline it replaced.

Architecture
------------
Stage 1 (fast):   Dense retrieval retrieves top-K*M candidates in ~5 ms
Stage 2 (slower): A reranker re-scores those candidates and returns the true top-K

Two rerankers implement that second stage:

* ``ColBERTReranker`` (default) — ColBERTv2 via RAGatouille. Query and passage
  are encoded *separately* into one vector per token, then scored by MaxSim:
  each query token takes its best-matching passage token and the maxima are
  summed. Token-level matching keeps most of a cross-encoder's precision, but
  the passage side is a plain BERT encode — no query concatenated onto every
  passage, no attention across the pair — so it batches far better.
* ``CrossEncoderReranker`` — ms-marco-MiniLM-L-6 reads query and passage
  jointly, one forward pass per pair. The most precise of the two and the most
  expensive, which made it the latency bottleneck it was swapped out for.

Which one runs is ``RERANKER_BACKEND`` (``colbert`` | ``cross-encoder``). The
cross-encoder stays selectable because the swap is only justified by measuring
both on the same data — see ``scripts/compare_rerankers.py``.
"""

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import List, Literal, Sequence

from sentence_transformers import CrossEncoder

from config.logging_config import get_logger
from config.settings import COLBERT_MODEL, CROSS_ENCODER_MODEL, RERANKER_BACKEND
from retrieval.ranking import RetrievalResult, StageScore

logger = get_logger(__name__)

RerankerBackend = Literal["colbert", "cross-encoder"]
RERANKER_BACKENDS: tuple[RerankerBackend, ...] = ("colbert", "cross-encoder")


class BaseReranker(ABC):
    """Provider-agnostic interface for all re-rankers."""

    #: Which backend this is, so the API and the benchmark can report it.
    backend: str = "unknown"

    @abstractmethod
    def rerank(
        self, query: str, results: List[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        """Re-score and re-rank results; return the top_k highest-scoring ones."""


def _ranked(
    results: Sequence[RetrievalResult], scores: Sequence[float], top_k: int
) -> List[RetrievalResult]:
    """Order results by reranker score, recording that score in each trace.

    The earlier stages' opinions are kept, not overwritten. The retrieval
    table's whole purpose is showing that the reranker promoted a chunk dense
    had ninth — which is unanswerable if reranking discards what came before it.
    """
    ranked = sorted(zip(results, scores), key=lambda x: float(x[1]), reverse=True)
    return [
        RetrievalResult(
            chunk=r.chunk,
            score=float(s),
            rank=i + 1,
            trace=replace(r.trace, reranker=StageScore(score=float(s), rank=i + 1)),
        )
        for i, (r, s) in enumerate(ranked[:top_k])
    ]


class CrossEncoderReranker(BaseReranker):
    """sentence-transformers CrossEncoder wrapped in the BaseReranker interface.

    Usage
    -----
    >>> reranker = CrossEncoderReranker()
    >>> reranked = reranker.rerank(query, initial_results, top_k=5)
    """

    backend = "cross-encoder"

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL) -> None:
        self._model = CrossEncoder(model_name)
        self._model_name = model_name
        logger.info("CrossEncoderReranker ready (model=%s)", model_name)

    def rerank(
        self, query: str, results: List[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        if not results:
            return []

        pairs = [(query, r.chunk.text) for r in results]
        scores = self._model.predict(pairs)
        reranked = _ranked(results, [float(s) for s in scores], top_k)

        logger.info(
            "Cross-encoder reranked %d -> %d results for query '%.40s...'",
            len(results),
            len(reranked),
            query,
        )
        return reranked


class ColBERTReranker(BaseReranker):
    """ColBERTv2 late-interaction reranking through RAGatouille.

    Index-free: RAGatouille encodes the candidates in memory for each query, so
    nothing about the FAISS index changes and no ColBERT index is built. That is
    the right trade at this candidate width (top-K × 4, i.e. 20 passages); a
    ColBERT *index* only pays off when ColBERT is also the first-stage retriever.

    Usage
    -----
    >>> reranker = ColBERTReranker()
    >>> reranked = reranker.rerank(query, initial_results, top_k=5)
    """

    backend = "colbert"

    def __init__(self, model_name: str = COLBERT_MODEL, model=None) -> None:
        if model is None:
            # Imported here, not at module level: RAGatouille pulls in torch,
            # colbert-ai and its langchain integration, and a deployment that
            # never reranks should not pay for any of it at import time.
            from ragatouille import RAGPretrainedModel

            # n_gpu=0: the deployment is CPU-only, and RAGatouille otherwise
            # probes for CUDA on every load.
            model = RAGPretrainedModel.from_pretrained(model_name, n_gpu=0, verbose=0)
        self._model = model
        self._model_name = model_name
        logger.info("ColBERTReranker ready (model=%s)", model_name)

    def rerank(
        self, query: str, results: List[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        if not results:
            return []

        ranked = self._model.rerank(
            query=query,
            documents=[r.chunk.text for r in results],
            k=len(results),
        )
        # `result_index` points back into the documents list, so two chunks
        # with identical text cannot be confused for one another the way
        # matching on `content` would. A candidate RAGatouille did not return
        # scores lowest rather than vanishing.
        scores = [float("-inf")] * len(results)
        for hit in ranked:
            scores[hit["result_index"]] = float(hit["score"])
        reranked = _ranked(results, scores, top_k)

        logger.info(
            "ColBERT reranked %d -> %d results for query '%.40s...'",
            len(results),
            len(reranked),
            query,
        )
        return reranked


def build_reranker(backend: str = RERANKER_BACKEND) -> BaseReranker:
    """Construct the reranker a backend name refers to."""
    if backend == "colbert":
        return ColBERTReranker()
    if backend == "cross-encoder":
        return CrossEncoderReranker()
    raise ValueError(
        f"Unknown reranker backend {backend!r}; expected one of {RERANKER_BACKENDS}"
    )
