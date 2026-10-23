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
  summed. Passages do not depend on the query, so their embeddings are
  computed once and cached; a query costs one short encode plus a matrix
  product per candidate.
* ``CrossEncoderReranker`` — ms-marco-MiniLM-L-6 reads query and passage
  jointly, one forward pass per pair. The most precise of the two and the most
  expensive, which made it the latency bottleneck it was swapped out for.

Which one runs is ``RERANKER_BACKEND`` (``colbert`` | ``cross-encoder``). The
cross-encoder stays selectable because the swap is only justified by measuring
both on the same data — see ``scripts/compare_rerankers.py``.
"""

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, List, Literal, Sequence

from sentence_transformers import CrossEncoder

from config.logging_config import get_logger
from config.settings import COLBERT_MODEL, CROSS_ENCODER_MODEL, RERANKER_BACKEND
from retrieval.ranking import RetrievalResult, StageScore

if TYPE_CHECKING:  # pragma: no cover - torch is only imported where it is used
    import torch

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

    Late interaction is what makes ColBERT cheap, and only if it is used as
    designed: a passage's token embeddings do not depend on the query, so they
    are computed once and reused. A query then costs one short BERT pass over
    the question plus a MaxSim — a matrix product per candidate.

    The first version called RAGatouille's index-free ``rerank()``, which
    re-encodes every candidate passage on every query. Measured on CI that was
    ~6× *slower* than the cross-encoder (BERT-base over 20 passages per query
    against MiniLM's 20 pair passes), which is the opposite of the reason for
    the swap. Passage embeddings are therefore cached by text: ``warm()``
    encodes a corpus up front, and a candidate not yet seen (a freshly uploaded
    chunk) is encoded on first use and kept.

    Usage
    -----
    >>> reranker = ColBERTReranker()
    >>> reranker.warm([chunk.text for chunk in corpus])
    >>> reranked = reranker.rerank(query, initial_results, top_k=5)
    """

    backend = "colbert"

    #: Upper bound on cached passages. ~70 tokens × 128 dims × 4 bytes ≈ 36 KB
    #: each, so the cap is ~700 MB at worst; the benchmark corpus is 148.
    max_cached = 20_000

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
        # Insertion-ordered, so eviction drops the oldest passage first.
        self._passages: Dict[str, "torch.Tensor"] = {}
        logger.info("ColBERTReranker ready (model=%s)", model_name)

    # -- encoding: the only two places that touch colbert-ai ----------------

    def _encode_passages(self, texts: List[str]) -> List["torch.Tensor"]:
        """One [tokens, dim] tensor per passage, padding removed."""
        colbert = self._model.model
        # Fixes the document length budget once, from the passages it sees;
        # a no-op after the first call.
        colbert._set_inference_max_tokens(documents=texts)
        embedded, *_ = colbert.inference_ckpt.docFromText(
            texts, bsize=32, keep_dims=False
        )
        return list(embedded)

    def _encode_query(self, query: str) -> "torch.Tensor":
        """[query tokens, dim] for one question."""
        (embedded,) = self._model.model._encode_index_free_queries(query)
        return embedded[0]

    # -----------------------------------------------------------------------

    def warm(self, texts: Sequence[str]) -> None:
        """Encode passages ahead of the first query that needs them."""
        missing = list(dict.fromkeys(t for t in texts if t not in self._passages))
        if not missing:
            return
        t0 = time.perf_counter()
        for text, emb in zip(missing, self._encode_passages(missing)):
            self._passages[text] = emb
        while len(self._passages) > self.max_cached:
            self._passages.pop(next(iter(self._passages)))
        logger.info(
            "ColBERT encoded %d passage(s) in %.0f ms (%d cached)",
            len(missing),
            (time.perf_counter() - t0) * 1000,
            len(self._passages),
        )

    def rerank(
        self, query: str, results: List[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        if not results:
            return []

        import torch

        texts = [r.chunk.text for r in results]
        self.warm(texts)
        q = self._encode_query(query)
        with torch.inference_mode():
            # MaxSim: each query token keeps its best-matching passage token,
            # and the maxima are summed. The clamp reproduces RAGatouille's
            # rerank() exactly: it scores zero-padded passages, so a padding
            # token (similarity 0) is always a candidate maximum. Same scores,
            # without re-encoding the passage to get them.
            scores = [
                float(
                    (self._passages[t] @ q.to(self._passages[t].dtype).T)
                    .max(dim=0)
                    .values.clamp(min=0)
                    .sum()
                )
                for t in texts
            ]
        reranked = _ranked(results, scores, top_k)

        logger.info(
            "ColBERT reranked %d -> %d results for query '%.40s...'",
            len(results),
            len(reranked),
            query,
        )
        return reranked


# Model construction goes through one lock. transformers builds a model on
# the meta device and then loads weights into it, and that machinery is not
# safe to run from two threads at once: on the live deployment a benchmark
# sweep and a query each loaded ColBERT concurrently, in FastAPI's threadpool,
# and the loser failed in ``model.to()`` with "Cannot copy out of meta tensor".
_LOAD_LOCK = threading.RLock()
_shared: Dict[str, BaseReranker] = {}


def build_reranker(backend: str = RERANKER_BACKEND) -> BaseReranker:
    """Construct the reranker a backend name refers to."""
    with _LOAD_LOCK:
        if backend == "colbert":
            return ColBERTReranker()
        if backend == "cross-encoder":
            return CrossEncoderReranker()
    raise ValueError(
        f"Unknown reranker backend {backend!r}; expected one of {RERANKER_BACKENDS}"
    )


def shared_reranker(backend: str = RERANKER_BACKEND) -> BaseReranker:
    """The process-wide reranker for ``backend``, built once.

    Every corpus's retriever asks for one, and building one per corpus would
    load the same ~440 MB ColBERT checkpoint once per corpus. The passage cache
    is keyed by text, so sharing it across corpora is safe.

    If ColBERT cannot load, the cross-encoder serves instead of every reranked
    request failing with a 500. Loud, not silent: the failure is logged with
    its traceback, and the reranker's ``backend`` says which one is live.
    """
    with _LOAD_LOCK:
        if backend not in _shared:
            try:
                _shared[backend] = build_reranker(backend)
            except Exception:
                if backend != "colbert":
                    raise
                logger.exception(
                    "ColBERT reranker failed to load — falling back to the "
                    "cross-encoder. Set RERANKER_BACKEND=cross-encoder to skip "
                    "the attempt."
                )
                _shared[backend] = build_reranker("cross-encoder")
        return _shared[backend]
