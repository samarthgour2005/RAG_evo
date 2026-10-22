"""Tests for the rerankers — mocked to avoid model downloads."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chunking.chunk import Chunk
from retrieval.ranking import RetrievalResult
from retrieval.reranker import (
    BaseReranker,
    ColBERTReranker,
    CrossEncoderReranker,
    build_reranker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(idx: int, score: float = 0.5) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=f"chunk_{idx}",
        document_id="doc",
        text=f"This is passage number {idx}.",
        start_char=idx * 50,
        end_char=idx * 50 + 30,
    )
    return RetrievalResult(chunk=chunk, score=score, rank=idx + 1)


# ---------------------------------------------------------------------------
# BaseReranker interface
# ---------------------------------------------------------------------------


class TestBaseRerankerInterface:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseReranker()

    def test_concrete_subclass_must_implement_rerank(self):
        class Bad(BaseReranker):
            pass

        with pytest.raises(TypeError):
            Bad()


# ---------------------------------------------------------------------------
# CrossEncoderReranker (mocked model)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cross_encoder(monkeypatch):
    """Patch CrossEncoder so no model is downloaded during tests."""
    mock_model = MagicMock()
    # Simulate scores: passage 2 is most relevant (score=0.9), then 0 (0.7), then 1 (0.3)
    mock_model.predict.return_value = np.array([0.7, 0.3, 0.9])

    with patch("retrieval.reranker.CrossEncoder", return_value=mock_model):
        reranker = CrossEncoderReranker(model_name="mock-model")
        reranker._model = mock_model
        yield reranker


class TestCrossEncoderReranker:
    def test_reranker_is_base_reranker(self, mock_cross_encoder):
        assert isinstance(mock_cross_encoder, BaseReranker)

    def test_reranks_by_score_descending(self, mock_cross_encoder):
        results = [_result(0, 0.5), _result(1, 0.4), _result(2, 0.3)]
        reranked = mock_cross_encoder.rerank("query", results, top_k=3)
        # Mock scores: idx0=0.7, idx1=0.3, idx2=0.9 → order: 2, 0, 1
        assert reranked[0].chunk.chunk_id == "chunk_2"
        assert reranked[1].chunk.chunk_id == "chunk_0"
        assert reranked[2].chunk.chunk_id == "chunk_1"

    def test_top_k_limits_output(self, mock_cross_encoder):
        results = [_result(i) for i in range(3)]
        reranked = mock_cross_encoder.rerank("query", results, top_k=2)
        assert len(reranked) == 2

    def test_ranks_are_reset_after_reranking(self, mock_cross_encoder):
        results = [_result(i) for i in range(3)]
        reranked = mock_cross_encoder.rerank("query", results, top_k=3)
        assert [r.rank for r in reranked] == [1, 2, 3]

    def test_scores_are_floats(self, mock_cross_encoder):
        results = [_result(i) for i in range(3)]
        reranked = mock_cross_encoder.rerank("query", results, top_k=3)
        assert all(isinstance(r.score, float) for r in reranked)

    def test_empty_input_returns_empty(self, mock_cross_encoder):
        assert mock_cross_encoder.rerank("query", [], top_k=5) == []

    def test_model_called_with_correct_pairs(self, mock_cross_encoder):
        results = [_result(0), _result(1)]
        mock_cross_encoder.rerank("what is this?", results, top_k=2)
        call_args = mock_cross_encoder._model.predict.call_args[0][0]
        assert call_args[0] == ("what is this?", results[0].chunk.text)
        assert call_args[1] == ("what is this?", results[1].chunk.text)


# ---------------------------------------------------------------------------
# ColBERTReranker (mocked RAGatouille model)
# ---------------------------------------------------------------------------


def _ragatouille_output(scores):
    """RAGatouille's rerank() shape: best first, `result_index` into the input."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [
        {"content": f"text {i}", "score": scores[i], "rank": r + 1, "result_index": i}
        for r, i in enumerate(order)
    ]


@pytest.fixture
def colbert():
    model = MagicMock()
    # Passage 2 most relevant, then 0, then 1 — same shape as the cross-encoder.
    model.rerank.return_value = _ragatouille_output([17.0, 9.5, 23.1])
    return ColBERTReranker(model_name="mock-colbert", model=model)


class TestColBERTReranker:
    def test_is_base_reranker(self, colbert):
        assert isinstance(colbert, BaseReranker)
        assert colbert.backend == "colbert"

    def test_reranks_by_maxsim_score(self, colbert):
        results = [_result(0), _result(1), _result(2)]
        reranked = colbert.rerank("query", results, top_k=3)
        assert [r.chunk.chunk_id for r in reranked] == [
            "chunk_2",
            "chunk_0",
            "chunk_1",
        ]
        assert [r.rank for r in reranked] == [1, 2, 3]
        assert reranked[0].score == pytest.approx(23.1)

    def test_top_k_limits_output(self, colbert):
        reranked = colbert.rerank("query", [_result(i) for i in range(3)], top_k=2)
        assert len(reranked) == 2

    def test_scores_every_candidate(self, colbert):
        """k is the candidate count, so nothing is dropped before top_k applies."""
        results = [_result(i) for i in range(3)]
        colbert.rerank("what is this?", results, top_k=1)
        kwargs = colbert._model.rerank.call_args.kwargs
        assert kwargs["query"] == "what is this?"
        assert kwargs["documents"] == [r.chunk.text for r in results]
        assert kwargs["k"] == 3

    def test_identical_texts_stay_distinct(self):
        """Matched by result_index, not by content, so duplicates are not merged."""
        model = MagicMock()
        model.rerank.return_value = _ragatouille_output([1.0, 5.0])
        reranker = ColBERTReranker(model_name="m", model=model)
        a, b = _result(0), _result(1)
        b = RetrievalResult(
            chunk=Chunk(
                chunk_id="chunk_dup",
                document_id="doc",
                text=a.chunk.text,
                start_char=0,
                end_char=1,
            ),
            score=0.1,
            rank=2,
        )
        reranked = reranker.rerank("q", [a, b], top_k=2)
        assert [r.chunk.chunk_id for r in reranked] == ["chunk_dup", "chunk_0"]

    def test_records_reranker_stage_in_trace(self, colbert):
        reranked = colbert.rerank("query", [_result(i) for i in range(3)], top_k=3)
        assert reranked[0].trace.reranker.rank == 1
        assert reranked[0].trace.reranker.score == pytest.approx(23.1)

    def test_empty_input_returns_empty(self, colbert):
        assert colbert.rerank("query", [], top_k=5) == []
        colbert._model.rerank.assert_not_called()

    def test_loads_through_ragatouille(self):
        fake = MagicMock()
        with patch.dict("sys.modules", {"ragatouille": fake}):
            reranker = ColBERTReranker(model_name="colbert-ir/colbertv2.0")
        fake.RAGPretrainedModel.from_pretrained.assert_called_once_with(
            "colbert-ir/colbertv2.0", n_gpu=0, verbose=0
        )
        assert reranker._model is fake.RAGPretrainedModel.from_pretrained.return_value


class TestBuildReranker:
    def test_colbert(self):
        with patch("retrieval.reranker.ColBERTReranker") as cls:
            assert build_reranker("colbert") is cls.return_value

    def test_cross_encoder(self):
        with patch("retrieval.reranker.CrossEncoderReranker") as cls:
            assert build_reranker("cross-encoder") is cls.return_value

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown reranker backend"):
            build_reranker("bm42")

    def test_default_is_colbert(self):
        from config.settings import RERANKER_BACKEND

        assert RERANKER_BACKEND == "colbert"
