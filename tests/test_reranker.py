"""Tests for the rerankers — mocked to avoid model downloads."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from chunking.chunk import Chunk
from retrieval.ranking import RetrievalResult
from retrieval.reranker import (
    BaseReranker,
    ColBERTReranker,
    CrossEncoderReranker,
    build_reranker,
    shared_reranker,
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
# ColBERTReranker (encoders stubbed with hand-made token embeddings)
# ---------------------------------------------------------------------------

# Two-dimensional "token embeddings" so MaxSim can be checked by hand. The
# query has two tokens, [1, 0] and [0, 1].
_QUERY = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
_PASSAGES = {
    "This is passage number 0.": torch.tensor([[0.7, 0.0], [0.0, 0.5]]),  # 1.2
    "This is passage number 1.": torch.tensor([[0.3, 0.1], [0.1, 0.2]]),  # 0.5
    "This is passage number 2.": torch.tensor([[0.9, 0.0], [0.0, 0.9]]),  # 1.8
}


class _StubColBERT(ColBERTReranker):
    def __init__(self):
        super().__init__(model_name="stub", model=MagicMock())
        self.encoded: list[list[str]] = []

    def _encode_passages(self, texts):
        self.encoded.append(list(texts))
        return [_PASSAGES[t] for t in texts]

    def _encode_query(self, query):
        return _QUERY


@pytest.fixture
def colbert():
    return _StubColBERT()


class TestColBERTReranker:
    def test_is_base_reranker(self, colbert):
        assert isinstance(colbert, BaseReranker)
        assert colbert.backend == "colbert"

    def test_reranks_by_maxsim(self, colbert):
        reranked = colbert.rerank("q", [_result(0), _result(1), _result(2)], top_k=3)
        assert [r.chunk.chunk_id for r in reranked] == [
            "chunk_2",
            "chunk_0",
            "chunk_1",
        ]
        assert [r.score for r in reranked] == pytest.approx([1.8, 1.2, 0.5])
        assert [r.rank for r in reranked] == [1, 2, 3]

    def test_top_k_limits_output(self, colbert):
        reranked = colbert.rerank("q", [_result(i) for i in range(3)], top_k=2)
        assert len(reranked) == 2

    def test_passages_are_encoded_once(self, colbert):
        """The point of the cache: a second query encodes no passage."""
        results = [_result(i) for i in range(3)]
        colbert.rerank("first", results, top_k=3)
        colbert.rerank("second", results, top_k=3)
        assert colbert.encoded == [[r.chunk.text for r in results]]

    def test_warm_precomputes_and_skips_known_passages(self, colbert):
        colbert.warm([_result(0).chunk.text, _result(1).chunk.text])
        colbert.rerank("q", [_result(0), _result(1), _result(2)], top_k=3)
        assert colbert.encoded == [
            [_result(0).chunk.text, _result(1).chunk.text],
            [_result(2).chunk.text],
        ]

    def test_cache_is_bounded(self, colbert):
        colbert.max_cached = 2
        colbert.warm([_result(i).chunk.text for i in range(3)])
        assert list(colbert._passages) == [
            _result(1).chunk.text,
            _result(2).chunk.text,
        ]

    def test_identical_texts_stay_distinct_results(self, colbert):
        a = _result(0)
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
        reranked = colbert.rerank("q", [a, b], top_k=2)
        assert {r.chunk.chunk_id for r in reranked} == {"chunk_0", "chunk_dup"}

    def test_negative_similarities_score_as_padding(self, colbert):
        """Matches RAGatouille, whose zero padding makes 0 a floor per token."""
        _PASSAGES["neg"] = torch.tensor([[-0.5, -0.5]])
        try:
            r = RetrievalResult(
                chunk=Chunk(
                    chunk_id="n", document_id="d", text="neg", start_char=0, end_char=3
                ),
                score=0.0,
                rank=1,
            )
            assert colbert.rerank("q", [r], top_k=1)[0].score == 0.0
        finally:
            del _PASSAGES["neg"]

    def test_records_reranker_stage_in_trace(self, colbert):
        reranked = colbert.rerank("q", [_result(i) for i in range(3)], top_k=3)
        assert reranked[0].trace.reranker.rank == 1
        assert reranked[0].trace.reranker.score == pytest.approx(1.8)

    def test_empty_input_returns_empty(self, colbert):
        assert colbert.rerank("q", [], top_k=5) == []
        assert colbert.encoded == []

    def test_loads_through_ragatouille(self):
        fake = MagicMock()
        with patch.dict("sys.modules", {"ragatouille": fake}):
            reranker = ColBERTReranker(model_name="colbert-ir/colbertv2.0")
        fake.RAGPretrainedModel.from_pretrained.assert_called_once_with(
            "colbert-ir/colbertv2.0", n_gpu=0, verbose=0
        )
        assert reranker._model is fake.RAGPretrainedModel.from_pretrained.return_value

    def test_encoders_call_colbert(self):
        model = MagicMock()
        model.model.inference_ckpt.docFromText.return_value = (["e0", "e1"],)
        model.model._encode_index_free_queries.return_value = [torch.zeros(1, 3, 2)]
        reranker = ColBERTReranker(model_name="m", model=model)

        assert reranker._encode_passages(["a", "b"]) == ["e0", "e1"]
        model.model._set_inference_max_tokens.assert_called_once_with(
            documents=["a", "b"]
        )
        model.model.inference_ckpt.docFromText.assert_called_once_with(
            ["a", "b"], bsize=32, keep_dims=False
        )
        assert reranker._encode_query("q").shape == (3, 2)


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


class TestSharedReranker:
    @pytest.fixture(autouse=True)
    def _empty_cache(self):
        from retrieval import reranker as module

        module._shared.clear()
        yield
        module._shared.clear()

    def test_built_once_and_shared(self):
        with patch("retrieval.reranker.ColBERTReranker") as cls:
            first = shared_reranker("colbert")
            assert shared_reranker("colbert") is first
        cls.assert_called_once_with()

    def test_colbert_load_failure_falls_back_to_cross_encoder(self):
        with (
            patch(
                "retrieval.reranker.ColBERTReranker",
                side_effect=NotImplementedError("Cannot copy out of meta tensor"),
            ),
            patch("retrieval.reranker.CrossEncoderReranker") as cross,
        ):
            assert shared_reranker("colbert") is cross.return_value

    def test_cross_encoder_failure_is_not_swallowed(self):
        with patch(
            "retrieval.reranker.CrossEncoderReranker", side_effect=OSError("no weights")
        ):
            with pytest.raises(OSError):
                shared_reranker("cross-encoder")

    def test_concurrent_callers_build_one_model(self):
        import threading
        import time

        started = []

        def slow_build():
            started.append(1)
            time.sleep(0.05)
            return MagicMock()

        with patch("retrieval.reranker.ColBERTReranker", side_effect=slow_build):
            got = []
            threads = [
                threading.Thread(target=lambda: got.append(shared_reranker("colbert")))
                for _ in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(started) == 1
        assert len({id(r) for r in got}) == 1
