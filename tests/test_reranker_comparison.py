"""Tests for the reranker head-to-head — a fake retriever, no models."""

import pytest

from chunking.chunk import Chunk
from evaluation.dataset import BenchmarkSample
from evaluation.reranker_comparison import (
    NO_RERANKER,
    Comparison,
    LatencyStats,
    headline,
    measure_backend,
    resume_bullets,
    run_comparison,
    to_markdown,
)
from retrieval.pipeline import PipelineStage
from retrieval.ranking import RetrievalResult


def _r(cid: str, rank: int) -> RetrievalResult:
    chunk = Chunk(chunk_id=cid, document_id="d", text=cid, start_char=0, end_char=1)
    return RetrievalResult(chunk=chunk, score=1.0 / rank, rank=rank)


class _FakeRetriever:
    """Puts the relevant chunk at `rank_when_reranked` or `rank_without`."""

    def __init__(self, rerank_ms: float, rank_when_reranked: int, rank_without=2):
        self.rerank_ms = rerank_ms
        self.rank_when_reranked = rank_when_reranked
        self.rank_without = rank_without
        self.calls = 0

    def retrieve_traced(self, question, top_k, mode, reranker):
        self.calls += 1
        pos = self.rank_when_reranked if reranker else self.rank_without
        ids = [f"other_{i}" for i in range(top_k)]
        ids[pos - 1] = f"gold_{question}"
        stages = [PipelineStage(name="dense", status="ok", latency_ms=1.0)]
        if reranker:
            stages.append(
                PipelineStage(name="reranker", status="ok", latency_ms=self.rerank_ms)
            )
        return [_r(c, i + 1) for i, c in enumerate(ids)], stages


DATASET = [
    BenchmarkSample(
        id=i, question=f"q{i}", expected_answer="", expected_chunk_ids=[f"gold_q{i}"]
    )
    for i in range(4)
]


def test_latency_stats_nearest_rank():
    stats = LatencyStats.of([10.0, 20.0, 30.0, 40.0])
    assert stats.mean_ms == pytest.approx(25.0)
    assert stats.p50_ms == 20.0
    assert stats.p95_ms == 40.0
    assert LatencyStats.of([]).p50_ms == 0.0


def test_measure_backend_quality_and_rerank_latency():
    fake = _FakeRetriever(rerank_ms=50.0, rank_when_reranked=1)
    result = measure_backend(
        fake, DATASET, backend="colbert", mode="dense", top_k=5, repeats=2, warmup=1
    )
    assert result.mrr == pytest.approx(1.0)
    assert result.hit_rate == pytest.approx(1.0)
    assert result.precision_at_k == pytest.approx(0.2)
    assert result.rerank_latency.p50_ms == 50.0
    # 1 warm-up + 2 timed passes over 4 questions.
    assert fake.calls == 1 + 2 * 4


def test_no_reranker_reports_no_rerank_latency():
    fake = _FakeRetriever(rerank_ms=50.0, rank_when_reranked=1, rank_without=2)
    result = measure_backend(
        fake, DATASET, backend=NO_RERANKER, mode="dense", top_k=5, repeats=1, warmup=0
    )
    assert result.mrr == pytest.approx(0.5)
    assert result.rerank_latency.p50_ms == 0.0


def _comparison() -> Comparison:
    fakes = {
        NO_RERANKER: _FakeRetriever(0.0, 1, rank_without=4),
        "cross-encoder": _FakeRetriever(200.0, 1),
        "colbert": _FakeRetriever(50.0, 2),
    }
    return run_comparison(
        fakes.__getitem__,
        DATASET,
        backends=list(fakes),
        retrievers=["dense", "hybrid"],
        top_k=5,
        candidate_multiplier=4,
        repeats=1,
        warmup=0,
    )


def test_run_comparison_covers_every_pair():
    comparison = _comparison()
    assert len(comparison.results) == 3 * 2
    assert comparison.get("colbert", "hybrid") is not None
    assert comparison.get("colbert", "sparse") is None


def test_headline_numbers():
    h = headline(_comparison())
    assert h["rerank_p50_reduction_pct"] == pytest.approx(75.0)
    assert h["speedup_p50"] == pytest.approx(4.0)
    assert h["mrr_retained_pct"] == pytest.approx(50.0)
    assert h["mrr_lift_colbert_pct"] == pytest.approx(100.0)


def test_headline_needs_both_backends():
    comparison = _comparison()
    comparison.results = [r for r in comparison.results if r.backend != "colbert"]
    assert headline(comparison) is None
    assert resume_bullets(comparison) == []


def test_resume_bullets_quote_measured_numbers():
    bullets = resume_bullets(_comparison())
    assert len(bullets) == 3
    assert "75%" in bullets[0]
    assert "200 ms → 50 ms" in bullets[0]
    assert "4 labelled questions" in bullets[1]


def test_resume_bullets_do_not_claim_a_speedup_that_did_not_happen():
    comparison = _comparison()
    for r in comparison.results:
        if r.backend == "colbert":
            r.rerank_latency = LatencyStats(400.0, 400.0, 400.0)
    assert "cutting" not in resume_bullets(comparison)[0]


def test_markdown_has_table_and_bullets():
    md = to_markdown(_comparison())
    assert "| dense | colbert |" in md
    assert "| dense | none |" in md
    assert "## Resume bullets" in md
