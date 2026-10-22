"""Head-to-head measurement of the rerankers on the labelled dataset.

The ColBERT swap is only worth making if it keeps the cross-encoder's quality at
a lower latency, and neither half of that claim can be asserted — both have to
be measured on the same questions, the same candidates and the same machine.
This module does that and nothing else; ``scripts/compare_rerankers.py`` is the
command-line entry point.

What is timed
-------------
The reranker *stage*, read from the pipeline trace, not the whole retrieval
call: first-stage retrieval is identical across backends, so including it would
only dilute the difference being measured. End-to-end retrieval latency is
reported alongside it, because that is what a user waits on.

Each backend gets warm-up queries before timing starts. The first forward pass
of a PyTorch model pays for lazy initialisation, and charging that to whichever
backend happens to run first would decide the comparison by ordering.
"""

import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from evaluation.dataset import BenchmarkSample
from evaluation.metrics import (
    hit_rate,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.retrieval_evaluator import percentile

#: Label for the first-stage-only baseline, so "how much does reranking buy at
#: all" sits in the same table as "which reranker".
NO_RERANKER = "none"


@dataclass
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float

    @classmethod
    def of(cls, samples: Sequence[float]) -> "LatencyStats":
        if not samples:
            return cls(0.0, 0.0, 0.0)
        return cls(
            mean_ms=statistics.fmean(samples),
            p50_ms=percentile(list(samples), 50),
            p95_ms=percentile(list(samples), 95),
        )


@dataclass
class BackendResult:
    backend: str
    retriever: str
    top_k: int
    precision_at_k: float
    recall_at_k: float
    hit_rate: float
    mrr: float
    rerank_latency: LatencyStats
    retrieval_latency: LatencyStats
    load_seconds: float = 0.0
    questions: int = 0


@dataclass
class Comparison:
    dataset_size: int
    top_k: int
    candidate_multiplier: int
    repeats: int
    machine: Dict[str, str] = field(default_factory=dict)
    results: List[BackendResult] = field(default_factory=list)

    def get(self, backend: str, retriever: str) -> Optional[BackendResult]:
        for r in self.results:
            if r.backend == backend and r.retriever == retriever:
                return r
        return None

    def as_dict(self) -> dict:
        return asdict(self)


def machine_info() -> Dict[str, str]:
    """Latency is a property of the machine, so the report names it."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    try:
        import os

        info["cpu_count"] = str(os.cpu_count())
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = str(torch.get_num_threads())
    except Exception:  # pragma: no cover - torch is always present in practice
        pass
    return info


def measure_backend(
    retriever,
    dataset: Sequence[BenchmarkSample],
    *,
    backend: str,
    mode: str,
    top_k: int,
    repeats: int = 3,
    warmup: int = 3,
    load_seconds: float = 0.0,
) -> BackendResult:
    """Quality and latency of one (reranker, first-stage retriever) pair.

    ``retriever`` is anything with HybridRetriever's ``retrieve_traced``. Quality
    is computed from the first pass; latency pools every pass, since rankings
    are deterministic and only the timings vary between repeats.
    """
    rerank = backend != NO_RERANKER

    for sample in list(dataset)[:warmup]:
        retriever.retrieve_traced(
            sample.question, top_k=top_k, mode=mode, reranker=rerank
        )

    rerank_ms: List[float] = []
    total_ms: List[float] = []
    precisions, recalls, hits, rrs = [], [], [], []

    for rep in range(max(repeats, 1)):
        for sample in dataset:
            t0 = time.perf_counter()
            results, stages = retriever.retrieve_traced(
                sample.question, top_k=top_k, mode=mode, reranker=rerank
            )
            total_ms.append((time.perf_counter() - t0) * 1000)
            for stage in stages:
                if stage.name == "reranker" and stage.status == "ok":
                    rerank_ms.append(stage.latency_ms)

            if rep == 0:
                ids = [r.chunk.chunk_id for r in results]
                expected = sample.expected_chunk_ids
                precisions.append(precision_at_k(ids, expected, top_k))
                recalls.append(recall_at_k(ids, expected))
                hits.append(hit_rate(ids, expected))
                rrs.append(reciprocal_rank(ids, expected))

    n = max(len(dataset), 1)
    return BackendResult(
        backend=backend,
        retriever=mode,
        top_k=top_k,
        precision_at_k=sum(precisions) / n,
        recall_at_k=sum(recalls) / n,
        hit_rate=sum(hits) / n,
        mrr=mean_reciprocal_rank(rrs),
        rerank_latency=LatencyStats.of(rerank_ms),
        retrieval_latency=LatencyStats.of(total_ms),
        load_seconds=load_seconds,
        questions=len(dataset),
    )


def run_comparison(
    make_retriever: Callable[[str], object],
    dataset: Sequence[BenchmarkSample],
    *,
    backends: Sequence[str],
    retrievers: Sequence[str],
    top_k: int,
    candidate_multiplier: int,
    repeats: int = 3,
    warmup: int = 3,
) -> Comparison:
    """Measure every (backend, retriever) pair.

    ``make_retriever(backend)`` returns a retriever wired to that reranker; the
    time it takes is reported as the model's load time.
    """
    comparison = Comparison(
        dataset_size=len(dataset),
        top_k=top_k,
        candidate_multiplier=candidate_multiplier,
        repeats=repeats,
        machine=machine_info(),
    )
    for backend in backends:
        t0 = time.perf_counter()
        retriever = make_retriever(backend)
        load_s = time.perf_counter() - t0
        for mode in retrievers:
            comparison.results.append(
                measure_backend(
                    retriever,
                    dataset,
                    backend=backend,
                    mode=mode,
                    top_k=top_k,
                    repeats=repeats,
                    warmup=warmup,
                    load_seconds=load_s,
                )
            )
    return comparison


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pct_change(new: float, old: float) -> float:
    return (new - old) / old * 100 if old else 0.0


def headline(comparison: Comparison, retriever: str = "dense") -> Optional[dict]:
    """ColBERT against the cross-encoder on one first-stage retriever."""
    ce = comparison.get("cross-encoder", retriever)
    cb = comparison.get("colbert", retriever)
    base = comparison.get(NO_RERANKER, retriever)
    if ce is None or cb is None:
        return None
    out = {
        "retriever": retriever,
        "rerank_p50_ce_ms": ce.rerank_latency.p50_ms,
        "rerank_p50_colbert_ms": cb.rerank_latency.p50_ms,
        "rerank_p95_ce_ms": ce.rerank_latency.p95_ms,
        "rerank_p95_colbert_ms": cb.rerank_latency.p95_ms,
        "rerank_p50_reduction_pct": -_pct_change(
            cb.rerank_latency.p50_ms, ce.rerank_latency.p50_ms
        ),
        "rerank_p95_reduction_pct": -_pct_change(
            cb.rerank_latency.p95_ms, ce.rerank_latency.p95_ms
        ),
        "speedup_p50": (
            ce.rerank_latency.p50_ms / cb.rerank_latency.p50_ms
            if cb.rerank_latency.p50_ms
            else 0.0
        ),
        "mrr_ce": ce.mrr,
        "mrr_colbert": cb.mrr,
        "mrr_retained_pct": cb.mrr / ce.mrr * 100 if ce.mrr else 0.0,
        "recall_ce": ce.recall_at_k,
        "recall_colbert": cb.recall_at_k,
        "hit_rate_ce": ce.hit_rate,
        "hit_rate_colbert": cb.hit_rate,
    }
    if base is not None:
        out["mrr_baseline"] = base.mrr
        out["mrr_lift_colbert_pct"] = _pct_change(cb.mrr, base.mrr)
        out["mrr_lift_ce_pct"] = _pct_change(ce.mrr, base.mrr)
    return out


def resume_bullets(comparison: Comparison, retriever: str = "dense") -> List[str]:
    """Bullets built only from measured numbers, so none can drift from the run."""
    h = headline(comparison, retriever)
    if h is None:
        return []
    k = comparison.top_k
    n = comparison.dataset_size
    cands = k * comparison.candidate_multiplier
    faster = h["rerank_p50_reduction_pct"] > 0
    bullets = [
        (
            "Replaced a cross-encoder reranker (ms-marco-MiniLM-L-6) with ColBERTv2 "
            "late-interaction reranking via RAGatouille, "
            + (
                f"cutting median reranking latency {h['rerank_p50_reduction_pct']:.0f}% "
                f"({h['rerank_p50_ce_ms']:.0f} ms → {h['rerank_p50_colbert_ms']:.0f} ms, "
                f"{h['speedup_p50']:.1f}× faster) on CPU"
                if faster
                else f"measuring median reranking latency at "
                f"{h['rerank_p50_colbert_ms']:.0f} ms vs {h['rerank_p50_ce_ms']:.0f} ms "
                f"for the cross-encoder on CPU"
            )
            + f" while retaining {h['mrr_retained_pct']:.0f}% of its MRR "
            f"({h['mrr_colbert']:.3f} vs {h['mrr_ce']:.3f})."
        ),
        (
            f"Built a reproducible reranker benchmark over {n} labelled questions "
            f"(top-{k} from {cands} candidates, dense/sparse/hybrid first stages), "
            f"reporting Precision@{k}, Recall@{k}, MRR, hit rate and p50/p95 latency "
            f"per stage."
        ),
        (
            f"Reached Recall@{k} {h['recall_colbert']:.3f} and hit rate "
            f"{h['hit_rate_colbert']:.3f} with ColBERT reranking "
            f"(cross-encoder: {h['recall_ce']:.3f} / {h['hit_rate_ce']:.3f})"
            + (
                f", lifting MRR {h['mrr_lift_colbert_pct']:+.1f}% over first-stage "
                f"dense retrieval ({h['mrr_baseline']:.3f} → {h['mrr_colbert']:.3f})."
                if "mrr_baseline" in h
                else "."
            )
        ),
    ]
    return bullets


def _direction(reduction_pct: float) -> str:
    """ "40% faster" or "20% slower", never "-509% reduction"."""
    if reduction_pct >= 0:
        return f"{reduction_pct:.0f}% faster"
    return f"{-reduction_pct:.0f}% slower"


def to_markdown(comparison: Comparison) -> str:
    k = comparison.top_k
    lines = [
        "# Reranker comparison: ColBERT vs cross-encoder",
        "",
        f"Measured over **{comparison.dataset_size} labelled questions**, top-{k} "
        f"returned from {k * comparison.candidate_multiplier} first-stage "
        f"candidates, {comparison.repeats} timed pass(es) per configuration after "
        "warm-up. Quality comes from the first pass; latency pools every pass.",
        "",
        "Regenerate with `python scripts/compare_rerankers.py`.",
        "",
        "## Quality and latency",
        "",
        f"| Retriever | Reranker | P@{k} | Recall@{k} | MRR | Hit rate "
        "| Rerank p50 | Rerank p95 | Retrieval p50 | Retrieval p95 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(comparison.results, key=lambda r: (r.retriever, r.backend)):
        rr = r.rerank_latency
        rt = r.retrieval_latency
        rerank_cols = (
            "— | —"
            if r.backend == NO_RERANKER
            else f"{rr.p50_ms:.1f} ms | {rr.p95_ms:.1f} ms"
        )
        lines.append(
            f"| {r.retriever} | {r.backend} | {r.precision_at_k:.3f} | "
            f"{r.recall_at_k:.3f} | {r.mrr:.3f} | {r.hit_rate:.3f} | "
            f"{rerank_cols} | {rt.p50_ms:.1f} ms | {rt.p95_ms:.1f} ms |"
        )

    loads = {r.backend: r.load_seconds for r in comparison.results}
    lines += [
        "",
        "Setup time (model load, plus ColBERT's one-off corpus encoding): "
        + ", ".join(f"{b} {s:.1f} s" for b, s in loads.items() if b != NO_RERANKER)
        + ".",
    ]

    h = headline(comparison)
    if h:
        lines += [
            "",
            "## Headline (dense first stage)",
            "",
            f"- Rerank latency p50: cross-encoder {h['rerank_p50_ce_ms']:.1f} ms → "
            f"ColBERT {h['rerank_p50_colbert_ms']:.1f} ms "
            f"({_direction(h['rerank_p50_reduction_pct'])}, "
            f"{h['speedup_p50']:.2f}× the cross-encoder's speed).",
            f"- Rerank latency p95: {h['rerank_p95_ce_ms']:.1f} ms → "
            f"{h['rerank_p95_colbert_ms']:.1f} ms "
            f"({_direction(h['rerank_p95_reduction_pct'])}).",
            f"- MRR: cross-encoder {h['mrr_ce']:.3f}, ColBERT {h['mrr_colbert']:.3f} "
            f"({h['mrr_retained_pct']:.1f}% retained).",
        ]
        bullets = resume_bullets(comparison)
        lines += ["", "## Resume bullets (generated from the numbers above)", ""]
        lines += [f"- {b}" for b in bullets]

    m = comparison.machine
    if m:
        lines += [
            "",
            "## Machine",
            "",
            *(f"- {key}: {value}" for key, value in m.items()),
            "",
            "Latency is a property of the machine; quality is a property of the "
            "index. Re-run on the deployment host before quoting latency for it.",
        ]
    return "\n".join(lines) + "\n"
