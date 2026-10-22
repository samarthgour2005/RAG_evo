"""Benchmark ColBERT against the cross-encoder on the labelled dataset.

Run from the project root, after building the index:
    python scripts/compare_rerankers.py
    python scripts/compare_rerankers.py --repeats 5 --retrievers dense hybrid

Writes reports/reranker_comparison.json and docs/reranker_comparison.md, the
latter ending in resume bullets built from the numbers just measured.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import BASE_DIR, REPORTS_DIR, TOP_K
from evaluation.dataset import DatasetLoader
from evaluation.ground_truth import ChunkResolver
from evaluation.reranker_comparison import (
    NO_RERANKER,
    resume_bullets,
    run_comparison,
    to_markdown,
)
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import build_reranker

CANDIDATE_MULTIPLIER = 4  # HybridRetriever's default: top-5 reranked from 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--retrievers", nargs="+", default=["dense", "sparse", "hybrid"]
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=[NO_RERANKER, "cross-encoder", "colbert"],
    )
    parser.add_argument(
        "--markdown", type=Path, default=BASE_DIR / "docs" / "reranker_comparison.md"
    )
    args = parser.parse_args()

    dataset = DatasetLoader.load(resolver=ChunkResolver.from_disk())
    base = HybridRetriever.from_disk()

    def make_retriever(backend: str) -> HybridRetriever:
        # One first stage shared by every backend, so the only thing that
        # differs between rows is the reranker.
        reranker = None if backend == NO_RERANKER else build_reranker(backend)
        retriever = HybridRetriever(
            dense=base._dense,
            bm25=base._bm25,
            candidate_multiplier=CANDIDATE_MULTIPLIER,
            reranker=reranker,
        )
        if reranker is not None:
            # Inside the timed load, so ColBERT's one-off corpus encoding is
            # reported as setup cost rather than hidden from the comparison.
            retriever.prepare_reranker()
        return retriever

    comparison = run_comparison(
        make_retriever,
        dataset,
        backends=args.backends,
        retrievers=args.retrievers,
        top_k=args.top_k,
        candidate_multiplier=CANDIDATE_MULTIPLIER,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "reranker_comparison.json"
    json_path.write_text(json.dumps(comparison.as_dict(), indent=2), encoding="utf-8")
    args.markdown.write_text(to_markdown(comparison), encoding="utf-8")

    print(to_markdown(comparison))
    print(f"Wrote {json_path} and {args.markdown}")
    if not resume_bullets(comparison):
        print("(No resume bullets: both 'cross-encoder' and 'colbert' must be run.)")


if __name__ == "__main__":
    main()
