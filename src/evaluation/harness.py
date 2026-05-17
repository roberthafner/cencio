from dataclasses import dataclass

from src.ingestion.store import ChunkStore


@dataclass
class QueryResult:
    query: str
    expected_chunk_id: str
    expected_name: str
    expected_type: str
    hit: bool
    rank: int | None  # 1-indexed; None if not found in top-k


@dataclass
class EvaluationReport:
    hit_rate: float
    mrr: float
    total: int
    hits: int
    top_k: int
    mode: str
    per_type: dict[str, dict]
    results: list[QueryResult]

    def print(self) -> None:
        w = 60
        print(f"\n{'=' * w}")
        print(f"  Evaluation Report  —  mode={self.mode}  top_k={self.top_k}")
        print(f"{'=' * w}")
        print(f"  Queries :  {self.total}")
        print(f"  Hit rate:  {self.hit_rate:.1%}  ({self.hits}/{self.total} found in top-{self.top_k})")
        print(f"  MRR     :  {self.mrr:.3f}")

        print(f"\n  By chunk type:")
        for chunk_type, stats in sorted(self.per_type.items()):
            print(
                f"    {chunk_type:<15}"
                f"  hit={stats['hit_rate']:.0%}"
                f"  mrr={stats['mrr']:.3f}"
                f"  (n={stats['total']})"
            )

        misses = [r for r in self.results if not r.hit]
        print(f"\n  Misses ({len(misses)}):")
        if misses:
            for r in misses:
                print(f"    [{r.expected_type}] {r.expected_name!r}")
                print(f"      query: {r.query}")
        else:
            print("    (none)")
        print(f"{'=' * w}\n")


def evaluate(
    store: ChunkStore,
    golden_set: list[dict],
    mode: str = "hybrid",
    top_k: int = 5,
) -> EvaluationReport:
    results: list[QueryResult] = []

    for entry in golden_set:
        query = entry["query"]
        expected_id = entry["chunk_id"]
        repo_name = entry.get("repo_name")

        if mode == "hybrid":
            hits = store.hybrid_search(query, top_k=top_k, repo_name=repo_name)
        elif mode == "semantic":
            hits = store.semantic_search(query, top_k=top_k, repo_name=repo_name)
        elif mode == "keyword":
            raw = store.keyword_search(query, top_k=top_k, repo_name=repo_name)
            hits = [{"id": r["id"]} for r in raw]
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        rank = None
        for i, result in enumerate(hits):
            if result["id"] == expected_id:
                rank = i + 1
                break

        results.append(QueryResult(
            query=query,
            expected_chunk_id=expected_id,
            expected_name=entry.get("chunk_name", ""),
            expected_type=entry.get("chunk_type", ""),
            hit=rank is not None,
            rank=rank,
        ))

    total = len(results)
    hits = sum(1 for r in results if r.hit)
    hit_rate = hits / total if total > 0 else 0.0
    mrr = (
        sum(1.0 / r.rank for r in results if r.rank is not None) / total
        if total > 0
        else 0.0
    )

    per_type: dict[str, dict] = {}
    for r in results:
        t = r.expected_type or "unknown"
        if t not in per_type:
            per_type[t] = {"total": 0, "hits": 0, "rr_sum": 0.0}
        per_type[t]["total"] += 1
        if r.hit:
            per_type[t]["hits"] += 1
            per_type[t]["rr_sum"] += 1.0 / r.rank
    for stats in per_type.values():
        n = stats["total"]
        stats["hit_rate"] = stats["hits"] / n if n > 0 else 0.0
        stats["mrr"] = stats["rr_sum"] / n if n > 0 else 0.0

    return EvaluationReport(
        hit_rate=hit_rate,
        mrr=mrr,
        total=total,
        hits=hits,
        top_k=top_k,
        mode=mode,
        per_type=per_type,
        results=results,
    )
