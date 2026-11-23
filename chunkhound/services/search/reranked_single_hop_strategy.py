"""Reranked single-hop search strategy: one vector search, then one rerank call.

1. One vector search for the top RERANK_CANDIDATES (no neighbour expansion)
2. Rerank them with the embedding provider's reranker
3. Paginate over that reranked set; nothing past it is returned

Best for:
- The default search when the provider can rerank: one embed and one rerank
  call, about half a second, with ranks that do not depend on a time limit
- Callers that want reranked quality without multi-hop's expansion

It returns at most RERANK_CANDIDATES results. A caller that needs more uses
single_hop (unreranked, pages through every match) or multi_hop.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from chunkhound.interfaces.embedding_provider import EmbeddingProvider

# Candidates handed to the reranker, and the most results this strategy
# returns. Fixed, so a page never depends on the page size or offset that
# asked for it; 90 is the pool the TFS 66054 measurements were taken on.
RERANK_CANDIDATES = 90

SingleHopSearchFn = Callable[
    ..., Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]]
]


class RerankedSingleHopStrategy:
    """Single-hop vector search whose top candidates are reranked."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        single_hop_search_fn: SingleHopSearchFn,
    ):
        """Initialize reranked single-hop search strategy.

        Args:
            embedding_provider: Embedding provider with reranking support
            single_hop_search_fn: Function performing the vector search
        """
        self._embedding_provider = embedding_provider
        self._single_hop_search = single_hop_search_fn

    async def search(
        self,
        query: str,
        page_size: int,
        offset: int,
        threshold: float | None,
        provider: str,
        model: str,
        path_filter: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Rerank the top RERANK_CANDIDATES vector results, then paginate them.

        The set is ordered by rerank score descending, vector rank breaking
        ties, and holds at most RERANK_CANDIDATES results; ``total`` counts
        that set, so every page it promises is reachable. Each result's
        ``score`` is its rerank score; ``threshold`` applies to ``score``. If
        the rerank call fails, or leaves a candidate unscored, that candidate
        keeps its similarity as ``score`` and sorts after every reranked one,
        in vector order.

        Args:
            query: Natural language search query
            page_size: Number of results per page
            offset: Starting position for pagination
            threshold: Optional minimum ``score`` to keep a result
            provider: Embedding provider name
            model: Embedding model name
            path_filter: Optional relative path to limit search scope

        Returns:
            Tuple of (results, pagination_metadata)
        """
        candidates, _ = await self._single_hop_search(
            query=query,
            page_size=RERANK_CANDIDATES,
            offset=0,
            threshold=None,
            provider=provider,
            model=model,
            path_filter=path_filter,
        )
        for result in candidates:
            result["score"] = result.get("similarity", 0.0)

        rerank_scores: dict[int, float] = {}
        if candidates:
            try:
                rerank_results = await self._embedding_provider.rerank(
                    query=query,
                    documents=[result["content"] for result in candidates],
                    top_k=len(candidates),
                )
                for rerank_result in rerank_results:
                    if 0 <= rerank_result.index < len(candidates):
                        rerank_scores[rerank_result.index] = rerank_result.score
            except Exception as e:
                logger.warning(f"Reranking failed, returning vector order: {e}")

        for index, score in rerank_scores.items():
            candidates[index]["score"] = score
        # Reranked rows first, then by rerank score, then by vector rank, so
        # equal scores keep one order across identical runs.
        order = sorted(
            range(len(candidates)),
            key=lambda i: (i not in rerank_scores, -rerank_scores.get(i, 0.0), i),
        )
        ranked = [candidates[i] for i in order]

        if threshold is not None:
            ranked = [r for r in ranked if r["score"] >= threshold]

        total = len(ranked)
        page = ranked[offset : offset + page_size]
        has_more = offset + page_size < total
        pagination = {
            "offset": offset,
            "page_size": page_size,
            "has_more": has_more,
            "next_offset": offset + page_size if has_more else None,
            "total": total,
        }
        logger.info(
            f"Reranked single-hop search completed: {len(page)} results returned "
            f"({len(rerank_scores)}/{len(candidates)} candidates reranked, "
            f"{total} total)"
        )
        return page, pagination
