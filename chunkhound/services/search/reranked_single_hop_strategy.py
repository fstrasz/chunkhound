"""Reranked single-hop search strategy: one vector search, then one rerank call.

1. One vector search for the top candidates (no neighbour expansion)
2. Rerank the top RERANK_CANDIDATES with the embedding provider's reranker
3. Return the reranked candidates, then any further vector results in vector order

Best for:
- The default search when the provider can rerank: one embed and one rerank
  call, about half a second, with ranks that do not depend on a time limit
- Callers that want reranked quality without multi-hop's expansion
"""

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from chunkhound.interfaces.embedding_provider import EmbeddingProvider

# Candidates handed to the reranker. Fixed, so a page never depends on the
# page size or offset that asked for it; 90 is the pool the TFS 66054
# measurements were taken on.
RERANK_CANDIDATES = 90

SingleHopSearchFn = Callable[..., Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]]]


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
        """Vector search, rerank the top RERANK_CANDIDATES, then paginate.

        The order is the reranked candidates (rerank score descending, vector
        rank breaking ties), then vector results beyond RERANK_CANDIDATES in
        vector order. Each result's ``score`` is its rerank score, or its
        similarity when it was not reranked; ``threshold`` applies to
        ``score``. If the rerank call fails, the vector order is returned.

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
        candidates, vector_pagination = await self._single_hop_search(
            query=query,
            page_size=max(RERANK_CANDIDATES, offset + page_size),
            offset=0,
            threshold=0.0,
            provider=provider,
            model=model,
            path_filter=path_filter,
        )
        for result in candidates:
            result["score"] = result.get("similarity", 0.0)

        head = candidates[:RERANK_CANDIDATES]
        rerank_scores: dict[int, float] = {}
        if head:
            try:
                rerank_results = await self._embedding_provider.rerank(
                    query=query,
                    documents=[result["content"] for result in head],
                    top_k=len(head),
                )
                for rerank_result in rerank_results:
                    if 0 <= rerank_result.index < len(head):
                        rerank_scores[rerank_result.index] = rerank_result.score
            except Exception as e:
                logger.warning(f"Reranking failed, returning vector order: {e}")

        for index, score in rerank_scores.items():
            head[index]["score"] = score
        # Reranked rows first, then by rerank score, then by vector rank, so
        # equal scores keep one order across identical runs.
        order = sorted(
            range(len(head)),
            key=lambda i: (i not in rerank_scores, -rerank_scores.get(i, 0.0), i),
        )
        ranked = [head[i] for i in order] + candidates[RERANK_CANDIDATES:]

        if threshold is not None:
            ranked = [r for r in ranked if r["score"] >= threshold]
            total = len(ranked)
        else:
            total = vector_pagination.get("total", len(ranked))

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
            f"({len(rerank_scores)}/{len(head)} candidates reranked, {total} total)"
        )
        return page, pagination
