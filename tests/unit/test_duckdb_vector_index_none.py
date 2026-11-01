"""database.duckdb_vector_index = "none": no HNSW index, exact semantic search."""

import asyncio
from pathlib import Path

import pytest

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator

_VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def _hnsw_index_names(provider: DuckDBProvider) -> list[str]:
    return [
        str(row["index_name"]) for row in provider.get_existing_vector_indexes()
    ]


def _embed(provider: DuckDBProvider, chunk_ids: list[int]) -> None:
    # batch_size=1 makes every insert take the large-batch path that drops
    # the HNSW indexes and recreates them afterwards.
    provider.insert_embeddings_batch(
        [
            {
                "chunk_id": chunk_id,
                "provider": "test",
                "model": "mini",
                "embedding": vector,
                "dims": 3,
            }
            for chunk_id, vector in zip(chunk_ids, _VECTORS)
        ],
        batch_size=1,
    )


def test_none_drops_existing_hnsw_and_never_rebuilds_it(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    db_path = tmp_path / "db.duckdb"
    source = tmp_path / "three.py"
    source.write_text(
        "def first():\n    return 1\n\n"
        "def second():\n    return 2\n\n"
        "def third():\n    return 3\n"
    )

    # An index built by an earlier run with the default setting.
    provider = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
    provider.connect()
    result = asyncio.run(
        IndexingCoordinator(provider, tmp_path).process_file(
            source, skip_embeddings=True
        )
    )
    chunks = provider.get_chunks_by_file_id(result["file_id"], as_model=False)
    chunk_ids = sorted(int(chunk["id"]) for chunk in chunks)[:2]
    assert len(chunk_ids) == 2
    _embed(provider, chunk_ids)
    if not _hnsw_index_names(provider):
        provider.disconnect()
        pytest.skip("DuckDB HNSW indexes are unavailable in this environment")
    provider.disconnect()

    provider = DuckDBProvider(
        db_path=db_path,
        base_directory=tmp_path,
        config=DatabaseConfig(duckdb_vector_index="none"),
    )
    provider.connect()
    try:
        assert _hnsw_index_names(provider) == []

        provider.execute_query("DELETE FROM embeddings_3", [])
        _embed(provider, chunk_ids)
        assert _hnsw_index_names(provider) == []

        results, _ = provider.search_semantic(
            [0.1, 1.0, 0.0], provider="test", model="mini", page_size=2
        )
        assert [int(row["chunk_id"]) for row in results][0] == chunk_ids[1]
        assert _hnsw_index_names(provider) == []
    finally:
        provider.disconnect()
