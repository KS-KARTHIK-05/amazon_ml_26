"""Streaming TSV ingestion for the entity-resolution pipeline.

Reads the challenge's source/ground-truth TSVs in chunks (never the whole file
at once) and writes an internal, atomically-published Parquet representation
with a stable integer ``row_id`` per source. Raw text is preserved verbatim;
missingness is tracked explicitly instead of being collapsed into the literal
string ``"nan"``.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

# Read every field as a string and keep blanks as "" rather than NaN. The
# challenge fields (name/address/country) are free text; pandas' default NA
# sniffing would otherwise turn tokens like the literal word "NA" (a US state
# abbreviation fragment) into a missing value.
_READ_KWARGS = dict(
    sep="\t",
    dtype=str,
    keep_default_na=False,
    na_filter=False,
    encoding="utf-8",
)


def _atomic_write_parquet(table: pa.Table, out_path: str) -> None:
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".parquet.tmp")
    os.close(fd)
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def iter_source_chunks(path: str, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    """Yield the source TSV at ``path`` in row chunks, validating its header."""
    reader = pd.read_csv(path, chunksize=chunksize, **_READ_KWARGS)
    for chunk in reader:
        missing = set(SOURCE_COLUMNS) - set(chunk.columns)
        if missing:
            raise ValueError(f"{path}: missing expected columns {sorted(missing)}")
        yield chunk[SOURCE_COLUMNS]


def iter_ground_truth_chunks(
    path: str, chunksize: int = 200_000
) -> Iterator[pd.DataFrame]:
    reader = pd.read_csv(path, chunksize=chunksize, **_READ_KWARGS)
    for chunk in reader:
        missing = set(GROUND_TRUTH_COLUMNS) - set(chunk.columns)
        if missing:
            raise ValueError(f"{path}: missing expected columns {sorted(missing)}")
        yield chunk[GROUND_TRUTH_COLUMNS]


@dataclass(frozen=True)
class IngestStats:
    source_label: str
    row_count: int
    out_path: str


def ingest_source(
    path: str, source_label: str, out_path: str, chunksize: int = 200_000
) -> IngestStats:
    """Stream ``path`` into a row_id-keyed Parquet table at ``out_path``.

    ``row_id`` is a per-source, 0-based sequential integer assigned in file
    order. It is an internal key only; it and the entity_id's own digits are
    never used as a model feature (see docs/IMPLEMENTATION_SPEC.md).
    """
    tables = []
    next_row_id = 0
    for chunk in iter_source_chunks(path, chunksize=chunksize):
        n = len(chunk)
        chunk = chunk.copy()
        chunk.insert(0, "row_id", range(next_row_id, next_row_id + n))
        chunk.insert(1, "source", source_label)
        next_row_id += n
        tables.append(pa.Table.from_pandas(chunk, preserve_index=False))
    if tables:
        full_table = pa.concat_tables(tables)
    else:
        full_table = pa.Table.from_pandas(
            pd.DataFrame(columns=["row_id", "source"] + SOURCE_COLUMNS),
            preserve_index=False,
        )
    _atomic_write_parquet(full_table, out_path)
    return IngestStats(source_label=source_label, row_count=next_row_id, out_path=out_path)


def build_entity_id_index(parquet_path: str) -> dict[str, int]:
    """Return ``{entity_id: row_id}`` for an ingested source table.

    Loaded fully into memory by design: a single source's ID index (millions
    of short strings) fits comfortably in the session's RAM budget and this
    avoids re-deriving it on every downstream lookup.
    """
    table = pq.read_table(parquet_path, columns=["entity_id", "row_id"])
    entity_ids = table.column("entity_id").to_pylist()
    row_ids = table.column("row_id").to_pylist()
    return dict(zip(entity_ids, row_ids))


def read_table(parquet_path: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pq.read_table(parquet_path, columns=columns).to_pandas()
