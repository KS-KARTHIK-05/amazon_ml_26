from er.io import build_entity_id_index, ingest_source, iter_source_chunks, read_table

SAMPLE_TSV = (
    "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    "S1-1\tAcme Corp\t1 Main St\tUS\n"
    "S1-2\tBeta Pvt Ltd\t\tIndia\n"  # missing address on purpose
    "S1-3\tGamma\t2 Oak Ave\tFrance\n"
)


def test_iter_source_chunks_preserves_row_order(tmp_path):
    path = tmp_path / "src.tsv"
    path.write_text(SAMPLE_TSV, encoding="utf-8")
    chunks = list(iter_source_chunks(str(path), chunksize=2))
    assert len(chunks) == 2  # 3 rows, chunksize 2 -> 2+1
    all_ids = [eid for chunk in chunks for eid in chunk["entity_id"]]
    assert all_ids == ["S1-1", "S1-2", "S1-3"]


def test_missing_address_stays_empty_string_not_nan(tmp_path):
    path = tmp_path / "src.tsv"
    path.write_text(SAMPLE_TSV, encoding="utf-8")
    chunk = next(iter_source_chunks(str(path), chunksize=10))
    row = chunk[chunk["entity_id"] == "S1-2"].iloc[0]
    assert row["business_address"] == ""


def test_ingest_source_assigns_sequential_row_id_and_writes_atomically(tmp_path):
    path = tmp_path / "src.tsv"
    path.write_text(SAMPLE_TSV, encoding="utf-8")
    out_path = tmp_path / "ingested.parquet"
    stats = ingest_source(str(path), "S1", str(out_path), chunksize=2)
    assert stats.row_count == 3
    assert out_path.exists()
    table = read_table(str(out_path))
    assert list(table["row_id"]) == [0, 1, 2]
    assert list(table["source"]) == ["S1", "S1", "S1"]
    assert list(table["entity_id"]) == ["S1-1", "S1-2", "S1-3"]
    # Open-set country handling: France flows through untouched.
    assert list(table["country"]) == ["US", "India", "France"]


def test_no_leftover_tmp_file_after_atomic_write(tmp_path):
    path = tmp_path / "src.tsv"
    path.write_text(SAMPLE_TSV, encoding="utf-8")
    out_path = tmp_path / "ingested.parquet"
    ingest_source(str(path), "S1", str(out_path))
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_build_entity_id_index(tmp_path):
    path = tmp_path / "src.tsv"
    path.write_text(SAMPLE_TSV, encoding="utf-8")
    out_path = tmp_path / "ingested.parquet"
    ingest_source(str(path), "S1", str(out_path))
    index = build_entity_id_index(str(out_path))
    assert index == {"S1-1": 0, "S1-2": 1, "S1-3": 2}


def test_ingest_handles_changed_row_order(tmp_path):
    reordered = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-3\tGamma\t2 Oak Ave\tFrance\n"
        "S1-1\tAcme Corp\t1 Main St\tUS\n"
        "S1-2\tBeta Pvt Ltd\t\tIndia\n"
    )
    path = tmp_path / "src.tsv"
    path.write_text(reordered, encoding="utf-8")
    out_path = tmp_path / "ingested.parquet"
    ingest_source(str(path), "S1", str(out_path))
    index = build_entity_id_index(str(out_path))
    # row_id follows file order, whatever it is; no dependence on ID sort order.
    assert index == {"S1-3": 0, "S1-1": 1, "S1-2": 2}
