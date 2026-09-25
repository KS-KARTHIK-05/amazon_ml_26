"""Smoke test: confirm the Polars GPU engine (cudf-polars) works on this machine.

Loads the first 50 rows of train_source1.tsv (TSV -> Parquet on CPU), scans
the Parquet on the GPU, runs a few simple operations with
the GPU engine (raise_on_fail=True, so a silent CPU fallback is impossible), and
checks the result against the CPU engine.

Run:  .venv/bin/python check_polars_gpu.py
"""

import sys
import tempfile
import time
from pathlib import Path

import polars as pl

DATA = Path(__file__).parent / "student_resource/dataset/train/train_source1.tsv"
N_ROWS = 50


def check_install() -> None:
    print(f"polars version: {pl.__version__}")
    try:
        import cudf_polars  # noqa: F401
    except ImportError:
        sys.exit(
            "cudf-polars is not installed, so the GPU engine is unavailable.\n"
            'Install it with:  pip install "polars[gpu]"'
        )
    print(f"cudf-polars version: {cudf_polars.__version__}")


def make_parquet() -> Path:
    # The GPU CSV reader crashes on quote_char=None (cudf-polars bug), and the
    # TSVs must be read without quote handling. So read the TSV on CPU once,
    # write Parquet, and let the GPU scan the Parquet — the real pipeline's path.
    df = pl.read_csv(
        DATA,
        separator="\t",
        quote_char=None,
        infer_schema=False,  # read every column as text
        n_rows=N_ROWS,
    )
    path = Path(tempfile.gettempdir()) / "polars_gpu_check.parquet"
    df.write_parquet(path)
    return path


def build_query(parquet: Path) -> pl.LazyFrame:
    lf = pl.scan_parquet(parquet)
    return (
        lf.with_columns(
            name_lower=pl.col("business_name").str.to_lowercase(),
            name_len=pl.col("business_name").str.len_chars(),
            addr_len=pl.col("business_address").str.len_chars(),
            is_ltd=pl.col("business_name").str.to_lowercase().str.contains("ltd|limited|llc|inc"),
        )
        .filter(pl.col("name_len") > 0)
        .sort("entity_id")
    )


def summary(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.group_by("country")
        .agg(
            rows=pl.len(),
            avg_name_len=pl.col("name_len").mean(),
            legal_suffix_rows=pl.col("is_ltd").sum(),
        )
        .sort("country")
    )


def main() -> None:
    check_install()
    gpu = pl.GPUEngine(device=0, raise_on_fail=True)
    parquet = make_parquet()

    t = time.perf_counter()
    rows_gpu = build_query(parquet).collect(engine=gpu)
    summary_gpu = summary(build_query(parquet)).collect(engine=gpu)
    gpu_s = time.perf_counter() - t

    rows_cpu = build_query(parquet).collect()
    summary_cpu = summary(build_query(parquet)).collect()

    with pl.Config(tbl_rows=10, fmt_str_lengths=40):
        print(f"\nFirst rows (GPU engine, {rows_gpu.height} rows loaded):")
        print(rows_gpu.head(10))
        print("\nPer-country summary (GPU engine):")
        print(summary_gpu)

    same = rows_gpu.equals(rows_cpu) and summary_gpu.equals(summary_cpu)
    print(f"\nGPU run time: {gpu_s:.2f}s (includes CUDA start-up)")
    print(f"GPU result identical to CPU result: {same}")
    if not same:
        sys.exit("FAIL: GPU and CPU results differ")
    print("PASS: Polars GPU engine is working")


if __name__ == "__main__":
    main()
