"""Vectorized character n-gram TF-IDF (Polars), a fast drop-in for
sklearn's TfidfVectorizer(analyzer="char") on millions of short strings.

sklearn's analyzer runs a Python loop per document (minutes for 4-6M names);
here n-grams come from Polars string slicing in Rust, and TF, DF, IDF and L2
norms are group-bys. Weights match sklearn's sublinear_tf + smooth_idf form:
    w = (1 + ln tf) * (ln((N + 1) / (df + 1)) + 1),  rows L2-normalized.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp


def _grams(strings: pl.Series, n: int, max_len: int) -> pl.DataFrame:
    """Long table (row, gram) of character n-grams of ' ' + s + ' '."""
    padded = (" " + strings.fill_null("") + " ").str.slice(0, max_len)
    df = pl.DataFrame({"s": padded}).with_row_index("row")
    slices = [pl.col("s").str.slice(i, n) for i in range(max_len - n + 1)]
    return (
        df.select("row", pl.concat_list(slices).alias("gram"))
        .explode("gram")
        .filter(pl.col("gram").str.len_chars() == n)
    )


class NgramTfidf:
    def __init__(self, n: int = 3, min_df: int = 2, max_df: float = 0.005, max_len: int = 48):
        self.n, self.min_df, self.max_df, self.max_len = n, min_df, max_df, max_len
        self.vocab: pl.DataFrame | None = None

    def fit(self, strings: pl.Series) -> "NgramTfidf":
        n_docs = len(strings)
        max_df = self.max_df if self.max_df >= 1 else int(self.max_df * n_docs)
        df = (
            _grams(strings, self.n, self.max_len).unique()
            .group_by("gram").len().rename({"len": "df"})
            .filter((pl.col("df") >= self.min_df) & (pl.col("df") <= max_df))
        )
        self.vocab = df.with_row_index("col").with_columns(
            (np.log((n_docs + 1) / (pl.col("df") + 1)) + 1).cast(pl.Float32).alias("idf")
        ).select("gram", "col", "idf")
        return self

    def transform(self, strings: pl.Series) -> sp.csr_matrix:
        tf = _grams(strings, self.n, self.max_len).group_by(["row", "gram"]).len().rename({"len": "tf"})
        w = (
            tf.join(self.vocab, on="gram")
            .with_columns(((1 + pl.col("tf").cast(pl.Float32).log()) * pl.col("idf")).alias("w"))
        )
        w = w.with_columns((pl.col("w") / (pl.col("w") ** 2).sum().over("row").sqrt()).alias("w"))
        return sp.csr_matrix(
            (w["w"].to_numpy(), (w["row"].to_numpy(), w["col"].to_numpy())),
            shape=(len(strings), self.vocab.height),
            dtype=np.float32,
        )
