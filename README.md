# Business Entity Resolution — Blocking Engine

Goal of this stage: for every Source 1 entity, produce a short list of plausible Source 2 / Source 3 records (`output/candidate_pairs.tsv`) quickly enough to leave time for the matching model.

```
Raw TSVs → TSV→Parquet (CPU) → script detection → transliteration (lookup table)
        → normalization → split by country → blocking passes A–D
        → union + adaptive top-K → candidate_pairs.tsv → [matching model, later]
```

## Verified so far

- **Polars GPU engine works** (`check_polars_gpu.py` passes) with `polars==1.42.1` and `cudf-polars-cu12==26.08.01`.
  - The GPU TSV reader crashes on `quote_char=None`, so TSVs are always read on the CPU and converted to Parquet first.
- **Pass A benchmark** (US train, the largest country: 976k distinct S1 names vs 5.04M distinct S2+S3 names; RTX 3060, 20 cores):

  | Step | Measured | Full-run estimate |
  |---|---|---|
  | Load + clean US names (polars, CPU) | 1.6 s | — |
  | TF-IDF fit + transform (char 3-grams, vocab 24,972) | 37.7 s | — |
  | CPU sparse multiply, 200 S1 rows | 7.1 s | ~9.6 h on 1 thread, **~36 min on 16 processes** (upper bound; `sparse_dot_topn` is faster) |
  | GPU PyTorch, dense-query tiles, 2,048 S1 rows | 39.7 s | ~5.3 h (**slower than CPU**) |
  | Untiled multiply of 5,000 S1 rows | killed (out of RAM) | — |

  **Takeaways:**
  - Pass A must run tiled with top-K pruning.
  - Its default backend is CPU `sparse_dot_topn` with multiple threads. The dense-query GPU method is not worth it.
  - Blocking for a whole split should fit in a few hours, not days.

---

## Coding-agent prompt

````markdown
# Task: Build the blocking engine (raw TSV → candidate_pairs.tsv) for a business entity-resolution challenge

## Context
- Data: /home/srmist/Desktop/Amazon_ML/student_resource/dataset/train
  - train_source1.tsv (~2.2M rows), train_source2.tsv (~5.0M), train_source3.tsv (~5.3M)
  - Columns: entity_id, business_name, business_address, country. Tab-separated; read ALL columns as strings.
  - train_ground_truth.tsv (source1_entity_id, matched_entity_ids): use it ONLY to evaluate blocking, never inside the method.
- Goal: for every S1 entity_id, list the plausible S2/S3 entity_ids. Blocking must use only S1, S2 and S3.
- The same code must later run on ../test unchanged. The test set adds the country "France" (not in train), so never hard-code country values.
- Facts already verified:
  - S1 is 100% Latin script.
  - Indian scripts (Devanagari, Bengali, Gujarati, Gurmukhi, Kannada, Malayalam, Odia, Tamil, Telugu) appear only in S2/S3 India rows. Some names mix scripts.
  - Across all S2/S3 train+test files there are only ~1,553 distinct Indian-script words: mostly English words written in Indian scripts (प्राइवेट = private), personal names and state names.
  - French rows are Latin with accents plus `º`.
  - No other scripts exist.
- Put code in /home/srmist/Desktop/Amazon_ML/code/business_entity_resolution/src/, with a README.md and a pinned requirements.txt.
- Only MIT/Apache/BSD/ISC dependencies. Do NOT use Unidecode (GPL), even though it is installed. No external APIs or internet lookups.

## Hardware & environment
- venv: /home/srmist/Desktop/Amazon_ML/.venv, Python 3.11.
- Already installed and verified:
  - **polars 1.42.1 + cudf-polars 26.08.01 (GPU engine working)**
  - pyarrow, numpy, scipy, scikit-learn, rapidfuzz, torch 2.6+cu124, psutil
- Pin exactly `polars==1.42.1` and `cudf-polars-cu12==26.08.01`. Do not upgrade or reinstall polars.
- GPU: RTX 3060, 12 GB VRAM (~11 GB free), CUDA 12.5 driver.
- CPU: 20 cores, 31 GB RAM (~21 GB available).
- Budgets: VRAM ≤ 9 GB, RAM ≤ 16 GB. Make both configurable.
- Working reference: /home/srmist/Desktop/Amazon_ML/check_polars_gpu.py passes. Reuse its loading pattern exactly.

## Measured benchmark (US train, Pass A) — design around these numbers
- 976k distinct S1 names vs 5.04M distinct S2+S3 names.
- TF-IDF fit+transform: 38 s; vocab 24,972; 85M non-zeros.
- CPU: 200 S1 rows took 7.1 s single-threaded (161M raw scores). That extrapolates to ~36 min on 16 processes.
- GPU dense-query tiles (torch.sparse.mm): 40 s per 2,048 rows, ~5.3 h total. **Slower than CPU; do not use it as the default.**
- An untiled multiply of 5,000 S1 rows ran out of RAM.
- Conclusion: Pass A runs on CPU with `sparse_dot_topn` (top-K pruning inside the multiply, multi-threaded), in S1 chunks.

## Data loading method (verified; use this for ALL THREE train files)
- ⚠️ **Never read a TSV with the GPU engine.** cudf-polars crashes on `quote_char=None` (`TypeError: 'NoneType' object cannot be interpreted as an integer`). `quote_char=None` is still required, because business names can contain `"` and normal quote handling would merge or split rows.
- **Step 1: TSV → Parquet on CPU, once per file.** Do this for train_source1.tsv, train_source2.tsv and train_source3.tsv:
  ```python
  pl.scan_csv(path, separator="\t", quote_char=None, infer_schema=False) \
    .sink_parquet(f"data/parquet/train/{source}.parquet")   # streams; low RAM
  ```
  - Add a `source` column (S1/S2/S3) and an int32 `rid` row index while converting.
  - Save the `rid → entity_id` lookup table separately.
  - Write partitioned by country (`data/parquet/train/{source}/country=X/`) so later stages can load one country at a time.
- **Sanity-check each file after conversion:**
  - the row count equals the TSV line count − 1;
  - there are exactly 4 original columns with no nulls in entity_id;
  - every entity_id in a file has the expected prefix (S1-/S2-/S3-) and is unique;
  - the country values are listed as found (never assumed).
  - Print one line per file: rows, countries, time.
- **Step 2: everything after conversion uses `pl.scan_parquet(...)`**, collected with `engine=pl.GPUEngine(device=0, raise_on_fail=...)`, or `engine="streaming"` on CPU when `--polars-engine cpu` is set.
- Skip the conversion if the Parquet already exists (`--force` rebuilds it).

## Acceleration strategy (the core design; follow it)

1. **Integer IDs and compact types.**
   - All joins, candidate tables and top-K lists use int32 `rid` and float32 scores.
   - Convert back to the original entity_id strings only when writing the final TSV.

2. **Work on distinct strings only.**
   - Transliteration, normalization, skeletons and TF-IDF vectors are computed on DISTINCT strings.
   - Rows reference them through an int32 string ID.
   - Duplicate names are vectorized and compared once. In US train, this reduces S2+S3 from 6.2M rows to 5.0M names.

3. **Polars engines and fallback policy.**
   - Use the GPU engine for Parquet-based table stages (normalization, joins, group-bys) with RMM managed memory, so the GPU spills into host RAM instead of running out of memory.
   - **Develop with `raise_on_fail=True`**, so unsupported operations fail loudly; rewrite them with GPU-supported expressions. Use `raise_on_fail=False` only in the final full run, and log any stage that fell back.
   - CPU fallback `--polars-engine cpu`: `collect(engine="streaming")`.
   - Benchmark both engines on the 5% sample and keep the faster one as the default.
   - Never use `map_elements` or other Python UDFs in large plans. Apply Python logic only to distinct-value tables, then join the results back.

4. **Process one country at a time.**
   - Load only that country's S1 and S2+S3 Parquet partitions.
   - Free all memory (`del`, `gc.collect()`) before moving to the next country.

5. **Keep hash-join passes (B, C, D) bounded.**
   - Compute key frequencies first. Drop keys shared by more than `max_key_size` rows (default 200).
   - **Before every join, log the estimated output size** = sum over keys of (S1 count × S2/S3 count). Abort the chunk if it exceeds the budget. That number, not the input size, determines memory.
   - Join S1 in chunks of 200k rows against the filtered S2+S3 key table.
   - Keep at most `per_pass_cap` (default 50) candidates per S1, preferring the rarest keys.
   - Write each chunk to `work/{split}/country=X/pass=P/part-NNNNN.parquet` straight away.

6. **Pass A: TF-IDF name similarity on CPU, chunked, with top-K pruning (measured to be the fastest option).**
   - Fit `TfidfVectorizer(analyzer='char_wb', ngram_range=(3,3), dtype=float32, min_df=2, max_df=0.05, sublinear_tf=True)` on that country's distinct names.
   - Transform S2+S3 once and keep `B.T` as CSR.
   - Split S1 distinct names into chunks of ~5k rows. Run `sparse_dot_topn.sp_matmul_topn(A_chunk, B_T, top_n=30, threshold=0.3, n_threads=16)`, or distribute the chunks over 16 joblib processes that share `B_T` through memory-mapping.
   - **Never multiply without top-K pruning.** A plain scipy multiply of 5,000 rows ran out of RAM.
   - Stream each chunk's top-K results to Parquet.
   - Tune `max_df` (try 0.01–0.05) on the sample. A lower value drops more common n-grams, which is the biggest speed-up; check the recall cost.
   - Target: ≤ ~40 min per country on the full train split.
   - Do not run Polars GPU stages and any PyTorch GPU code in the same process. If a GPU TF-IDF variant is tried later, run it as its own process communicating via Parquet, and keep it only if it beats the CPU benchmark.

7. **Resume and monitor.**
   - Each stage skips its work if its output exists (`--force` recomputes).
   - Set `POLARS_MAX_THREADS=16`.
   - Log time, peak RAM (psutil) and peak VRAM for each stage.
   - Show progress bars per country/chunk.
   - **Treat any out-of-memory error or CPU fallback as a bug to investigate, not normal behaviour.**

## Pipeline

1. **Load:** convert all three train TSVs → Parquet (CPU, `quote_char=None`), then `scan_parquet`. Keep the raw columns and add `source` + int32 `rid`.

2. **Detect script for each word** using Unicode ranges → Latin / one of the Indian scripts / other.

3. **Transliterate by table lookup.**
   - Build `translit_table.tsv` once with `build_translit_table.py`:
     - Collect every distinct Indian-script word from S2/S3 (train + test).
     - Transliterate each word to Latin letters with AI4Bharat IndicXlit (indic→en) on the GPU. Map script to language code: Devanagari→hi, Bengali→bn, Gujarati→gu, Gurmukhi→pa, Kannada→kn, Malayalam→ml, Odia→or, Tamil→ta, Telugu→te.
     - IndicXlit's `fairseq` dependency may fail on Python 3.11. If so, run this one script in a separate Python 3.9/3.10 environment. Do not modify the main venv for it.
     - Fall back to `indic-transliteration` (rule-based) if IndicXlit is unavailable.
     - **Snap** each result to the closest word in the Latin vocabulary of the same country (built from S1+S2+S3). Use consonant-skeleton match + rapidfuzz Jaro-Winkler ≥ 0.85; otherwise keep the transliteration. Example: "piraivet" → "private".
   - Main pipeline:
     - Indian-script words → look up in the table (a join).
     - Other non-Latin scripts → `anyascii`.
     - Latin words → NFKD, strip accents, `º` → `o`.
   - Handle mixed-script names word by word.

4. **Normalize** (on distinct strings). Produce these columns:
   - `name_key`: lowercase, no punctuation, collapsed spaces. Change 0→o and 1→l only inside words that are otherwise letters. Remove legal suffixes: pvt, private, ltd, limited, llp, llc, inc, corp, co, sarl, sas, sa, eurl, and DBA prefixes. Split `.com`/`www` names.
   - `addr_key`: expand abbreviations (rd→road, st→street, ave→avenue, bd→boulevard, av→avenue). Drop `null`/`NULL`.
   - `city`: last address parts before the state.
   - `house_no`: first number-like token.
   - `skeleton`: per name word, w→v, c→k, ph→f, sh→s, doubled letters collapsed, vowels dropped except the first.
   - `initials`: first letter of each word in `name_key`.
   - Prefer native polars string expressions (GPU-capable).

5. **Split by country** using the distinct values in the data (dynamic).

6. **Blocking passes** for each country. The index is S2+S3; run one search per S1 entity_id.
   - A. Character 3-gram TF-IDF on `name_key` (CPU `sparse_dot_topn`, chunked).
   - B. Hash join on `skeleton` words.
   - C. Inverted index of rare name and city words (appearing in < ~500 rows).
   - D. Hash join on `(city, house_no)` and `(city, initials)`.
   - No Python loops over rows.

7. **Combine** (lazy, streaming).
   - Union all passes and score each candidate with `max(name_cos, addr_cos)`. Compute `addr_cos` only for pairs that were generated.
   - Keep candidates with score ≥ max(floor, best − margin), capped at 50.
   - Record which pass(es) found each pair.

8. **Write output.** `output/candidate_pairs.tsv` with columns `source1_entity_id`, `candidate_entity_ids`:
   - one row for EVERY S1 id (left-join onto the full S1 list);
   - comma-separated, deduplicated, S2/S3 ids only;
   - empty when no candidates;
   - no quoting.

## Evaluation (train only, in a separate `evaluate_blocking.py`)
Report overall and per country:
- pair recall: share of true pairs present in the candidates;
- share of S1 entities whose true matches are all present;
- average and p95 candidates per S1;
- reduction ratio versus the full cross join within each country;
- recall gain from each pass;
- time, peak RAM, peak VRAM and engine used, for each stage.

Target: ≥ 97% pair recall with ≤ 50 candidates per S1, and fewer on average.

## Deliverables
- Scripts for each stage plus a single CLI:
  `python -m src.run_blocking --data-dir ... --split train|test --out ... --polars-engine gpu|cpu --n-threads 16 --vram-budget-gb 9 --ram-budget-gb 16`
- `build_translit_table.py` and the generated `translit_table.tsv`.
- A short report: the load sanity checks, evaluation numbers, the engine benchmark, and the settings used (K, floor, margin, max_df, key caps, chunk sizes).

Start by building and running only the loading stage for all three train files, and report the sanity checks. Then develop the rest on a 5% sample of S1 (seed 42), then run on the full data. Do not build the matching model yet.
````
