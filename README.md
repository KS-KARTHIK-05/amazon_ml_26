# Business Entity Resolution — Amazon ML Challenge 2026

For every Source-1 business, find its matching records in Source 2 and Source 3 — or say "no match". Scored with macro F0.5, so a wrong merge hurts about twice as much as a missed one.

**Current result:** held-out macro F0.5 **0.953** (precision 0.986, recall 0.899), and the generated submission passes `validate_submission.py`.

## How it works

```
S1 + S2 + S3 TSVs
   │  1. master table         one table per split, ground-truth cluster + fold attached
   │  2. transliteration      Indic words -> English, table learned from training pairs
   │  3. normalization        country-aware names & addresses (US / India / France)
   │  4. blocking             inverted-index key overlap  ->  ~52 candidates per S1
   │  5. pair features        fuzzy name/address similarity + blocking scores + per-S1 ranks
   │  6. LightGBM (GPU)       match probability per candidate pair
   │  7. decision             threshold tuned for macro F0.5, each S2/S3 goes to one S1
   ▼
output/matching_results.tsv  +  output/candidate_pairs.tsv
```

A few things worth knowing:

- **Transliteration matters a lot.** S1 is all Latin, but ~40% of India's S2/S3 rows use one of 9 Indian scripts. There are only ~1,500 distinct Indic words, and every test word also shows up in train — so instead of a heavy model, we learn each word's English spelling from the matched S1 record (`प्राइवेट → private`, `தமிழ்நாடு → tamil nadu`, even `পশ্চিমবঙ্গ → west bengal`). This lifted India's name overlap from 76% to 92%.
- **Normalization is per country.** State codes vs names (`Iowa`↔`IA`, `GJ`↔`Gujarat`), street abbreviations (`BLVD`, `Rd`, French `R.`/`Bd`), legal forms, honorifics, digit-for-letter swaps (`m0tors`), old city names. Country is treated as an open set, so France works without special casing.
- **Blocking uses keys, not a giant similarity matrix.** Name tokens, house numbers, address components, and name×locality conjunctions. Candidates are picked from four separate rankings (overall cosine, name score, address score, conjunction score) so a good name match can't be crowded out by neighbours at the same address. F0.5 ceiling on held-out: 0.984 (India) / 0.987 (US).
- **The matcher looks at candidates relative to each other.** Rank and gap-to-best within an S1's candidate list are among the strongest features — that's what keeps precision high.

## Layout

```
src/
  data/        load.py (TSV -> parquet), split.py (grouped 90/10 split), master.py
  preprocess/  translit.py, lexicons.py, normalize_master.py
  blocking/    blocker.py, ngram_tfidf.py, key_blocker.py (stopword lists)
  matching/    pair_features.py
  decision/    writer.py (TSV output + validator wrapper)
  eval/        f05_score.py, blocking_recall.py
  run_submission.py   end-to-end driver
```

## Running it

The code expects the challenge folder layout: this folder at `student_resource/code/business_entity_resolution/`, with `dataset/` and `utils/validate_submission.py` in `student_resource/`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 -m src.data.load train && python3 -m src.data.load test
python3 -m src.data.split
python3 -m src.data.master train && python3 -m src.data.master test
python3 -m src.preprocess.translit
python3 -m src.preprocess.normalize_master train && python3 -m src.preprocess.normalize_master test

python3 -u -m src.run_submission all     # block -> features -> train -> predict -> validate
```

Each stage caches its output, so a crash only costs that stage. The full run takes about 1.5 hours on a 20-core machine with an RTX 3060 and peaks around 15 GB of RAM. `python3 -m src.eval.blocking_recall --sample 20000` measures blocking recall on held-out.

## What's next

- The matcher hit its 2,000-round cap while still improving — retrain with a higher learning rate.
- Re-enable the character-trigram name TF-IDF source in blocking to catch typo'd names.
- France has no training labels, so its accuracy is only visible on the leaderboard.
