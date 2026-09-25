# Model Way: Entity Resolution with Pretrained NLP Models and LLMs

A plan for solving the Business Entity Resolution challenge mainly with pretrained models from Hugging Face. Every model runs locally, and each one has at most 8B parameters and an MIT or Apache-2.0 license.

---

## 1. The challenge in brief

- **Input:** three TSV files of business records: `entity_id`, `business_name`, `business_address`, `country`.
  - S1 is the deduplicated reference: 2.2M rows in train, 1.7M in test.
  - S2 and S3 are noisy copies: about 10M rows combined, per split.
- **Task:** for every S1 entity, output all S2/S3 records that refer to the same business. An empty list is a valid answer.
- **Metric:** macro F0.5, computed for each S1 entity and then averaged.
  - Precision counts about twice as much as recall.
  - A singleton (an S1 entity with no match) scores **1.0 if you predict an empty list** and 0.0 if you predict anything.
- **Two output files:**
  - `matching_results.tsv`: the only file scored on the leaderboard.
  - `candidate_pairs.tsv`: the blocking output. Every predicted match must also appear in it.
- **Rules:**
  - The final model must be MIT or Apache-2.0 and at most 8B parameters.
  - **No external APIs, lookups or internet data.** Local pretrained weights are fine.
  - The test set contains **France**, which does not appear in train. Treat country as an open set of labels.

### Data facts already verified
| Fact | Why it matters |
|---|---|
| S1 is 100% Latin script | Every comparison ends up in Latin letters |
| 9 Indian scripts appear only in S2/S3 India rows (Devanagari, Bengali, Gujarati, Gurmukhi, Kannada, Malayalam, Odia, Tamil, Telugu) | A multilingual model or transliteration is needed |
| Only **~1,553 distinct Indian-script words** across all S2/S3 train+test files | Transliteration is a tiny one-off job, even for a 7B LLM |
| French rows are Latin with accents and `º` | No transliteration needed; strip accents |
| In train, no S2/S3 id is linked to more than one S1 | Each S2/S3 record can be given to at most one S1 |
| ~123k train S1 entities (5.6%) are singletons | Strict thresholds pay off |
| A matched S1 has on average ~3.7 matches | Candidate lists can stay short |

### What LLMs can and can't do here
| Task | Pure LLM? | Best model-based way |
|---|---|---|
| Transliteration | ✅ Excellent, and only ~1.5k words | Run an LLM or IndicXlit once over the distinct words |
| Blocking (finding candidates) | ❌ An LLM can't search 10M records; about 17 trillion possible pairs | **Embedding model + vector index (FAISS)**, which is also a neural model |
| Matching (yes/no per pair) | ⚠️ Accurate but slow: tens of millions of pairs would take days | **Fine-tuned cross-encoder** for all pairs, **LLM judge** only for uncertain pairs |

So the model approach is a **retrieve → rerank → judge** chain, with a model at every stage.

---

## 2. Pipeline

```
Raw TSVs
 │ 1. Load (polars) → Parquet, int32 ids, keep raw text
 │ 2. Transliteration table (LLM + IndicXlit) → replace Indian-script words with Latin
 │ 3. Build one text string per record: "name | address | country"
 │ 4. BLOCKING: fine-tuned multilingual embedding model → FAISS-GPU top-K per S1
 │              (+ optional char TF-IDF safety net, unioned in)
 │    → candidate_pairs.tsv
 │ 5. MATCHING-1: fine-tuned cross-encoder scores every (S1, candidate) pair → probability
 │ 6. MATCHING-2: LLM judge (Qwen2.5-7B + LoRA) re-checks only uncertain pairs
 │ 7. DECISION: threshold tuned for macro F0.5 + give each S2/S3 to one S1 only
 │    → matching_results.tsv
```

---

## 3. Models (all MIT or Apache-2.0, all ≤ 8B)

### Recommended stack
| Stage | Model (Hugging Face id) | Size | License | Role |
|---|---|---|---|---|
| Transliteration | `Qwen/Qwen2.5-7B-Instruct` | 7.6B | Apache-2.0 | Turns ~1.5k Indian-script words into their **English spelling** (प्राइवेट → private, केयर → care) |
| Transliteration (cross-check) | AI4Bharat **IndicXlit** (pip `ai4bharat-transliteration`) | ~11M | MIT | Phonetic Latin spelling. Used where the LLM output looks doubtful |
| Blocking embeddings | `intfloat/multilingual-e5-base` | 278M | MIT | Bi-encoder for retrieval; fine-tuned on train pairs |
| Vector search | `faiss-gpu-cu12` (library) | — | MIT | Top-K nearest neighbours on the GPU |
| Cross-encoder matcher | `FacebookAI/xlm-roberta-base` (fine-tuned) | 278M | MIT | Scores each candidate pair |
| LLM judge | `Qwen/Qwen2.5-7B-Instruct` + LoRA | 7.6B | Apache-2.0 | Hard, uncertain pairs only |
| Serving | `vllm` (library) | — | Apache-2.0 | Fast batched LLM inference |

### Alternatives (same license rules)
| Use | Alternatives |
|---|---|
| Embeddings | `BAAI/bge-m3` (568M, MIT: stronger, multilingual, slower) · `intfloat/multilingual-e5-small` (118M, MIT: fastest) · `sentence-transformers/LaBSE` (471M, Apache: good across scripts) · `Qwen/Qwen3-Embedding-0.6B` (Apache) |
| Cross-encoder | `BAAI/bge-reranker-v2-m3` (568M, Apache: stronger, about 2× slower) · `google/muril-base-cased` (Apache: Indian-language focus) · `Qwen/Qwen3-Reranker-0.6B` (Apache) |
| LLM judge | `Qwen/Qwen3-4B-Instruct-2507` (4B, Apache: faster) · `microsoft/Phi-4-mini-instruct` (3.8B, MIT: weaker on Indian scripts) · `mistralai/Mistral-7B-Instruct-v0.3` (Apache: weak on Indian scripts) |

### Do NOT use
| Model | Reason |
|---|---|
| Llama 2/3/3.x | Llama community license, not MIT/Apache |
| Gemma / Gemma 2/3 | Gemma terms, not MIT/Apache |
| `Qwen2.5-3B`, `Qwen2.5-72B` | Qwen research/custom license (unlike the 0.5B/1.5B/7B/14B/32B sizes) |
| `Qwen3-8B` | ~8.2B parameters, which is over the 8B limit |
| Any hosted API (OpenAI, Claude, Gemini…) | External service, so disqualification |

Check the license on each model card before downloading, and list every model with its license in the methodology document.

---

## 4. Step-by-step

### Step 1: Load
- Read TSVs with `separator="\t"`, `quote_char=None` and every column as a string, then save as Parquet.
- Map entity_ids to int32 ids. Keep the raw text.
- **Deduplicate strings.** Embed and score each distinct string once.

### Step 2: Transliteration (one-off, minutes)
1. Collect every distinct Indian-script word from S2/S3 (train + test), about 1,553 words.
2. Ask **Qwen2.5-7B-Instruct** (vLLM, greedy decoding) for each word:
   ```
   You convert Indian-script words to how they are written in English.
   If the word is an English word written in Indian script, give the English spelling.
   If it is an Indian name or place, give the common English spelling.
   Script: Devanagari. Word: प्राइवेट
   Answer with the word only.
   ```
3. Also run IndicXlit (indic→en). If the two differ, keep the LLM answer when it is a word that occurs in the Latin vocabulary of the same country. Otherwise keep the IndicXlit answer.
4. Save the result as `translit_table.tsv` and review it by eye. With 1.5k rows this is realistic.
5. Replace the words word by word, so mixed-script names like `Swastik কেয়ার` also work.
   - French and English need no transliteration: lowercase, strip accents, `º` → `o`.

### Step 3: Record text
Make one short string per record, the same way for all three sources:
```
"<name> | <address> | <country>"
```
Keep the country in the text. The model learns that the country must agree, and France works without special handling.

### Step 4: Blocking with fine-tuned embeddings + FAISS
1. **Fine-tune** `multilingual-e5-base` with `sentence-transformers`:
   - Training pairs: (S1 text, matched S2/S3 text) from `train_ground_truth.tsv`. Sample about 300k–1M positives.
   - Loss: `MultipleNegativesRankingLoss`, which uses the other pairs in the batch as negatives. Optionally add **hard negatives** found by the untuned model: same name, different address.
   - e5 models expect prefixes: `"query: "` for S1 and `"passage: "` for S2/S3.
   - Use about 1 epoch, fp16, max_len 64–96, batch 128–256.
2. **Encode** all distinct S1 and S2/S3 strings (fp16, batch 512–1024). Normalize the vectors.
3. **Index per country** with FAISS-GPU: `IndexFlatIP` if it fits in VRAM, otherwise `IndexIVFFlat`/`IVFPQ`.
4. **Search** top-K = 20–50 per S1, in chunks of S1 queries.
5. **Recommended safety net:** union with a cheap character-trigram TF-IDF top-K. It catches typos and codes that embeddings sometimes miss (`Metr0politan`, `#13347`).
6. Write `candidate_pairs.tsv`. On a held-out train split, **measure recall** (the share of true pairs present) and average candidates per S1. Target ≥ 97% recall.

Rough cost on an RTX 3060-class GPU: about 12M distinct strings per split with a 278M model in fp16 is on the order of an hour of encoding. FAISS search is minutes. **Benchmark on 100k strings first** and switch to `multilingual-e5-small` if it is too slow.

### Step 5: Cross-encoder matcher (the main accuracy engine)
1. Build training pairs **from the Step 4 candidates on train**. Label 1 if the pair is in the ground truth, 0 otherwise. These are realistic hard negatives.
2. Input: `"[S1] name | address | country"` and `"[CAND] name | address | country"`, max_len 128.
3. Fine-tune `xlm-roberta-base` as a sequence-pair classifier (binary cross-entropy, 1–2 epochs, fp16).
4. Score every candidate pair → probability p.
   - To save time, cross-score only the top 10–20 candidates by embedding score.
5. Optional boost: feed p together with a few cheap features into a small LightGBM model, which usually improves calibration. Features: rapidfuzz ratio of names, token Jaccard, house-number equal, embedding cosine.

Cost: tens of millions of pairs at max_len 128 is several GPU-hours with a base model. Keep K small, sort pairs by length so batches waste little padding, and use fp16. Benchmark first.

### Step 6: LLM judge for uncertain pairs only
- Send only pairs with p in an **uncertain band** (for example 0.3–0.7) to the LLM. That should be a small share of pairs.
- Use a **list-style prompt**: one S1 record plus its uncertain candidates, numbered. Ask for the numbers that match, so the model never writes IDs and can't invent them:
  ```
  Reference business:
  Name: Premio Pharma LLP | Address: 12 MG Road, Pune, Maharashtra | Country: India

  Candidates:
  1. PP | 12 M.G. Rd, Pune | India
  2. Premio Foods Pvt Ltd | 88 FC Road, Pune | India
  3. प्रीमियो फार्मा | 12 एमजी रोड, पुणे | India

  Which candidates are the SAME business as the reference? Different branches or
  different addresses with the same name are NOT the same unless the address agrees.
  Answer with JSON: {"match": [numbers]}
  ```
- **Fine-tune with LoRA** (PEFT/QLoRA, 4-bit) on train examples built the same way. This matters much more than prompt wording. A 7B QLoRA fit on 12 GB is tight: use a short context, or use `Qwen3-4B-Instruct-2507`.
- Serve with vLLM (AWQ/4-bit if VRAM is limited) and use constrained JSON output.

### Step 7: Decision rules (where F0.5 is won)
1. **Threshold:** predict a match only if the final probability ≥ t. Tune t by **directly maximizing macro F0.5** on the held-out split, singletons included. Expect t to be high, around 0.6–0.8.
2. **One owner per S2/S3:** if a record passes the threshold for several S1 rows, keep only the highest-scoring one.
3. **Singletons:** if nothing passes, output an empty list. That is worth a full 1.0 for true singletons.
4. Only predict IDs that are in `candidate_pairs.tsv`.
5. Run `student_resource/utils/validate_submission.py` before every upload.

---

## 5. Validation
- Hold out about 10% of **train S1 entities** (group split: an S1 and all its matches stay on the same side). Include singletons.
- Report on the held-out set: blocking recall, macro F0.5, precision, recall, singleton accuracy, and F0.5 per country.
- The leaderboard's public part is a subset. Trust the held-out F0.5 over single leaderboard jumps.

## 6. Timeline for the remaining ~60 hours
| Hours | Work |
|---|---|
| 0–3 | Loading, transliteration table, held-out split, metric script |
| 3–10 | Fine-tune the embedding model, encode, FAISS blocking, measure recall |
| 10–20 | Build cross-encoder training pairs, fine-tune, score held-out set, tune threshold → **first leaderboard submission** |
| 20–35 | LoRA LLM judge for the uncertain band; check the F0.5 gain on held-out |
| 35–50 | Full test run (embedding → cross-encoder → judge), error analysis, second submission |
| 50–60 | Freeze, reproducibility README, requirements, methodology document |

Submit something at around hour 20, even without the LLM judge. Every later stage should only add to a working baseline.

## 7. Hardware notes
- Minimum: one GPU with 12 GB (RTX 3060 class). Everything above fits with fp16 inference, 4-bit LLM serving and QLoRA. A 16–24 GB GPU makes the 7B LoRA and cross-encoder training much easier.
- CPU RAM ≥ 32 GB for the 10M-record tables and the FAISS index construction.
- Encode and score in chunks and write every chunk to Parquet, so a crash never loses hours of GPU work.

## 8. Working with the other approach
The other teammate builds a classic blocking engine (character TF-IDF + key matching, CPU/Polars GPU) with the **same `candidate_pairs.tsv` format**. Agree on:
- the **same held-out split** (same S1 ids, seed 42), so scores are comparable;
- sharing candidate files. The **union** of both blockers gives higher recall for either matcher;
- a final ensemble: average the two matchers' probabilities per pair before thresholding. That usually beats either one alone.
