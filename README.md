# Output sets (submission files)

| Folder | Pipeline version | Held-out macro F0.5 | Precision | Recall | Singleton acc | Public LB |
|---|---|---|---|---|---|---|
| `v4ce2/`  | v4 (forward + wider reverse blocking) + e5-small cross-encoder v2 | 0.9862 | 0.9983 | 0.9615 | 0.9946 | 0.97889 |
| `v4ce2x/` | v4ce2 + xlm-roberta-base cross-encoder score (`ce_x`)             | 0.9863 | 0.9986 | 0.9606 | 0.9951 | – |
| `v5/`     | v4ce2x + exact-name blocking channel (identical / space-free core name shared by <= 3 S1s) | **0.9874** | 0.9983 | **0.9649** | 0.9942 | – |

Held-out = 220,682 source-1 entities never used in training. v5 per country: India 0.9866, US 0.9879;
blocking recall 0.9805, blocking ceiling 0.9938, 75.8 candidates per S1.

All outputs passed `utils/validate_submission.py`; v5 also passed it with `--check-ids`
(every matched / candidate ID exists in the test set). `models/` holds each run's stage-2 LightGBM
and its state (threshold) file; `v5/models/` also has both stage-1 models.

GitHub rejects files over 100 MB, so the TSVs are zstd-compressed and the candidate
file is split into parts. Restore (byte-identical to the originals):

```bash
cd v5            # or v4ce2x / v4ce2
zstd -d matching_results.tsv.zst -o matching_results.tsv
cat candidate_pairs.tsv.zst.part* | zstd -d -o candidate_pairs.tsv
```
(`sudo apt install zstd` if the command is missing.)
