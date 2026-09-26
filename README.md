# Output sets (submission files)

| Folder | Pipeline version | Held-out macro F0.5 | Precision | Recall | Singleton acc | Public LB |
|---|---|---|---|---|---|---|
| `v4ce2/`  | v4 (forward + wider reverse blocking) + e5-small cross-encoder v2 | 0.9862 | 0.9983 | 0.9615 | 0.9946 | 0.97889 |
| `v4ce2x/` | v4ce2 + xlm-roberta-base cross-encoder score (`ce_x`)             | 0.9863 | 0.9986 | 0.9606 | 0.9951 | – |

Both passed `utils/validate_submission.py`. `models/` holds each run's stage-2 LightGBM
and its state (threshold) file.

GitHub rejects files over 100 MB, so the TSVs are zstd-compressed and the candidate
file is split into parts. Restore (byte-identical to the originals):

```bash
cd v4ce2x            # or v4ce2
zstd -d matching_results.tsv.zst -o matching_results.tsv
cat candidate_pairs.tsv.zst.part* | zstd -d -o candidate_pairs.tsv
```
(`sudo apt install zstd` if the command is missing.)
