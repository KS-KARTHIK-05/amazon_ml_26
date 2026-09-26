"""Cross-encoder pair scorer for the uncertain band.

Stage 1 is between 1% and 99% sure on only ~3% of candidate pairs, yet those
pairs hold ~92% of the matcher's errors (held-out). A transformer that reads
both records' text together can see what ~50 hand-made similarity numbers
cannot (a renamed business at the same address, a website-style name, a
spelling variant vs a different shop in the same building). Its score is a
stage-2 feature, computed only for band pairs.

Model: any MIT/Apache encoder with a sequence-classification head, e.g.
intfloat/multilingual-e5-small (MIT, 118M) or FacebookAI/xlm-roberta-base
(MIT, 279M). Input: "<name> ; <address>" for the S1 record and the candidate,
normalized text (transliterated, abbreviations expanded).
"""

from __future__ import annotations

import math
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MAX_LEN = 128
BAND = (0.01, 0.99)


TEXT_COLS = {"v1": ["name_norm", "addr_norm"], "v2": ["name_norm", "addr_norm", "business_name"]}


def record_text(rec: pl.DataFrame, fmt: str = "v1") -> pl.DataFrame:
    """entity_id -> text. v1: 'name ; address' (normalized). v2 appends the raw name
    (last, so truncation drops it first): legal forms / casing the normalized view
    removes still separate US look-alikes (audit: raw-name AUC 0.937 vs 0.933)."""
    t = pl.col("name_norm").fill_null("") + " ; " + pl.col("addr_norm").fill_null("")
    if fmt == "v2":
        t = t + " ; " + pl.col("business_name").fill_null("")
    return rec.select("entity_id", t.alias("text"))


def text_format(model_dir: str) -> str:
    import json
    from pathlib import Path
    meta = Path(model_dir) / "ce_meta.json"
    return json.loads(meta.read_text())["text_format"] if meta.exists() else "v1"


def attach_text(pairs: pl.DataFrame, text: pl.DataFrame) -> pl.DataFrame:
    return (pairs.join(text.rename({"entity_id": "s1_id", "text": "ta"}), on="s1_id", how="left")
            .join(text.rename({"entity_id": "cand_id", "text": "tb"}), on="cand_id", how="left")
            .with_columns(pl.col("ta").fill_null(""), pl.col("tb").fill_null("")))


class _Collate:
    def __init__(self, tok, with_labels: bool):
        self.tok, self.with_labels = tok, with_labels

    def __call__(self, batch):
        a, b = [x[0] for x in batch], [x[1] for x in batch]
        enc = self.tok(a, b, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        if self.with_labels:
            enc["labels"] = torch.tensor([x[2] for x in batch], dtype=torch.float32)
        return enc


def _rows(df: pl.DataFrame, with_labels: bool) -> list:
    cols = [df["ta"].to_list(), df["tb"].to_list()] + ([df["label"].cast(pl.Float32).to_list()] if with_labels else [])
    return list(zip(*cols))


def train(model_name: str, tr: pl.DataFrame, va: pl.DataFrame, out_dir: str, epochs: int = 1, batch: int = 64,
          lr: float = 3e-5, log_every: int = 500) -> None:
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(dev)
    dl = DataLoader(_rows(tr, True), batch_size=batch, shuffle=True, num_workers=4, collate_fn=_Collate(tok, True),
                    persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, step, run = time.time(), 0, 0.0
    for _ in range(epochs):
        for enc in dl:
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
            y = enc.pop("labels")
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(**enc).logits.squeeze(-1)
            loss = loss_fn(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % log_every == 0 or step == steps:
                el = time.time() - t0
                print(f"    ce step {step}/{steps}  loss {run:.4f}  {step * batch / el:.0f} pairs/s  "
                      f"eta {el / step * (steps - step) / 60:.0f} min", flush=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    if va is not None and va.height:
        p = score(out_dir, va, model=model, tok=tok)
        y = va["label"].to_numpy()
        eps = 1e-6
        ll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        print(f"    ce valid logloss {ll:.4f} on {va.height} pairs", flush=True)


@torch.no_grad()
def score(model_dir: str, pairs: pl.DataFrame, batch: int = 512, model=None, tok=None) -> np.ndarray:
    """P(match) for pairs with columns ta, tb."""
    dev = torch.device("cuda")
    tok = tok or AutoTokenizer.from_pretrained(model_dir)
    model = model or AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev)
    model.eval()
    # length-sorted batches: far less padding
    order = np.argsort((pairs["ta"].str.len_chars() + pairs["tb"].str.len_chars()).to_numpy())
    rows = _rows(pairs[order], False)
    dl = DataLoader(rows, batch_size=batch, shuffle=False, num_workers=4, collate_fn=_Collate(tok, False))
    out = []
    t0 = time.time()
    for i, enc in enumerate(dl):
        enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy())
        if i and i % 500 == 0:
            print(f"    ce score {i * batch}/{pairs.height}  {i * batch / (time.time() - t0):.0f} pairs/s", flush=True)
    p = np.empty(pairs.height, dtype=np.float32)
    p[order] = np.concatenate(out) if out else np.empty(0, dtype=np.float32)
    model.train()
    return p


def n_steps(n: int, batch: int, epochs: int = 1) -> int:
    return epochs * math.ceil(n / batch)
