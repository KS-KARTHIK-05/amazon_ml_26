#!/usr/bin/env python3
"""Sample S1 records and all their ground-truth matches using only stdlib.

Run: python3 sample_dataset.py --seed 42
Outputs default to student_resource/dataset/sampled/. Omit --seed for a
fresh random sample. Empty match lists and original text fields are preserved.
"""

import argparse
import csv
from pathlib import Path
import random


def read_rows(path, required):
    """Yield the header, then rows without loading the entire dataset."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames
        if not fields or not set(required).issubset(fields):
            raise ValueError(f"{path}: required columns missing: {required}")
        yield fields
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: malformed row at line {reader.line_num}")
            yield row


def select_rows(path, key, wanted, required=()):
    reader = read_rows(path, (key, *required))
    fields = next(reader)
    found = {}
    for row in reader:
        entity_id = row[key]
        if entity_id in wanted:
            if entity_id in found:
                raise ValueError(f"{path}: duplicate ID {entity_id}")
            found[entity_id] = row
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"{path}: {len(missing)} missing IDs: {sorted(missing)[:5]}")
    return fields, found


def write_rows(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sample_dataset(train_dir, output_dir, size=50, seed=None):
    if size < 1:
        raise ValueError("Sample size must be positive")
    rng = random.Random(seed)
    reader = read_rows(train_dir / "train_source1.tsv", ("entity_id",))
    source1_fields = next(reader)
    # Reservoir sampling gives each S1 row equal probability using O(size) memory.
    sampled = []
    for index, row in enumerate(reader):
        if index < size:
            sampled.append(row)
        else:
            slot = rng.randrange(index + 1)
            if slot < size:
                sampled[slot] = row
    if len(sampled) != size:
        raise ValueError(f"Requested {size} samples, but S1 has only {len(sampled)} rows")
    rng.shuffle(sampled)
    source1_ids = {row["entity_id"] for row in sampled}
    if len(source1_ids) != size:
        raise ValueError("Sampled source 1 records contain duplicate entity IDs")

    truth_fields, truth = select_rows(
        train_dir / "train_ground_truth.tsv", "source1_entity_id", source1_ids,
        required=("matched_entity_ids",),
    )
    truth_rows = [truth[row["entity_id"]] for row in sampled]
    matched = {"S2": set(), "S3": set()}
    for row in truth_rows:
        for value in row["matched_entity_ids"].split(","):
            entity_id = value.strip()
            if not entity_id:
                continue
            prefix = entity_id.split("-", 1)[0]
            if prefix not in matched:
                raise ValueError(f"Unexpected matched entity ID: {entity_id}")
            matched[prefix].add(entity_id)

    outputs = {
        "sampled_truth.tsv": (truth_fields, truth_rows),
        "sampled_source1.tsv": (source1_fields, sampled),
    }
    for number in (2, 3):
        fields, records = select_rows(
            train_dir / f"train_source{number}.tsv", "entity_id", matched[f"S{number}"]
        )
        outputs[f"sampled_source{number}.tsv"] = (fields, list(records.values()))

    # Validate every reference before writing any output.
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, (fields, rows) in outputs.items():
        path = output_dir / filename
        write_rows(path, fields, rows)
        print(f"{path}: {len(rows)} records")


def main():
    dataset = Path(__file__).resolve().parent / "student_resource" / "dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=dataset / "train")
    parser.add_argument("--output-dir", type=Path, default=dataset / "sampled")
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    try:
        sample_dataset(args.train_dir, args.output_dir, args.size, args.seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
