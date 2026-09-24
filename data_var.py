#!/usr/bin/env python3
"""Audit linked entity variations and export a diverse, ground-truth-complete sample.

Python standard library only. Example: python3 data_var.py --seed 42
Randomly audits 50,000 S1 entities across the FULL training file, then selects
50 entities for variation coverage. Counts of variations describe that audit,
not the whole dataset. All source files are streamed for country counts.
Labels are overlapping heuristics, not verified linguistic/geographic facts.
"""

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import random
import re
import unicodedata

from sample_dataset import read_rows, select_rows, write_rows


SOURCE_FIELDS = ("entity_id", "business_name", "business_address", "country")
STATE_PAIRS = """AL:Alabama|AK:Alaska|AZ:Arizona|AR:Arkansas|CA:California|
CO:Colorado|CT:Connecticut|DE:Delaware|DC:District of Columbia|FL:Florida|
GA:Georgia|HI:Hawaii|ID:Idaho|IL:Illinois|IN:Indiana|IA:Iowa|KS:Kansas|
KY:Kentucky|LA:Louisiana|ME:Maine|MD:Maryland|MA:Massachusetts|MI:Michigan|
MN:Minnesota|MS:Mississippi|MO:Missouri|MT:Montana|NE:Nebraska|NV:Nevada|
NH:New Hampshire|NJ:New Jersey|NM:New Mexico|NY:New York|NC:North Carolina|
ND:North Dakota|OH:Ohio|OK:Oklahoma|OR:Oregon|PA:Pennsylvania|RI:Rhode Island|
SC:South Carolina|SD:South Dakota|TN:Tennessee|TX:Texas|UT:Utah|VT:Vermont|
VA:Virginia|WA:Washington|WV:West Virginia|WI:Wisconsin|WY:Wyoming"""
STATES = dict(part.strip().split(":") for part in STATE_PAIRS.split("|"))
STATE_LOOKUP = {v.casefold(): k for k, v in STATES.items()}
STATE_LOOKUP.update({k.casefold(): k for k in STATES})
LEGAL = set("inc incorporated llc ltd limited pvt private corp corporation co company llp lp pllc pc".split())
ABBREV = dict(zip("rd st ave blvd dr ln ct hwy apt fl ste pvt ltd corp inc co".split(),
                  "road street avenue boulevard drive lane court highway apartment floor suite private limited corporation incorporated company".split()))
ADDRESS_ABBREV = {**ABBREV, **dict(zip("n s e w ne nw se sw".split(), "north south east west northeast northwest southeast southwest".split()))}
DEFINITIONS = {
    "name_case_only": "Different raw names, identical after case folding.",
    "name_punctuation_spacing": "Case-folded names differ but alphanumeric token sequences agree.",
    "name_accent_change": "Token sequences differ but agree after Unicode accent removal.",
    "name_script_change_possible_transliteration": "Unicode letter scripts differ; may be transliteration or translation, not verified.",
    "name_mixed_scripts": "Matched name contains letters from multiple scripts.",
    "name_legal_suffix_change": "Legal-form token counts differ; may overlap other name changes.",
    "name_abbreviation_equivalent": "Name tokens become equal under the small documented abbreviation map.",
    "name_word_reordering": "Same token multiset but different sequence.",
    "name_repeated_word_added": "A matched-name token of at least two characters repeats more often than in S1; single-letter initial fragments are excluded.",
    "name_token_addition_or_removal": "One name's token multiset is a strict subset of the other.",
    "name_possible_typo": "Same-script nonidentical normalized names have character similarity >= 0.72; excludes pure reorder/subset cases.",
    "name_digit_letter_substitution": "A differing token agrees after 0->o, 1->i, 3->e, 4->a, 5->s, 7->t substitution.",
    "name_url_or_handle": "Changed name includes a URL, .com/.in domain, or hashtag marker.",
    "name_numeric_identifier_added": "Matched name has a standalone digit sequence absent from S1; possible appended identifier.",
    "name_joined_or_split_words": "Name token sequences differ but concatenating their normalized tokens yields identical text.",
    "name_initialism_candidate": "Matched name reduces to one 2-8 character token equal to the initials of S1 tokens after removing legal-form tokens.",
    "name_honorific_added": "Matched name adds Mr/Mrs/Ms/Dr/Shri/Sri tokens absent from S1.",
    "name_trade_alias_marker": "Changed name includes DBA, formerly, or trading-as text.",
    "name_low_lexical_overlap": "Nonempty normalized names share at most 25% of their token union.",
    "name_missing": "One name is empty or a whole-field null marker.",
    "address_missing": "One address is empty or a whole-field null marker.",
    "address_case_punctuation_spacing": "Different raw addresses have the same alphanumeric token sequence.",
    "address_script_change": "Address Unicode letter scripts differ; not a verified transliteration.",
    "address_abbreviation_or_state_expansion": "Raw address token multiset differs but agrees after abbreviation/US-state normalization.",
    "address_component_reordering": "Normalized address tokens have the same multiset but a different sequence.",
    "address_component_addition_or_removal": "One normalized address token multiset is a strict subset of the other.",
    "address_numeric_tokens_changed": "Nonempty addresses have different digit-token multisets; may be formatting, unit, postcode, or house-number changes.",
    "address_unit_po_box_change": "Apartment/unit/suite/floor/PO-box/PMB marker counts differ.",
    "address_embedded_null_marker": "Address contains null/none/nan as a token within other text.",
    "address_landmark_marker_change": "Near/opposite/opp/behind/beside landmark-marker counts differ.",
    "address_leading_zero_number_format": "Digit-token sequences differ but agree after stripping leading zeros.",
    "address_possible_typo": "Same-script normalized addresses differ with character similarity >= 0.80, excluding identical token multisets.",
    "address_low_lexical_overlap": "Nonempty addresses share at most 25% of their normalized token union; not proof of relocation.",
    "us_state_conflict_candidate": "Each US address has one unambiguous comma-delimited state token and they disagree; review manually.",
    "us_city_conflict_candidate": "States agree, conservative city extraction succeeds, and city similarity is < 0.75; review manually.",
    "country_mismatch": "Ground-truth-linked records have different case-folded country labels.",
    "country_missing": "At least one country is empty or a whole-field null marker.",
    "unchanged_record": "Name, address, and country are exactly identical.",
    "other_text_variation": "Text differs but no other detector fired.",
    "singleton_control": "S1 has an empty ground-truth match list; retained as a control.",
}


def tokens(text):
    # Python's \w excludes combining marks. Keep them attached to Indic words.
    text = unicodedata.normalize("NFC", text).casefold()
    if text.isascii():
        return re.findall(r"[a-z0-9]+", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    return "".join(c if c.isalnum() or unicodedata.category(c).startswith("M") else " " for c in text).split()


def unaccent(text):
    # Strip Latin accents only; Indic vowel signs and viramas carry meaning.
    if text.isascii():
        return text
    result, latin_base = [], False
    for char in unicodedata.normalize("NFD", text):
        if unicodedata.category(char).startswith("M"):
            if not latin_base:
                result.append(char)
        else:
            latin_base = "LATIN" in unicodedata.name(char, "")
            result.append(char)
    return unicodedata.normalize("NFC", "".join(result))


def scripts(text):
    return {unicodedata.name(c, "UNKNOWN").split()[0] for c in text if c.isalpha()}


def empty(text):
    return text.strip().casefold() in {"", "null", "none", "nan", "n/a"}


def similarity(a, b):
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def overlap(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a or b else 1.0


def canonical_address(text, country):
    text = unaccent(text).casefold()
    if country.casefold() == "us":
        # Only normalize whole comma components; e.g. do not map "in" in prose.
        parts = text.split(",")
        text = ",".join(STATE_LOOKUP.get(p.strip(), p) for p in parts).casefold()
    return [ADDRESS_ABBREV.get(t, t) for t in tokens(text)]


def us_location(address):
    parts = [p.strip() for p in address.split(",")]
    found = []
    for i, part in enumerate(parts):
        value = re.sub(r"\s+\d{5}(?:-\d{4})?$", "", part).strip().casefold()
        if value in STATE_LOOKUP:
            found.append((i, STATE_LOOKUP[value]))
    if len(found) != 1:
        return "", ""
    i, state = found[0]
    city = ""
    if i == len(parts) - 1 and i > 0:
        candidate = parts[i - 1]
        if (not re.search(r"\d", candidate) and
                not set(tokens(candidate)) & {"road", "street", "avenue", "unit", "apt", "suite", "rd", "st", "dr", "ln", "ct", "ave", "blvd", "boulevard", "drive", "lane", "court", "hwy", "highway", "pkwy", "parkway", "way", "terrace", "circle", "pl", "place", "cove", "pass", "trail", "crescent"}):
            city = " ".join(tokens(unaccent(candidate)))
    return state, city


def classify(a, b):
    labels = set()
    name_a, name_b = a["business_name"], b["business_name"]
    ta, tb = tokens(name_a), tokens(name_b)
    na, nb = tokens(unaccent(name_a)), tokens(unaccent(name_b))
    ca, cb = Counter(na), Counter(nb)
    sa, sb = scripts(name_a), scripts(name_b)
    if empty(name_a) or empty(name_b):
        labels.add("name_missing")
    elif name_a != name_b:
        if name_a.casefold() == name_b.casefold():
            labels.add("name_case_only")
        elif ta == tb:
            labels.add("name_punctuation_spacing")
        if ta != tb and na == nb:
            labels.add("name_accent_change")
        if sa != sb:
            labels.add("name_script_change_possible_transliteration")
        if len(sb) > 1:
            labels.add("name_mixed_scripts")
        if Counter(t for t in na if t in LEGAL) != Counter(t for t in nb if t in LEGAL):
            labels.add("name_legal_suffix_change")
        if na != nb and [ABBREV.get(t, t) for t in na] == [ABBREV.get(t, t) for t in nb]:
            labels.add("name_abbreviation_equivalent")
        if ca == cb and na != nb:
            labels.add("name_word_reordering")
        if any(len(t) > 1 and count > 1 and count > ca[t] for t, count in cb.items()):
            labels.add("name_repeated_word_added")
        if ca < cb or cb < ca:
            labels.add("name_token_addition_or_removal")
        if sa == sb and ca != cb and not (ca < cb or cb < ca) and similarity(" ".join(na), " ".join(nb)) >= .72:
            labels.add("name_possible_typo")
        leet = str.maketrans("013457", "oieast")
        if any(t != u and t.translate(leet) == u.translate(leet) for t in ca.keys() - cb.keys() for u in cb.keys() - ca.keys()):
            labels.add("name_digit_letter_substitution")
        if re.search(r"www\.|https?://|\.com\b|\.in\b|#[^\W\d_]", name_b, re.I):
            labels.add("name_url_or_handle")
        if set(re.findall(r"(?<!\w)\d+(?!\w)", name_b)) - set(re.findall(r"(?<!\w)\d+(?!\w)", name_a)):
            labels.add("name_numeric_identifier_added")
        if na != nb and "".join(na) == "".join(nb):
            labels.add("name_joined_or_split_words")
        base_a, base_b = [t for t in na if t not in LEGAL], [t for t in nb if t not in LEGAL]
        if len(base_a) > 1 and len(base_b) == 1 and 2 <= len(base_b[0]) <= 8 and base_b[0] == "".join(t[0] for t in base_a):
            labels.add("name_initialism_candidate")
        if (set(nb) - set(na)) & {"mr", "mrs", "ms", "dr", "shri", "sri"}:
            labels.add("name_honorific_added")
        if re.search(r"\bdba\b|\bformerly\b|\btrading as\b", name_b, re.I):
            labels.add("name_trade_alias_marker")
        if overlap(na, nb) <= .25:
            labels.add("name_low_lexical_overlap")

    ad_a, ad_b = a["business_address"], b["business_address"]
    aa, ab = canonical_address(ad_a, a["country"]), canonical_address(ad_b, b["country"])
    ac, bc = Counter(aa), Counter(ab)
    if empty(ad_a) or empty(ad_b):
        labels.add("address_missing")
    elif ad_a != ad_b:
        if tokens(ad_a) == tokens(ad_b):
            labels.add("address_case_punctuation_spacing")
        if scripts(ad_a) != scripts(ad_b):
            labels.add("address_script_change")
        if Counter(tokens(unaccent(ad_a))) != Counter(tokens(unaccent(ad_b))) and ac == bc:
            labels.add("address_abbreviation_or_state_expansion")
        if ac == bc and aa != ab:
            labels.add("address_component_reordering")
        if ac < bc or bc < ac:
            labels.add("address_component_addition_or_removal")
        if Counter(re.findall(r"\d+", ad_a)) != Counter(re.findall(r"\d+", ad_b)):
            labels.add("address_numeric_tokens_changed")
        markers = {"unit", "apartment", "suite", "floor", "po", "box", "pmb"}
        if Counter(t for t in aa if t in markers) != Counter(t for t in ab if t in markers):
            labels.add("address_unit_po_box_change")
        if {"null", "none", "nan"} & (set(aa) | set(ab)):
            labels.add("address_embedded_null_marker")
        landmarks = {"near", "opposite", "opp", "behind", "beside"}
        if Counter(t for t in aa if t in landmarks) != Counter(t for t in ab if t in landmarks):
            labels.add("address_landmark_marker_change")
        da, db = re.findall(r"\d+", ad_a), re.findall(r"\d+", ad_b)
        if da != db and [int(v) for v in da] == [int(v) for v in db]:
            labels.add("address_leading_zero_number_format")
        if scripts(ad_a) == scripts(ad_b) and ac != bc and similarity(" ".join(aa), " ".join(ab)) >= .8:
            labels.add("address_possible_typo")
        if overlap(aa, ab) <= .25:
            labels.add("address_low_lexical_overlap")
    state_a = state_b = city_a = city_b = ""
    if a["country"] == b["country"] == "US":
        state_a, city_a = us_location(ad_a)
        state_b, city_b = us_location(ad_b)
        if state_a and state_b:
            if state_a != state_b:
                labels.add("us_state_conflict_candidate")
            elif city_a and city_b and similarity(city_a, city_b) < .75:
                labels.add("us_city_conflict_candidate")
    if a["country"].casefold() != b["country"].casefold():
        labels.add("country_mismatch")
    if empty(a["country"]) or empty(b["country"]):
        labels.add("country_missing")
    identical = all(a[k] == b[k] for k in SOURCE_FIELDS[1:])
    if identical:
        labels.add("unchanged_record")
    if not labels:
        labels.add("other_text_variation")
    return labels, {
        "name_token_jaccard": round(overlap(na, nb), 4),
        "address_token_jaccard": round(overlap(aa, ab), 4),
        "s1_us_state": state_a, "matched_us_state": state_b,
        "s1_us_city": city_a, "matched_us_city": city_b,
    }


def stream(path):
    rows = read_rows(path, SOURCE_FIELDS)
    next(rows)
    return rows


def run(args):
    rng = random.Random(args.seed)
    country_counts = {}
    source1 = []
    counts = Counter()
    path = args.dataset_dir / "train/train_source1.tsv"
    print(f"Sampling {args.audit_size:,} S1 entities from {path}", flush=True)
    for i, row in enumerate(stream(path)):
        counts[row["country"]] += 1
        if i < args.audit_size:
            source1.append(row)
        else:
            slot = rng.randrange(i + 1)
            if slot < args.audit_size:
                source1[slot] = row
    country_counts["train_source1"] = dict(counts)
    if len(source1) < args.size:
        raise ValueError("Not enough S1 entities for requested output size")
    s1 = {r["entity_id"]: r for r in source1}
    if len(s1) != len(source1):
        raise ValueError("Duplicate S1 entity IDs in audit sample")
    truth_fields, truth = select_rows(args.dataset_dir / "train/train_ground_truth.tsv",
                                      "source1_entity_id", set(s1), ("matched_entity_ids",))
    wanted = set()
    links = {}
    for sid, row in truth.items():
        ids = [v.strip() for v in row["matched_entity_ids"].split(",") if v.strip()]
        if len(ids) != len(set(ids)) or any(not v.startswith(("S2-", "S3-")) for v in ids):
            raise ValueError(f"Invalid ground-truth match list: {sid}")
        links[sid] = ids
        wanted.update(ids)
    records = {}
    for number in (2, 3):
        print(f"Scanning full training S{number} and retaining audit matches", flush=True)
        counts = Counter()
        for row in stream(args.dataset_dir / f"train/train_source{number}.tsv"):
            counts[row["country"]] += 1
            if row["entity_id"] in wanted:
                if row["entity_id"] in records:
                    raise ValueError(f"Duplicate source ID: {row['entity_id']}")
                records[row["entity_id"]] = row
        country_counts[f"train_source{number}"] = dict(counts)
    if wanted - records.keys():
        raise ValueError(f"Missing ground-truth references: {sorted(wanted - records.keys())[:5]}")

    if not args.skip_test_census:
        for number in (1, 2, 3):
            path = args.dataset_dir / f"test/test_source{number}.tsv"
            if path.exists():
                print(f"Counting country labels in {path.name}", flush=True)
                country_counts[f"test_source{number}"] = dict(Counter(r["country"] for r in stream(path)))

    print("Classifying linked pairs (heuristics, no external data)", flush=True)
    pair_counts, entity_counts, diagnostics = Counter(), Counter(), Counter()
    per_entity, pairs, example_pairs = {}, defaultdict(list), {}
    for index, (sid, a) in enumerate(s1.items()):
        labels_for_entity = set()
        if not links[sid]:
            labels_for_entity.add("singleton_control")
        for mid in links[sid]:
            b = records[mid]
            labels, metrics = classify(a, b)
            pair_counts.update(labels)
            labels_for_entity.update(labels)
            row = {"source1_entity_id": sid, "matched_entity_id": mid, "variation_labels": ";".join(sorted(labels))}
            for prefix, record in (("s1", a), ("matched", b)):
                row.update({f"{prefix}_{key}": record[key] for key in SOURCE_FIELDS[1:]})
            row.update(metrics)
            pairs[sid].append(row)
            for label in sorted(labels):
                example_pairs.setdefault(label, row)
            diagnostics["linked_pairs"] += 1
            diagnostics["us_linked_pairs"] += a["country"] == b["country"] == "US"
            diagnostics["us_pairs_with_both_states_parsed"] += bool(metrics["s1_us_state"] and metrics["matched_us_state"])
            diagnostics["us_pairs_with_both_cities_parsed_same_state"] += bool(metrics["s1_us_city"] and metrics["matched_us_city"] and metrics["s1_us_state"] == metrics["matched_us_state"])
            diagnostics["exact_same_name_pairs"] += a["business_name"] == b["business_name"]
            diagnostics["exact_same_name_different_raw_address_pairs"] += a["business_name"] == b["business_name"] and a["business_address"] != b["business_address"]
        per_entity[sid] = labels_for_entity
        entity_counts.update(labels_for_entity)
        if (index + 1) % 10000 == 0:
            print(f"  Classified {index + 1:,} S1 entities", flush=True)

    # Find name collisions against ALL S1 rows for names appearing in the audit.
    # These are distinct reference IDs, not positive links between those IDs.
    print("Checking reused business names across distinct S1 entities", flush=True)
    name_anchors = {}
    for sid in sorted(s1):
        name_anchors.setdefault((s1[sid]["country"], tuple(tokens(unaccent(s1[sid]["business_name"])))), s1[sid])
    collisions = []
    collision_counts = Counter()
    for row in stream(args.dataset_dir / "train/train_source1.tsv"):
        key = (row["country"], tuple(tokens(unaccent(row["business_name"]))))
        anchor = name_anchors.get(key)
        if not anchor or anchor["entity_id"] == row["entity_id"]:
            continue
        if canonical_address(anchor["business_address"], anchor["country"]) == canonical_address(row["business_address"], row["country"]):
            continue
        collision_counts["additional_s1_rows_with_audited_name_and_different_normalized_address"] += 1
        state_a, _ = us_location(anchor["business_address"]) if row["country"] == "US" else ("", "")
        state_b, _ = us_location(row["business_address"]) if row["country"] == "US" else ("", "")
        state_conflict = bool(state_a and state_b and state_a != state_b)
        collision_counts["different_us_state_candidates"] += state_conflict
        item = {"anchor_s1_id": anchor["entity_id"], "other_s1_id": row["entity_id"],
                "anchor_name": anchor["business_name"], "other_name": row["business_name"],
                "anchor_address": anchor["business_address"], "other_address": row["business_address"],
                "country": row["country"], "anchor_us_state": state_a, "other_us_state": state_b}
        if len(collisions) < 100 or (state_conflict and len(collisions) < 200):
            collisions.append(item)

    # Cover the rarest observed categories first, twice when size allows.
    chosen, selected = [], set()
    for quota in (1, 2):
        for label in sorted(entity_counts, key=lambda x: (entity_counts[x], x)):
            if len(chosen) >= args.size:
                break
            if sum(label in per_entity[sid] for sid in chosen) >= quota:
                continue
            candidates = [sid for sid in s1 if sid not in selected and label in per_entity[sid]]
            if candidates:
                sid = rng.choice(sorted(candidates))
                chosen.append(sid)
                selected.add(sid)
    remaining = sorted(set(s1) - selected)
    chosen.extend(rng.sample(remaining, args.size - len(chosen)))
    selected = set(chosen)
    selected_mids = {mid for sid in chosen for mid in links[sid]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sampled_truth.tsv": (truth_fields, [truth[sid] for sid in chosen]),
        "sampled_source1.tsv": (SOURCE_FIELDS, [s1[sid] for sid in chosen]),
    }
    for number in (2, 3):
        outputs[f"sampled_source{number}.tsv"] = (SOURCE_FIELDS, [records[mid] for mid in sorted(selected_mids) if mid.startswith(f"S{number}-")])
    selected_pairs = [row for sid in chosen for row in pairs[sid]]
    pair_fields = list(next(iter(example_pairs.values()))) if example_pairs else ["source1_entity_id", "matched_entity_id", "variation_labels"]
    outputs["variation_pairs.tsv"] = (pair_fields, selected_pairs)
    category_rows = [{"variation": label, "audit_pairs": pair_counts[label], "audit_entities": entity_counts[label],
                      "selected_entities": sum(label in per_entity[sid] for sid in chosen), "definition": definition}
                     for label, definition in DEFINITIONS.items()]
    outputs["variation_categories.tsv"] = (list(category_rows[0]), category_rows)
    if collisions:
        outputs["same_name_different_entities.tsv"] = (list(collisions[0]), collisions)
    covered = set().union(*(per_entity[sid] for sid in chosen))
    summary = {
        "seed": args.seed, "requested_audit_entities": args.audit_size, "audit_entities": len(s1),
        "selected_entities": len(chosen), "country_counts_full_files": country_counts,
        "diagnostics_audit_only": dict(diagnostics), "variation_pair_counts_audit_only": dict(pair_counts),
        "variation_entity_counts_audit_only": dict(entity_counts),
        "same_name_scan_for_audited_names_only": dict(collision_counts),
        "uncovered_observed_categories": sorted(set(entity_counts) - covered),
        "selected_singleton_ids": [sid for sid in chosen if not links[sid]],
        "output_record_counts": {filename: len(rows) for filename, (_, rows) in outputs.items()},
        "limitations": ["Variation labels overlap and are heuristic.",
                        "Country counts scan full files; variation analysis uses a random S1 audit sample.",
                        "Diverse final selection is intentionally biased; do not estimate prevalence or model accuracy from it.",
                        "Script changes do not prove phonetic transliteration or semantic equivalence.",
                        "Low address overlap and parsed state/city conflicts do not prove a business moved.",
                        "No test ground truth is available; test labels are counted only.",
                        "Same-name collision counts apply only to normalized names present in the audit, not all possible name groups."],
    }
    for filename, (fields, rows) in outputs.items():
        write_rows(args.output_dir / filename, fields, rows)
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    # Keep one auditable example of every detected pair category, even with small output sizes.
    write_rows(args.output_dir / "audit_category_examples.tsv", ["category", *pair_fields],
               [{"category": label, **row} for label, row in sorted(example_pairs.items())])
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main():
    root = Path(__file__).resolve().parent / "student_resource/dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=root / "variance_sampled")
    parser.add_argument("--audit-size", type=int, default=50000)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-test-census", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.size <= args.audit_size:
        parser.error("Require 1 <= --size <= --audit-size")
    try:
        run(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
