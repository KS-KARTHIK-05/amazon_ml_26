# Dataset variation audit

The dataset is not a simple spelling-correction task. Ground-truth matches can have different scripts, aliases, abbreviations, missing fields, altered address numbers, and different locality or state labels. Conversely, identical business names can belong to different reference entities at entirely different addresses.

## Scope and evidence

I read `ML_Problem.pdf` and inspected the supplied data. The PDF describes expected noise; the observations below come from the actual TSVs. Its instructions are challenge requirements, not additional actions requested by the user.

- Full-file country census: all six training/test source files.
- Detailed variation audit: 50,000 S1 entities sampled uniformly from the full training source with seed 42, plus all 172,910 ground-truth-linked S2/S3 pairs.
- Final diverse sample: 50 S1 entities and 50 truth rows, with 83 S2 records and 91 S3 records. These include three singleton controls with no matches; the remaining 47 entities have linked records.
- Every observed category is represented in the final selection. Labels overlap and are heuristic. This is a broad catalogue of observed variations, not proof that every possible variation in the full dataset has been discovered.
- The 50-entity selection deliberately favors variety. Use the 50,000-entity audit counts for descriptive frequencies, and a separate random held-out split for model evaluation.

## Name variations

The data contains casing and punctuation changes, accents, legal suffix changes, word reordering, insertions/deletions, repeated words, character substitutions, abbreviations and initials, full or mixed-script names, domain/handle-style names, trade-name markers, honorifics, numeric additions, and names with almost no shared words.

These examples are present in the final sampled files and `variation_pairs.tsv`:

| Variation | S1 text | Matched text | Ground-truth link |
| --- | --- | --- | --- |
| Script change, transliteration candidate | Swastik Care | স্বস্তিক কেয়ার | S1-631675403 → S2-365831746 |
| Mixed scripts | Swastik Care | Swastik কেয়ার | S1-631675403 → S2-662528192 |
| Repeated word | Murali Fragrances Private Limited | murali fragrances private limited limited | S1-662425767 → S3-875191268 |
| Letter/digit substitution | Metropolitan Nexera LLC | Metr0politan Nexera | S1-51779529 → S3-383774890 |
| Initialism | Premio Pharma LLP | PP | S1-923701924 → S3-814212886 |
| Legal-form punctuation | NK Infrastructure LLP | NK-Infrastructure L.L.P. | S1-830681339 → S3-823090613 |
| Domain-style name | NK Infrastructure LLP | infrastructurenk.com | S1-830681339 → S3-863208128 |
| Trade alias | Eastern Heartland Angel LP | Synflux DBA: Eastern Heartland Angel LP | S1-542381734 → S3-570344577 |
| Numeric addition | Glyphora Islands LP | #13347 Glyphora Islands | S1-455792228 → S3-307064744 |
| Very different wording | Pranya Dass Private Limited | Vionyla | S1-513175924 → S3-752735437 |

The match labels establish entity correspondence; a script detector does not establish whether a particular transformation is transliteration, translation, or a mixture. Likewise, a high character-similarity label is only a possible typo: abbreviations and suffix changes can also trigger it. The rule definitions in `variation_categories.tsv` make these boundaries explicit.

Preserve Unicode when preparing model inputs. In particular, removing all combining marks damages Indian-script vowel signs and viramas. The script preserves them and only removes Latin accents for its comparison features. All exported source fields retain their original text.

## Address variations and the location question

Observed changes include road/state abbreviations, capitalization, punctuation, reordered components, omitted street/locality/landmark details, empty addresses, literal `NULL` placeholders, unit/PO-box changes, script changes, leading zeros, changed numbers, and locality labels with different wording or granularity.

There are three separate situations:

**1. One labelled entity can have materially different address text.**

`S1-455792228` has `8865 Baseline Road, Unit 1301, Mesa, AZ`; its labelled match `S2-763790228` has `8865 BASELINE ROAD, MESA FOUR PEAKS, AZ`. The locality label changes and the unit disappears. This is more than a typo, although it does not establish relocation.

`S1-262489695` (Shakti Business Limited) has:

> H.No. 2-159/15, Prapurna Enclave, Suchitra Circle, Jeedimetla Village, Hyderabad, Rangareddy, Telangana

Its labelled S3 match `S3-107167362` has:

> 3-159/15, Prapurna Enclave, Suchitra Circle, Jeedimetla Village, Rangareddy, null, Andhra Pradesh

Here the house number and state label differ, Hyderabad is omitted, and a null marker appears. Other labelled S2 records for this same S1 use `2-159/15/6`. The files do not establish whether these differences reflect administrative naming, corruption, or another cause. Do not require exact house-number or state-text agreement.

**2. The audit did not find a US cross-state positive match.**

Among 103,322 US positive pairs, the conservative parser extracted a state from both sides for 97,595 pairs. All those state codes agreed after abbreviation normalization: zero parsed US state conflicts. This does not prove that cross-state positive matches never occur in the entire dataset, and it does not cover Indian state labels.

The city/locality detector flagged 7,371 pairs for manual review. It includes city/county wording and other administrative differences, and can still have parsing errors. This is not a count of confirmed geographically different locations. Low address token overlap is similarly not proof of relocation: missing fields and script changes can cause it.

All 7,949 audited pairs with an exactly identical raw business name nevertheless had different raw address strings. Raw differences include capitalization and abbreviation, so this statistic alone says nothing about geographic distance.

**3. The same business name can occur in New York and Texas under different S1 IDs.**

| Name | S1 entity ID | Address |
| --- | --- | --- |
| Puhl Global, Inc | S1-854291295 | 8 Alfred Drive, Colonie, NY |
| Puhl Global, Inc | S1-402994011 | 1411 Seneca Court, Granbury, TX |

These are separate reference entities, not a ground-truth positive pair. Another example is Helios: `S1-48693199` in East Dubuque, Illinois, versus `S1-865131206` in Bridgeport, Connecticut. See `same_name_different_entities.tsv` for 200 examples. That file is an ambiguity-review table; it must not be interpreted as positive ground truth.

The full S1 scan found 241,719 additional rows sharing a country and normalized name with an audited name anchor but having a different normalized address. This is a count of additional rows relative to one anchor per audited name, not a count of all duplicate-name groups or all possible pairs in the dataset.

## How to use country

Keep country as metadata and use it for candidate grouping. There is no need to predict it or discover country clusters: the labels already exist. No country mismatch occurred among the 172,910 audited positive pairs. The full-file census found only US/India in training and US/India/France in test, with no empty country labels.

| File | US | India | France |
| --- | ---: | ---: | ---: |
| train_source1 | 1,323,633 | 883,188 | 0 |
| train_source2 | 3,016,817 | 2,017,799 | 0 |
| train_source3 | 3,170,056 | 2,115,547 | 0 |
| test_source1 | 663,106 | 809,986 | 259,452 |
| test_source2 | 1,871,330 | 2,312,565 | 703,378 |
| test_source3 | 1,945,701 | 2,405,000 | 731,615 |

Grouping test candidates by country reduces the theoretical S1 × (S2 + S3) search space from 17,272,751,604,416 to 6,724,569,566,212 pairs, a 61.07% reduction. Training reduction is 48.01%. These are combinatorial counts, not measured runtime improvements; within-country matching is still much too large for a full cross join.

For a matching model:

- If all candidate pairs are already within one country, `same_country` is constant and adds no discrimination. The raw country can still help choose text-normalization rules or calibrate scores, but this should be tested.
- If candidates cross country boundaries, country equality/missingness can be pair features. Validate that a hard country filter preserves recall before relying on it universally, and retain a fallback for missing or unreliable country labels.
- Create groups dynamically from country strings. France is an actual new label to process, not a reason to discard records. Do not build a pipeline restricted to the two training labels or rely exclusively on a country embedding that has never learned France.
- Country grouping is candidate filtering, not entity clustering. Within each country, use name and address evidence together. Avoid hard exact-name, exact-city, exact-state, or exact-number requirements.
- Maintain raw and normalized text views. Address information is especially useful when names are initials, domain-style strings, or aliases; name information becomes more important when the address is absent.
- Build difficult negative examples from similar or identical names at different addresses. Keep every S1 entity and its linked variants together when constructing train/validation splits. Retain no-match entities because the challenge scores them too.

The problem statement specifies macro F0.5, which favors precision, and includes singleton/no-match cases. A variety sample is useful for debugging preprocessing and candidate recall, but it is not a representative validation benchmark. No external business lookup or geocoding was used for this audit.

## Files and reproduction

Run from the project directory:

```bash
python3 data_var.py --audit-size 50000 --size 50 --seed 42
```

The script uses Python's standard library and the existing TSV helpers in `sample_dataset.py`. Paths default relative to the script, so it also works when called by absolute path from another working directory. It samples the full S1 training file, follows ground truth, classifies pairs, selects for category coverage, and exports every match for every selected S1, even if only one match triggered its selection.

Outputs are under `student_resource/dataset/variance_sampled/`, leaving the earlier random sample in `dataset/sampled/` separate:

- `sampled_truth.tsv`, `sampled_source1.tsv`, `sampled_source2.tsv`, `sampled_source3.tsv`: original challenge schemas and raw values.
- `variation_pairs.tsv`: all 174 selected positive links, with side-by-side fields, heuristic labels, overlap scores, and parsed US locations.
- `variation_categories.tsv`: all detector definitions and counts in the audit and selected sample, including zero-count categories.
- `audit_category_examples.tsv`: one audit example per observed pair category; some are outside the final 50-entity selection.
- `same_name_different_entities.tsv`: name-collision examples across separate S1 IDs.
- `audit_summary.json`: scope, full country census, counts, coverage, controls, and limitations.

`--output-dir`, `--dataset-dir`, `--seed`, `--size`, and `--audit-size` are configurable. `--skip-test-census` skips the test-file counts on exploratory reruns. Larger audits need more memory because the script retains audited entities, their matches, and annotations; source files themselves are streamed.

Validation checked exact truth/source correspondence, unique exported IDs, preservation of all referenced S2/S3 records, complete annotation coverage, and coverage of every observed category. Small fixtures also checked reproducibility, Unicode preservation, empty match lists, missing-reference failures, abbreviations, initialisms, and numeric/word distinctions.

## Detailed audit counts

Counts below refer to the random 50,000-S1 audit, not the final diverse sample. Pair percentages use 172,910 positive links as the denominator and overlap. Singleton controls are counted per S1 entity instead.

| Detector | Positive pairs | % of positive pairs | S1 entities | Selected S1 entities |
| --- | ---: | ---: | ---: | ---: |
| `name_case_only` | 10,402 | 6.02% | 9,316 | 7 |
| `name_punctuation_spacing` | 19,265 | 11.14% | 15,896 | 13 |
| `name_accent_change` | 6,921 | 4.00% | 6,430 | 7 |
| `name_script_change_possible_transliteration` | 12,713 | 7.35% | 6,224 | 8 |
| `name_mixed_scripts` | 904 | 0.52% | 843 | 2 |
| `name_legal_suffix_change` | 65,511 | 37.89% | 34,107 | 35 |
| `name_abbreviation_equivalent` | 4,953 | 2.86% | 4,377 | 3 |
| `name_word_reordering` | 9,631 | 5.57% | 8,612 | 10 |
| `name_repeated_word_added` | 4,131 | 2.39% | 3,961 | 5 |
| `name_token_addition_or_removal` | 46,909 | 27.13% | 29,861 | 25 |
| `name_possible_typo` | 41,440 | 23.97% | 27,724 | 29 |
| `name_digit_letter_substitution` | 2,617 | 1.51% | 2,522 | 3 |
| `name_url_or_handle` | 9,582 | 5.54% | 8,717 | 12 |
| `name_numeric_identifier_added` | 1,718 | 0.99% | 1,693 | 2 |
| `name_joined_or_split_words` | 1,986 | 1.15% | 1,836 | 2 |
| `name_initialism_candidate` | 265 | 0.15% | 265 | 2 |
| `name_honorific_added` | 3,685 | 2.13% | 3,251 | 5 |
| `name_trade_alias_marker` | 1,614 | 0.93% | 1,614 | 4 |
| `name_low_lexical_overlap` | 26,931 | 15.58% | 17,895 | 25 |
| `name_missing` | 0 | 0.00% | 0 | 0 |
| `address_missing` | 7,743 | 4.48% | 7,196 | 10 |
| `address_case_punctuation_spacing` | 10,424 | 6.03% | 8,109 | 5 |
| `address_script_change` | 15,542 | 8.99% | 10,883 | 11 |
| `address_abbreviation_or_state_expansion` | 27,286 | 15.78% | 15,481 | 15 |
| `address_component_reordering` | 14,005 | 8.10% | 9,786 | 8 |
| `address_component_addition_or_removal` | 32,823 | 18.98% | 20,699 | 19 |
| `address_numeric_tokens_changed` | 56,132 | 32.46% | 27,532 | 28 |
| `address_unit_po_box_change` | 17,374 | 10.05% | 8,278 | 8 |
| `address_embedded_null_marker` | 4,360 | 2.52% | 2,269 | 5 |
| `address_landmark_marker_change` | 1,669 | 0.97% | 1,086 | 2 |
| `address_leading_zero_number_format` | 5,408 | 3.13% | 3,034 | 7 |
| `address_possible_typo` | 62,708 | 36.27% | 31,529 | 27 |
| `address_low_lexical_overlap` | 4,481 | 2.59% | 3,190 | 5 |
| `us_state_conflict_candidate` | 0 | 0.00% | 0 | 0 |
| `us_city_conflict_candidate` | 7,371 | 4.26% | 4,219 | 6 |
| `country_mismatch` | 0 | 0.00% | 0 | 0 |
| `country_missing` | 0 | 0.00% | 0 | 0 |
| `unchanged_record` | 0 | 0.00% | 0 | 0 |
| `other_text_variation` | 819 | 0.47% | 774 | 2 |
| `singleton_control` | 0 | not applicable | 2,800 | 3 |

Zero observed examples do not prove a category is absent from the full dataset. Script changes, possible typos, and locality conflicts require interpretation; these labels are not independently verified annotations.
