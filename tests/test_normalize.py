from er.normalize import (
    casefold_unicode,
    fold_latin_accents,
    is_missing,
    legal_form_reduced,
    normalize_address,
    normalize_country,
    normalize_text,
    numeric_tokens,
    tokenize,
)


def test_missingness_detection():
    assert is_missing(None)
    assert is_missing("")
    assert is_missing("   ")
    assert not is_missing("a")


def test_missingness_not_collapsed_to_nan_token():
    # An empty address must never render as the literal string "nan".
    norm = normalize_address("")
    assert norm.missing is True
    assert "nan" not in norm.tokens


def test_unicode_round_trip_devanagari():
    # From the real train_source2.tsv: राम मार्केटिंग प्राइवेट लिमिटेड
    raw = "राम मार्केटिंग प्राइवेट लिमिटेड"
    norm = normalize_text(raw)
    assert norm.raw == raw  # source text preserved verbatim
    assert norm.missing is False
    assert len(norm.tokens) > 0
    # Casefolding a script without case should be idempotent / harmless.
    assert casefold_unicode(raw) == casefold_unicode(casefold_unicode(raw))


def test_accent_folding():
    assert fold_latin_accents("Café Müller") == "Cafe Muller"


def test_legal_form_reduction_strips_trailing_suffix_only():
    assert legal_form_reduced("Acme Corp") == "acme"
    assert legal_form_reduced("Acme Private Ltd") == "acme"
    assert legal_form_reduced("Acme Ltd Consulting") == "acme ltd consulting"


def test_legal_form_reduction_does_not_eat_short_names():
    # "Co" as a legal-form token must not corrupt an unrelated name via
    # substring matching; only trailing whole tokens are stripped.
    assert legal_form_reduced("Coco Bakery") == "coco bakery"


def test_numeric_tokens_extracted():
    assert numeric_tokens("1795 Westchester Drive, Suite 12") == ("1795", "12")
    assert numeric_tokens("No PIN") == ()


def test_tokenize_handles_punctuation_and_ampersand():
    assert tokenize("Smith & Sons, LLC.") == ("smith", "sons", "llc")


def test_country_is_open_string_not_hardcoded():
    # France must normalize the same way as any other value: no special-casing.
    assert normalize_country("France") == "france"
    assert normalize_country(" US ") == "us"
    assert normalize_country(None) == ""


def test_normalized_text_preserves_raw_for_reversal():
    raw = "Wilfordhancock.com"
    norm = normalize_text(raw)
    assert norm.raw == raw
    assert norm.casefold == "wilfordhancock.com"
