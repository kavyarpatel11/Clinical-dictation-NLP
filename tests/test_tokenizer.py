"""
Unit tests for Clinical Dictation Sentence Boundary Detector & Tokenizer.

Verifies handling of critical clinical transcription patterns:
- Dr. abbreviation
- mg/dL unit
- p.o. administration route
- q.i.d. frequency
- dates
- blood pressure
- decimal values
- negative clinical statements
"""

import pytest
from src.pipeline import ClinicalDictationPipeline
from src.models import ClinicalSentenceSegmenter, ClinicalTokenizer
from src.preprocessing import detect_clinical_patterns


@pytest.fixture
def pipeline():
    return ClinicalDictationPipeline()


@pytest.fixture
def segmenter():
    return ClinicalSentenceSegmenter()


@pytest.fixture
def tokenizer():
    return ClinicalTokenizer()


def test_dr_abbreviation(segmenter, pipeline):
    """
    Test that 'Dr.' title abbreviation does not trigger a false sentence boundary.
    Example: 'Patient was seen by Dr. Smith for consultation.'
    """
    text = "Patient was seen by Dr. Smith for consultation."
    sentences = segmenter.segment(text)
    assert len(sentences) == 1, f"Expected 1 sentence, got {len(sentences)}: {sentences}"
    assert "Dr. Smith" in sentences[0]


def test_mg_per_dl_unit(tokenizer):
    """
    Test that compound measurement unit 'mg/dL' is preserved as a single token
    without splitting the slash or isolating the unit.
    Example: 'Blood glucose level was 110 mg/dL before breakfast.'
    """
    sentence = "Blood glucose level was 110 mg/dL before breakfast."
    tokens = tokenizer.tokenize(sentence)
    assert any("mg/dL" in t for t in tokens), f"'mg/dL' not found in tokens: {tokens}"


def test_po_abbreviation(segmenter):
    """
    Test that Latin medical route 'p.o.' (per os / orally) is protected from sentence splitting.
    Example: 'Administer Metformin 500 mg p.o. with meals.'
    """
    text = "Administer Metformin 500 mg p.o. with meals."
    sentences = segmenter.segment(text)
    assert len(sentences) == 1, f"Expected 1 sentence, got {len(sentences)}: {sentences}"
    assert "p.o." in sentences[0]


def test_qid_abbreviation(segmenter):
    """
    Test that Latin medical frequency 'q.i.d.' (four times a day) is protected from sentence splitting.
    Example: 'Take Amoxicillin 250 mg q.i.d. for 7 days.'
    """
    text = "Take Amoxicillin 250 mg q.i.d. for 7 days."
    sentences = segmenter.segment(text)
    assert len(sentences) == 1, f"Expected 1 sentence, got {len(sentences)}: {sentences}"
    assert "q.i.d." in sentences[0]


def test_dates_pattern(tokenizer):
    """
    Test that date formats (e.g., '12/04/2023', '04-12-2023') are preserved as entities.
    Example: 'Last clinic visit was on 12/04/2023 without incident.'
    """
    sentence = "Last clinic visit was on 12/04/2023 without incident."
    tokens = tokenizer.tokenize(sentence)
    assert "12/04/2023" in tokens, f"'12/04/2023' not found in tokens: {tokens}"

    patterns = detect_clinical_patterns(sentence)
    date_patterns = [p["value"] for p in patterns if p["category"] == "date"]
    assert "12/04/2023" in date_patterns


def test_blood_pressure_pattern(tokenizer):
    """
    Test that blood pressure readings (e.g., '120/80 mmHg') remain intact.
    Example: 'Vitals showed BP 120/80 mmHg with pulse 72 bpm.'
    """
    sentence = "Vitals showed BP 120/80 mmHg with pulse 72 bpm."
    tokens = tokenizer.tokenize(sentence)
    assert "120/80" in tokens, f"'120/80' ratio not preserved in tokens: {tokens}"
    assert "mmHg" in tokens or "120/80 mmHg" in tokens

    patterns = detect_clinical_patterns(sentence)
    bp_patterns = [p["value"] for p in patterns if p["category"] == "blood_pressure"]
    assert any("120/80" in bp for bp in bp_patterns)


def test_decimal_values(segmenter):
    """
    Test that decimal numeric values (e.g., '2.5 mg', '0.75 ml') do not cause sentence splits at periods.
    Example: 'Patient was prescribed 2.5 mg of Amlodipine daily.'
    """
    text = "Patient was prescribed 2.5 mg of Amlodipine daily."
    sentences = segmenter.segment(text)
    assert len(sentences) == 1, f"Expected 1 sentence, got {len(sentences)}: {sentences}"
    assert "2.5" in sentences[0]


def test_negative_clinical_statements(tokenizer):
    """
    Test that clinical negations ('no', 'not', 'denies', 'without') are strictly preserved
    during stop-word removal and tokenization.
    Example: 'Patient denies chest pain and reports no shortness of breath.'
    """
    sentence = "Patient denies chest pain and reports no shortness of breath."
    raw_tokens = tokenizer.tokenize(sentence)
    filtered_tokens = tokenizer.filter_stopwords(raw_tokens)

    # Convert to lowercase for checking
    filtered_lower = [t.lower() for t in filtered_tokens]

    assert "denies" in filtered_lower, f"'denies' was mistakenly removed: {filtered_tokens}"
    assert "no" in filtered_lower, f"'no' was mistakenly removed: {filtered_tokens}"


def test_nltk_punkt_sentence_segmentation(segmenter):
    """
    Test that the domain-adapted NLTK PunktSentenceTokenizer correctly handles
    medical abbreviations (Dr., p.o., q.i.d.) and numeric decimals.
    """
    text = "Dr. Smith treated the patient. BP was 120/80 mmHg. Prescribed 2.5 mg p.o. q.i.d. for 7 days."
    sents = segmenter.segment(text, engine="hybrid")
    assert len(sents) == 3, f"Expected 3 sentences, got {len(sents)}: {sents}"
    assert "Dr. Smith" in sents[0]
    assert "120/80 mmHg" in sents[1]
    assert "2.5 mg p.o. q.i.d." in sents[2]

    # Test direct NLTK method
    nltk_sents = segmenter.segment_with_nltk(text)
    assert len(nltk_sents) >= 2


def test_spacy_clinical_tokenization(tokenizer):
    """
    Test that the domain-adapted spaCy tokenizer preserves compound clinical units (mg/dL),
    ratios (120/80), Latin abbreviations (p.o., q.i.d.), and hyphenated terms (follow-up).
    """
    sentence = "Vitals: BP 120/80 mmHg, glucose 110 mg/dL, take 500 mg p.o. q.i.d., follow-up in 2 wks."
    spacy_toks = tokenizer.tokenize_with_spacy(sentence)

    assert "mg/dL" in spacy_toks, f"'mg/dL' should be a single token in spaCy: {spacy_toks}"
    assert "120/80" in spacy_toks, f"'120/80' should be a single token in spaCy: {spacy_toks}"
    assert "p.o." in spacy_toks, f"'p.o.' should be preserved in spaCy: {spacy_toks}"
    assert "q.i.d." in spacy_toks, f"'q.i.d.' should be preserved in spaCy: {spacy_toks}"
    assert "follow-up" in spacy_toks, f"'follow-up' should be preserved in spaCy: {spacy_toks}"


def test_nltk_clinical_tokenization(tokenizer):
    """
    Test that the NLTK RegexpTokenizer correctly extracts clinical entities.
    """
    sentence = "Lab: HbA1c 6.8%, glucose 126.4 mg/dL on 14/11/2023."
    nltk_toks = tokenizer.tokenize_with_nltk(sentence)

    assert "126.4" in nltk_toks
    assert "mg/dL" in nltk_toks
    assert "14/11/2023" in nltk_toks
    assert "6.8%" in nltk_toks


def test_spacy_linguistic_doc(tokenizer):
    """
    Test that spaCy Doc object provides linguistic inspection (shapes, lemmas, tokens).
    """
    sentence = "Patient presents with persistent cough."
    doc = tokenizer.get_spacy_doc(sentence)
    tokens = [t.text for t in doc]
    assert "Patient" in tokens
    assert "presents" in tokens
    assert all(hasattr(t, "shape_") for t in doc)


def test_combined_stopwords_negation_guard():
    """
    Test that the combined NLTK and spaCy clinical stopword set excludes all critical diagnostic negations.
    """
    from src.preprocessing import CLINICAL_STOPWORDS, PRESERVED_NEGATIONS
    for neg in PRESERVED_NEGATIONS:
        assert neg not in CLINICAL_STOPWORDS, f"Negation '{neg}' found in stopword set!"
