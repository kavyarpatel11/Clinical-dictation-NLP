"""
test_segmentation.py

Comprehensive test suite verifying clinical sentence segmentation bug fixes:
- Bug 1: Dictation boundary cues only fire on unpunctuated text, scoped case-sensitivity, no mid-sentence splits.
- Bug 2: Medical route/frequency abbreviations preserve sentence-final dot when followed by capitalized SENTENCE_STARTERS.
- Bug 3: ClinicalSentenceSegmenter.fit() preserves trained Punkt parameters.
"""

import pytest
from src.preprocessing import segment_sentences, is_unpunctuated, UNPUNCTUATED_WORDS_PER_PUNCT_THRESHOLD
from src.models import ClinicalSentenceSegmenter
from src.pipeline import ClinicalDictationPipeline


@pytest.fixture
def segmenter():
    return ClinicalSentenceSegmenter()


@pytest.fixture
def pipeline():
    return ClinicalDictationPipeline()


# ==============================================================================
# The 8 Explicit Assignment Test Cases
# ==============================================================================

def test_01_started_on_single_sentence(segmenter, pipeline):
    """
    Case 1: 'Patient was started on lisinopril last week.'
    Expected: 1 sentence, unchanged.
    """
    text = "Patient was started on lisinopril last week."
    sents = segment_sentences(text)
    assert len(sents) == 1, f"Expected 1 sentence, got {len(sents)}: {sents}"
    assert sents[0] == "Patient was started on lisinopril last week."

    pipeline_res = pipeline.run(text)
    assert pipeline_res["sentence_count"] == 1
    assert pipeline_res["sentences"][0] == "Patient was started on lisinopril last week."


def test_02_reports_no_single_sentence(segmenter, pipeline):
    """
    Case 2: 'Patient reports no chest pain.'
    Expected: 1 sentence.
    """
    text = "Patient reports no chest pain."
    sents = segment_sentences(text)
    assert len(sents) == 1, f"Expected 1 sentence, got {len(sents)}: {sents}"
    assert sents[0] == "Patient reports no chest pain."

    pipeline_res = pipeline.run(text)
    assert pipeline_res["sentence_count"] == 1


def test_03_next_appointment_single_sentence(segmenter, pipeline):
    """
    Case 3: 'She will return for her next appointment in two weeks.'
    Expected: 1 sentence.
    """
    text = "She will return for her next appointment in two weeks."
    sents = segment_sentences(text)
    assert len(sents) == 1, f"Expected 1 sentence, got {len(sents)}: {sents}"
    assert sents[0] == "She will return for her next appointment in two weeks."

    pipeline_res = pipeline.run(text)
    assert pipeline_res["sentence_count"] == 1


def test_04_vital_signs_colon_single_sentence(segmenter, pipeline):
    """
    Case 4: 'Please review her vital signs: stable.'
    Expected: 1 sentence.
    """
    text = "Please review her vital signs: stable."
    sents = segment_sentences(text)
    assert len(sents) == 1, f"Expected 1 sentence, got {len(sents)}: {sents}"
    assert sents[0] == "Please review her vital signs: stable."

    pipeline_res = pipeline.run(text)
    assert pipeline_res["sentence_count"] == 1


def test_05_sentence_final_abbreviation_po(segmenter, pipeline):
    """
    Case 5: 'Medication given p.o. Patient tolerated well.'
    Expected: 2 sentences.
    """
    text = "Medication given p.o. Patient tolerated well."
    sents = segment_sentences(text)
    assert len(sents) == 2, f"Expected 2 sentences, got {len(sents)}: {sents}"
    assert sents[0] == "Medication given p.o."
    assert sents[1] == "Patient tolerated well."

    pipeline_res = pipeline.run(text)
    assert pipeline_res["sentence_count"] == 2
    assert pipeline_res["sentences"][0] == "Medication given p.o."
    assert pipeline_res["sentences"][1] == "Patient tolerated well."


def test_06_multi_abbreviation_three_sentences(segmenter, pipeline):
    """
    Case 6: 'Patient was evaluated today by Dr. Robert Smith. Vital signs: BP 138/84 mmHg, HR 74 bpm. Prescribed Metformin 500 mg p.o. b.i.d. along with Tab. Aspirin 81 mg daily.'
    Expected: 3 sentences.
    """
    text = (
        "Patient was evaluated today by Dr. Robert Smith. "
        "Vital signs: BP 138/84 mmHg, HR 74 bpm. "
        "Prescribed Metformin 500 mg p.o. b.i.d. along with Tab. Aspirin 81 mg daily."
    )
    sents = segment_sentences(text)
    assert len(sents) == 3, f"Expected 3 sentences, got {len(sents)}: {sents}"
    assert sents[0] == "Patient was evaluated today by Dr. Robert Smith."
    assert sents[1] == "Vital signs: BP 138/84 mmHg, HR 74 bpm."
    assert sents[2] == "Prescribed Metformin 500 mg p.o. b.i.d. along with Tab. Aspirin 81 mg daily."

    pipeline_res = pipeline.run(text)
    assert pipeline_res["sentence_count"] == 3


def test_07_decimals_and_dosages_two_sentences(segmenter, pipeline):
    """
    Case 7: 'Creatinine 1.2 mg/dL today. Glucose 126.4 mg/dL.'
    Expected: 2 sentences.
    """
    text = "Creatinine 1.2 mg/dL today. Glucose 126.4 mg/dL."
    sents = segment_sentences(text)
    assert len(sents) == 2, f"Expected 2 sentences, got {len(sents)}: {sents}"
    assert sents[0] == "Creatinine 1.2 mg/dL today."
    assert sents[1] == "Glucose 126.4 mg/dL."

    pipeline_res = pipeline.run(text)
    assert pipeline_res["sentence_count"] == 2


def test_08_unpunctuated_stream_four_sentences(segmenter, pipeline):
    """
    Case 8: 'pt admitted with chest pain vitals stable pt denies nausea discharge planned tomorrow'
    Expected: 4 sentences.
    """
    text = "pt admitted with chest pain vitals stable pt denies nausea discharge planned tomorrow"
    sents = segment_sentences(text)
    assert len(sents) == 4, f"Expected 4 sentences, got {len(sents)}: {sents}"
    assert sents[0] == "pt admitted with chest pain."
    assert sents[1] == "vitals stable."
    assert sents[2] == "pt denies nausea."
    assert sents[3] == "discharge planned tomorrow."

    pipeline_res = pipeline.run(text)
    assert pipeline_res["sentence_count"] == 4


# ==============================================================================
# Additional Edge Case & Regression Tests
# ==============================================================================

def test_mid_sentence_route_frequency_not_split(segmenter):
    """
    Verifies that 'p.o. b.i.d. along with Tab. Aspirin' is ONE sentence.
    """
    text = "Metformin 500 mg p.o. b.i.d. along with Tab. Aspirin 81 mg daily."
    sents = segmenter.segment(text)
    assert len(sents) == 1, f"Expected 1 sentence, got {len(sents)}: {sents}"
    assert sents[0] == "Metformin 500 mg p.o. b.i.d. along with Tab. Aspirin 81 mg daily."


def test_title_abbreviations_never_end_sentence(segmenter):
    """
    Dr., Tab., Cap., Prof., Mr., Mrs. should never trigger a sentence boundary.
    """
    text = "Dr. Robert Smith prescribed Tab. Metformin 500 mg and Cap. Amoxicillin 250 mg."
    sents = segmenter.segment(text)
    assert len(sents) == 1
    assert "Dr. Robert Smith" in sents[0]
    assert "Tab. Metformin" in sents[0]


def test_unpunctuated_threshold_detection():
    """
    Verifies that is_unpunctuated correctly classifies transcripts
    based on the UNPUNCTUATED_WORDS_PER_PUNCT_THRESHOLD.
    """
    assert UNPUNCTUATED_WORDS_PER_PUNCT_THRESHOLD == 25

    # Punctuated: 1 period per 6 words
    punctuated = "Patient was started on lisinopril last week."
    assert not is_unpunctuated(punctuated)

    # Completely unpunctuated: 0 punctuation marks
    unpunctuated = "pt admitted with chest pain vitals stable pt denies nausea discharge planned tomorrow"
    assert is_unpunctuated(unpunctuated)


def test_clinical_sentence_segmenter_fit_preserves_parameters():
    """
    Bug 3: ClinicalSentenceSegmenter.fit() trains a PunktTrainer and rebuilds the
    tokenizer with the trained parameters plus the clinical abbreviation lexicon.
    """
    seg = ClinicalSentenceSegmenter()
    mock_training_data = [
        {
            "expected_sentences": [
                "Patient was admitted for chest pain.",
                "Cardiac enzymes were within normal limits.",
                "Echocardiogram demonstrated normal ejection fraction."
            ],
            "annotations": {
                "protected_abbreviations": ["trop.", "echo."],
                "abbreviations": ["cath."]
            }
        }
    ]
    seg.fit(mock_training_data)

    # Tokenizer must be an instance with abbrev_types populated
    abbrevs = seg.punkt_tokenizer._params.abbrev_types
    # Core clinical abbreviations must be present
    assert "dr" in abbrevs
    assert "po" in abbrevs
    assert "bid" in abbrevs
    assert "tab" in abbrevs
    # Custom abbreviations learned from training data must also be present
    assert "trop" in abbrevs
    assert "echo" in abbrevs
    assert "cath" in abbrevs

    # Model can still segment text accurately after fitting
    test_text = "Medication given p.o. Patient tolerated well."
    res = seg.segment(test_text)
    assert len(res) == 2
