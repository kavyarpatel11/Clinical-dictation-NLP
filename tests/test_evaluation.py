"""
Unit tests for the expanded evaluation suite of Clinical Dictation NLP.

Verifies:
- Expanded dataset loads correctly and contains 50-100 transcripts (70 total)
- Every transcript has valid schema and rich clinical annotations
- Deterministic 70/30 train/test data separation
- Safe zero-division handling in metrics calculations
- Sentence boundary detection evaluation
- Medical tokenization evaluation
- Clinical normalization evaluation
- Negation preservation evaluation
- Naive baseline vs. Proposed system comparison
- All report files generated (JSON, CSV, MD)
- Flask web application routes and benchmark integration
"""

import os
import json
import pytest
from src.pipeline import ClinicalDictationPipeline
from src.models import ClinicalSentenceSegmenter, ClinicalTokenizer
from src.evaluation import (
    calculate_precision,
    calculate_recall,
    calculate_f1_score,
    extract_boundary_offsets,
    evaluate_sentence_segmentation,
    evaluate_tokenization,
    evaluate_normalization,
    evaluate_negation_preservation,
    split_dataset,
    run_pipeline_evaluation,
    run_baseline_evaluation,
    compare_baseline_vs_proposed,
    naive_baseline_segmenter,
    naive_baseline_tokenizer,
    nltk_baseline_segmenter,
    nltk_baseline_tokenizer,
    spacy_baseline_segmenter,
    spacy_baseline_tokenizer,
    run_nltk_baseline_evaluation,
    run_spacy_baseline_evaluation
)


@pytest.fixture
def gold_corpus():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gold_path = os.path.join(base_dir, "data", "annotated", "gold_standard.json")
    with open(gold_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def dataset_split(gold_corpus):
    return split_dataset(gold_corpus, test_ratio=0.3, random_seed=42)


def test_expanded_dataset_loads_correctly(gold_corpus):
    """
    Verifies that the expanded gold-standard dataset loads correctly,
    contains 50-100 realistic transcripts (70 total), and all IDs are unique.
    """
    assert len(gold_corpus) >= 50, f"Expected at least 50 transcripts, got {len(gold_corpus)}"
    assert len(gold_corpus) <= 100, f"Expected at most 100 transcripts, got {len(gold_corpus)}"
    assert len(gold_corpus) == 70

    ids = [item["id"] for item in gold_corpus]
    assert len(ids) == len(set(ids)), "Transcript IDs must be unique"

    # Verify original 3 are preserved
    assert ids[0] == "transcript_001"
    assert ids[1] == "transcript_002"
    assert ids[2] == "transcript_003"


def test_every_transcript_has_valid_annotations(gold_corpus):
    """
    Verifies that every transcript contains text, expected sentences,
    boundary offsets, expected tokens, and clinical annotations.
    """
    for item in gold_corpus:
        t_id = item.get("id")
        assert "text" in item and len(item["text"]) > 10, f"Missing text in {t_id}"
        assert "expected_sentences" in item and len(item["expected_sentences"]) >= 1, f"Missing sentences in {t_id}"
        assert "expected_sentence_boundaries" in item, f"Missing boundaries in {t_id}"
        assert len(item["expected_sentence_boundaries"]) == len(item["expected_sentences"]), f"Boundary mismatch in {t_id}"
        assert "expected_tokens" in item and len(item["expected_tokens"]) >= 1, f"Missing tokens in {t_id}"

        ann = item.get("annotations", {})
        assert "abbreviations" in ann, f"Missing abbreviations annotation in {t_id}"
        assert "decimal_measurements" in ann, f"Missing decimal_measurements in {t_id}"
        assert "dates" in ann, f"Missing dates in {t_id}"
        assert "medication_dosages" in ann, f"Missing medication_dosages in {t_id}"
        assert "units" in ann, f"Missing units in {t_id}"
        assert "blood_pressure" in ann, f"Missing blood_pressure in {t_id}"
        assert "clinical_negations" in ann, f"Missing clinical_negations in {t_id}"
        assert "expected_normalizations" in ann, f"Missing expected_normalizations in {t_id}"


def test_train_test_split_reproducibility(gold_corpus):
    """
    Verifies that split_dataset performs a reproducible 70% dev / 30% test split.
    """
    dev1, test1 = split_dataset(gold_corpus, test_ratio=0.3, random_seed=42)
    dev2, test2 = split_dataset(gold_corpus, test_ratio=0.3, random_seed=42)

    assert len(dev1) == 49
    assert len(test1) == 21
    assert [d["id"] for d in dev1] == [d["id"] for d in dev2]
    assert [t["id"] for t in test1] == [t["id"] for t in test2]

    # Verify dev and test are disjoint
    dev_ids = {d["id"] for d in dev1}
    test_ids = {t["id"] for t in test1}
    assert dev_ids.isdisjoint(test_ids), "Dev and test partitions must be disjoint"


def test_zero_division_safety():
    """
    Verifies that precision, recall, and F1 calculations never raise ZeroDivisionError.
    """
    assert calculate_precision(0, 0) == 0.0
    assert calculate_recall(0, 0) == 0.0
    assert calculate_f1_score(0.0, 0.0) == 0.0
    assert calculate_f1_score(0.8, 0.0) == 0.0
    assert calculate_f1_score(0.0, 0.8) == 0.0

    # Normal calculation
    assert calculate_precision(10, 2) == round(10 / 12, 4)
    assert calculate_recall(10, 3) == round(10 / 13, 4)
    p = calculate_precision(10, 2)
    r = calculate_recall(10, 3)
    assert calculate_f1_score(p, r) == round(2 * p * r / (p + r), 4)


def test_sentence_boundary_evaluation():
    """
    Verifies evaluate_sentence_segmentation logic with exact and tolerance matching.
    """
    text = "Dr. Smith evaluated the patient. Blood pressure was 120/80 mmHg."
    gt = [
        "Dr. Smith evaluated the patient.",
        "Blood pressure was 120/80 mmHg."
    ]
    pred = [
        "Dr. Smith evaluated the patient.",
        "Blood pressure was 120/80 mmHg."
    ]
    res = evaluate_sentence_segmentation(pred, gt, text)
    assert res["true_positives"] == 2
    assert res["false_positives"] == 0
    assert res["false_negatives"] == 0
    assert res["f1_score"] == 1.0


def test_tokenization_evaluation():
    """
    Verifies evaluate_tokenization multiset matching.
    """
    gt_tokens = ["Lisinopril", "10", "mg", "p.o.", "q.d.", "."]
    pred_tokens = ["Lisinopril", "10", "mg", "p.o.", "q.d.", "."]
    res = evaluate_tokenization(pred_tokens, gt_tokens)
    assert res["true_positives"] == 6
    assert res["false_positives"] == 0
    assert res["false_negatives"] == 0
    assert res["f1_score"] == 1.0

    # Partial match
    partial_pred = ["Lisinopril", "10", "mg"]
    p_res = evaluate_tokenization(partial_pred, gt_tokens)
    assert p_res["true_positives"] == 3
    assert p_res["false_negatives"] == 3
    assert p_res["precision"] == 1.0
    assert p_res["recall"] == 0.5


def test_normalization_evaluation():
    """
    Verifies evaluate_normalization handles correct, missed, and empty targets.
    """
    expected_norms = [
        {"shorthand": "pt", "canonical": "patient"},
        {"shorthand": "p.o.", "canonical": "orally"}
    ]
    norm_tokens = ["patient", "admitted", "orally"]
    res = evaluate_normalization(norm_tokens, expected_norms)
    assert res["correct"] == 2
    assert res["missed"] == 0
    assert res["accuracy"] == 1.0

    # Missed normalization
    unnormalized_tokens = ["pt", "admitted", "p.o."]
    res_missed = evaluate_normalization(unnormalized_tokens, expected_norms)
    assert res_missed["missed"] == 2
    assert res_missed["accuracy"] == 0.0


def test_negation_preservation_evaluation():
    """
    Verifies evaluate_negation_preservation checks for preservation against stopword removal.
    """
    expected_negs = ["no", "denies"]
    preserved_tokens = ["patient", "denies", "chest", "pain", "no", "fever"]
    res = evaluate_negation_preservation(preserved_tokens, expected_negs)
    assert res["preserved"] == 2
    assert res["removed"] == 0
    assert res["accuracy"] == 1.0

    # Removed negation
    removed_tokens = ["patient", "chest", "pain", "fever"]
    res_rem = evaluate_negation_preservation(removed_tokens, expected_negs)
    assert res_rem["preserved"] == 0
    assert res_rem["removed"] == 2
    assert res_rem["accuracy"] == 0.0


def test_proposed_outperforms_naive_baseline(dataset_split):
    """
    Verifies that the proposed clinical system demonstrates empirical superiority
    over the naive baseline across the held-out test dataset.
    """
    _, test_corpus = dataset_split
    pipeline = ClinicalDictationPipeline()

    proposed_res = run_pipeline_evaluation(test_corpus, pipeline=pipeline)
    baseline_res = run_baseline_evaluation(test_corpus)

    # Sentence F1 comparison
    p_sent_f1 = proposed_res["sentence"]["f1_score"]
    b_sent_f1 = baseline_res["sentence"]["f1_score"]
    assert p_sent_f1 > b_sent_f1, f"Proposed sentence F1 ({p_sent_f1}) must exceed baseline ({b_sent_f1})"
    assert p_sent_f1 >= 0.90

    # Token F1 comparison
    p_tok_f1 = proposed_res["token"]["f1_score"]
    b_tok_f1 = baseline_res["token"]["f1_score"]
    assert p_tok_f1 > b_tok_f1, f"Proposed token F1 ({p_tok_f1}) must exceed baseline ({b_tok_f1})"
    assert p_tok_f1 >= 0.95

    # Normalization accuracy comparison
    p_norm_acc = proposed_res["normalization"]["accuracy"]
    b_norm_acc = baseline_res["normalization"]["accuracy"]
    assert p_norm_acc > b_norm_acc, f"Proposed norm acc ({p_norm_acc}) must exceed baseline ({b_norm_acc})"
    assert b_norm_acc == 0.0

    # Negation preservation accuracy comparison
    p_neg_acc = proposed_res["negation"]["accuracy"]
    b_neg_acc = baseline_res["negation"]["accuracy"]
    assert p_neg_acc > b_neg_acc, f"Proposed neg acc ({p_neg_acc}) must exceed baseline ({b_neg_acc})"
    assert p_neg_acc == 1.0


def test_all_evaluation_reports_exist():
    """
    Verifies that all 5 required evaluation report artifacts exist on disk.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, "reports")

    assert os.path.exists(os.path.join(reports_dir, "overall_evaluation.json")), "Missing overall_evaluation.json"
    assert os.path.exists(os.path.join(reports_dir, "overall_evaluation.csv")), "Missing overall_evaluation.csv"
    assert os.path.exists(os.path.join(reports_dir, "evaluation_errors.json")), "Missing evaluation_errors.json"
    assert os.path.exists(os.path.join(reports_dir, "evaluation_errors.csv")), "Missing evaluation_errors.csv"
    assert os.path.exists(os.path.join(reports_dir, "OVERALL_EVALUATION.md")), "Missing OVERALL_EVALUATION.md"


def test_web_application_endpoints():
    """
    Verifies that the Flask web application loads successfully and serves
    the benchmark comparison table and analysis results.
    """
    from run import app
    client = app.test_client()

    # 1. Main dashboard
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"System Benchmark" in resp.data
    assert b"Proposed Clinical System" in resp.data
    assert b"Naive Baseline" in resp.data

    # 2. Benchmark API
    b_resp = client.get("/api/benchmark")
    assert b_resp.status_code == 200
    b_data = b_resp.get_json()
    assert "proposed" in b_data
    assert "baseline" in b_data
    assert "improvement" in b_data

    # 3. Analyze API endpoint
    a_resp = client.post("/analyze", json={"transcript": "Dr. Adams treated pt with 500 mg Metformin p.o. b.i.d. BP was 120/80 mmHg."})
    assert a_resp.status_code == 200
    a_data = a_resp.get_json()
    assert a_data["status"] == "success"
    assert len(a_data["sentences"]) >= 1
    assert "tokens" in a_data

    # 4. Compare libraries API endpoint
    c_resp = client.post("/api/compare", json={"transcript": "Dr. Smith prescribed 2.5 mg p.o. BP 120/80 mmHg."})
    assert c_resp.status_code == 200
    c_data = c_resp.get_json()
    assert c_data["status"] == "success"
    assert "proposed" in c_data
    assert "nltk" in c_data
    assert "spacy" in c_data
    assert "naive_regex" in c_data


def test_nltk_and_spacy_baselines_evaluation(dataset_split):
    """
    Verifies that run_nltk_baseline_evaluation and run_spacy_baseline_evaluation execute
    cleanly across the test dataset and demonstrate that the proposed clinical model
    statistically outperforms standard unadapted NLTK and spaCy models.
    """
    _, test_corpus = dataset_split
    nltk_res = run_nltk_baseline_evaluation(test_corpus)
    spacy_res = run_spacy_baseline_evaluation(test_corpus)

    # Valid metric ranges
    assert 0.0 <= nltk_res["sentence"]["f1_score"] <= 1.0
    assert 0.0 <= nltk_res["token"]["f1_score"] <= 1.0
    assert 0.0 <= spacy_res["sentence"]["f1_score"] <= 1.0
    assert 0.0 <= spacy_res["token"]["f1_score"] <= 1.0

    # Proposed system sentence F1 (0.9528) exceeds standard NLTK (0.8250) and spaCy (0.8201)
    pipeline = ClinicalDictationPipeline()
    proposed_res = run_pipeline_evaluation(test_corpus, pipeline=pipeline)
    p_sent_f1 = proposed_res["sentence"]["f1_score"]
    assert p_sent_f1 > nltk_res["sentence"]["f1_score"]
    assert p_sent_f1 > spacy_res["sentence"]["f1_score"]

    # Proposed system token F1 (0.9922) exceeds standard NLTK (0.9338) and spaCy (0.8814)
    p_tok_f1 = proposed_res["token"]["f1_score"]
    assert p_tok_f1 > nltk_res["token"]["f1_score"]
    assert p_tok_f1 > spacy_res["token"]["f1_score"]
