"""
Pipeline module for Clinical Dictation NLP.

This module orchestrates the complete processing pipeline for clinical dictation transcripts:

Complete Pipeline Architecture:
-------------------------------
Raw Transcript
  ↓
Text Normalization
  ↓
Clinical Pattern Protection
  ↓
Sentence Segmentation
  ↓
Medical Tokenization
  ↓
Stop-word Processing
  ↓
Token Normalization
  ↓
Evaluation/Output
"""

import os
import sys
import re
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path when run directly as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.models import ClinicalSentenceSegmenter, ClinicalTokenizer
    from src.preprocessing import detect_clinical_patterns
except ModuleNotFoundError:
    from models import ClinicalSentenceSegmenter, ClinicalTokenizer
    from preprocessing import detect_clinical_patterns


class ClinicalDictationPipeline:
    """
    Orchestrator for the clinical dictation sentence boundary detection
    and tokenization pipeline.
    """

    def __init__(self):
        """
        Initializes pipeline components and models.
        """
        self.segmenter = ClinicalSentenceSegmenter()
        self.tokenizer = ClinicalTokenizer()

    def normalize_raw_text(self, text: str) -> str:
        """
        Stage 2: Text Normalization
        Cleans abnormal whitespace, linebreaks, and non-printable control characters.
        """
        cleaned = re.sub(r"[\r\n\t]+", " ", text)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    def run(self, raw_transcript: str) -> Dict[str, Any]:
        """
        Executes the end-to-end processing pipeline on a raw clinical transcript.

        Pipeline Stages:
        ================
        1. Raw Transcript:
           - Ingestion and validation of raw transcript string.

        2. Text Normalization:
           - Stripping anomalous whitespace and standardizing spacing.

        3. Clinical Pattern Protection & Detection:
           - Identification of dosages, blood pressure, dates, decimals, abbreviations, and negations.

        4. Sentence Segmentation:
           - Segmenting transcript without false splits on protected entities.

        5. Medical Tokenization:
           - Extracting medical terms, units (mg/dL), and clinical shorthand per sentence.

        6. Stop-word Processing:
           - Removing non-diagnostic stopwords while preserving diagnostic negations.

        7. Token Normalization:
           - Normalizing case and mapping medical abbreviations to canonical forms.

        8. Evaluation/Output:
           - Assembling structured analysis result dictionary.

        Args:
            raw_transcript (str): Raw input clinical transcript text.

        Returns:
            Dict[str, Any]: Structured dictionary with processing results.
        """
        if not raw_transcript or not raw_transcript.strip():
            return {
                "raw_text": "",
                "normalized_text": "",
                "sentences": [],
                "sentence_count": 0,
                "tokens": [],
                "sentence_tokens": [],
                "normalized_tokens": [],
                "detected_patterns": {},
                "statistics": {
                    "total_sentences": 0,
                    "total_tokens": 0,
                    "normalized_tokens_count": 0,
                    "protected_abbreviations_count": 0,
                    "preserved_negations_count": 0
                }
            }

        # Stage 1 & 2: Raw Text Ingestion & Text Normalization
        normalized_text = self.normalize_raw_text(raw_transcript)

        # Stage 3: Clinical Pattern Detection
        detected_patterns = detect_clinical_patterns(normalized_text)

        # Categorize patterns for reporting
        pattern_categories = {
            "blood_pressure": [],
            "date": [],
            "dosage_or_measurement": [],
            "decimal_value": [],
            "clinical_abbreviation": [],
            "clinical_negation": []
        }
        for pat in detected_patterns:
            cat = pat.get("category")
            if cat in pattern_categories:
                pattern_categories[cat].append(pat.get("value"))

        # Stage 4: Sentence Segmentation
        sentences = self.segmenter.segment(normalized_text)

        # Stage 5, 6 & 7: Medical Tokenization, Stop-word Processing, and Token Normalization
        all_tokens = []
        sentence_tokens = []
        all_normalized_tokens = []

        for sentence in sentences:
            # Stage 5: Tokenization
            raw_sent_tokens = self.tokenizer.tokenize(sentence)
            sentence_tokens.append(raw_sent_tokens)
            all_tokens.extend(raw_sent_tokens)

            # Stage 6: Stop-word filtering preserving negations
            filtered_tokens = self.tokenizer.filter_stopwords(raw_sent_tokens)

            # Stage 7: Rule-based normalization
            norm_tokens = self.tokenizer.normalize(filtered_tokens)
            all_normalized_tokens.extend(norm_tokens)

        # Stage 8: Multi-Library Diagnostic Execution
        standard_nltk_sents = []
        try:
            import nltk
            standard_nltk_sents = nltk.tokenize.sent_tokenize(normalized_text)
        except Exception:
            pass

        spacy_sents = []
        spacy_tokens_list = []
        try:
            spacy_doc = self.tokenizer.get_spacy_doc(normalized_text)
            spacy_sents = [s.text.strip() for s in spacy_doc.sents if s.text.strip()]
            spacy_tokens_list = [t.text for t in spacy_doc if not t.is_space]
        except Exception:
            pass

        # Stage 9: Assemble Output Statistics
        negations_found = [
            t for t in all_tokens
            if t.lower() in {"no", "not", "denies", "denied", "without", "never", "none"}
        ]

        statistics = {
            "total_sentences": len(sentences),
            "total_tokens": len(all_tokens),
            "normalized_tokens_count": len(all_normalized_tokens),
            "protected_abbreviations_count": len(pattern_categories["clinical_abbreviation"]),
            "preserved_negations_count": len(negations_found),
            "detected_dosages_count": len(pattern_categories["dosage_or_measurement"]),
            "detected_bp_count": len(pattern_categories["blood_pressure"])
        }

        return {
            "raw_text": raw_transcript,
            "normalized_text": normalized_text,
            "sentences": sentences,
            "sentence_count": len(sentences),
            "tokens": all_tokens,
            "sentence_tokens": sentence_tokens,
            "normalized_tokens": all_normalized_tokens,
            "detected_patterns": pattern_categories,
            "statistics": statistics,
            "library_comparison": {
                "proposed_sentences_count": len(sentences),
                "nltk_sentences_count": len(standard_nltk_sents),
                "spacy_sentences_count": len(spacy_sents),
                "standard_nltk_sentences": standard_nltk_sents,
                "standard_spacy_sentences": spacy_sents,
                "spacy_tokens": spacy_tokens_list
            }
        }


if __name__ == "__main__":
    pipeline = ClinicalDictationPipeline()
    sample_text = (
        "Patient was evaluated today by Dr. Robert Smith. Vital signs: BP 138/84 mmHg, HR 74 bpm. "
        "Prescribed Metformin 500 mg p.o. b.i.d. along with Tab. Aspirin 81 mg daily."
    )
    print("=" * 60)
    print("CLINICAL DICTATION NLP PIPELINE DEMONSTRATION")
    print("=" * 60)
    output = pipeline.run(sample_text)
    print(f"\n1. Segmented Sentences ({output['sentence_count']}):")
    for i, s in enumerate(output["sentences"], 1):
        print(f"   [{i}] {s}")

    print(f"\n2. Extracted Tokens ({len(output['tokens'])}):")
    print(f"   {output['tokens']}")

    print(f"\n3. Multi-Library Comparison:")
    comp = output["library_comparison"]
    print(f"   Proposed Sentences: {comp['proposed_sentences_count']}")
    print(f"   Standard NLTK Sentences: {comp['nltk_sentences_count']}")
    print(f"   Standard spaCy Sentences: {comp['spacy_sentences_count']}")
    print("=" * 60)
