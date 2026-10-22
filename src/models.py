"""
Models module for Clinical Dictation NLP.

Defines the core model classes:
- ClinicalSentenceSegmenter: Sentence Boundary Detector for clinical dictation.
- ClinicalTokenizer: Domain-specific Tokenizer tailored for clinical text.
"""

import os
import sys
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path when run directly as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nltk.tokenize.punkt import PunktTrainer, PunktSentenceTokenizer

try:
    from src.preprocessing import (
        segment_sentences,
        tokenize_medical_words,
        tokenize_with_spacy,
        tokenize_with_nltk,
        remove_custom_stopwords,
        normalize_tokens,
        detect_clinical_patterns,
        protect_medical_abbreviations,
        restore_protected_text,
        create_clinical_punkt_tokenizer,
        CLINICAL_ABBREVIATIONS,
        CLINICAL_PUNKT_TOKENIZER,
        CLINICAL_SPACY_NLP,
        CLINICAL_NLTK_TOKENIZER
    )
except ModuleNotFoundError:
    from preprocessing import (
        segment_sentences,
        tokenize_medical_words,
        tokenize_with_spacy,
        tokenize_with_nltk,
        remove_custom_stopwords,
        normalize_tokens,
        detect_clinical_patterns,
        protect_medical_abbreviations,
        restore_protected_text,
        create_clinical_punkt_tokenizer,
        CLINICAL_ABBREVIATIONS,
        CLINICAL_PUNKT_TOKENIZER,
        CLINICAL_SPACY_NLP,
        CLINICAL_NLTK_TOKENIZER
    )


class ClinicalSentenceSegmenter:
    """
    Sentence Boundary Detector tailored specifically for clinical dictation transcripts.
    Integrates:
    - Domain-adapted NLTK PunktSentenceTokenizer with clinical abbreviation learning
    - spaCy sentencizer engine support
    - Pre-protection regex sentinel shielding for medical abbreviations and decimals
    - Unpunctuated continuous dictation boundary detection
    """

    def __init__(self, abbreviation_lexicon: Optional[List[str]] = None, engine: str = "hybrid"):
        """
        Initializes the clinical sentence segmenter with optional custom abbreviation lexicon
        and chosen segmentation engine ("hybrid", "punkt", "spacy", "regex").
        """
        self.abbreviation_lexicon = abbreviation_lexicon or []
        self.engine = engine
        self.punkt_tokenizer = create_clinical_punkt_tokenizer(self.abbreviation_lexicon)
        self.spacy_nlp = CLINICAL_SPACY_NLP
        self.trained_boundary_patterns = []

    def fit(self, training_data: List[Dict[str, Any]]) -> "ClinicalSentenceSegmenter":
        """
        Tunes boundary detection parameters and trains the NLTK PunktSentenceTokenizer
        using annotated gold standard clinical records.

        Args:
            training_data (List[Dict[str, Any]]): List of annotated transcript records.

        Returns:
            self
        """
        trainer = PunktTrainer()
        for record in training_data:
            annotations = record.get("annotations", {})
            custom_abbrs = annotations.get("protected_abbreviations", []) + annotations.get("abbreviations", [])
            for abbr in custom_abbrs:
                if abbr not in self.abbreviation_lexicon:
                    self.abbreviation_lexicon.append(abbr)

            # Train Punkt unsupervised boundary learner on gold clinical sentences
            for sent in record.get("expected_sentences", []):
                if sent and sent.strip():
                    trainer.train(sent, finalize=False)

        trainer.finalize_training()

        # Rebuild PunktSentenceTokenizer with learned parameters & clinical abbreviations
        params = trainer.get_params()
        for ab in CLINICAL_ABBREVIATIONS:
            clean_ab = ab.replace(r"\.", ".").rstrip(".").lower()
            params.abbrev_types.add(clean_ab)
            params.abbrev_types.add(clean_ab.replace(".", ""))
        for ab in self.abbreviation_lexicon:
            clean_ab = ab.replace(r"\.", ".").rstrip(".").lower()
            params.abbrev_types.add(clean_ab)
            params.abbrev_types.add(clean_ab.replace(".", ""))

        self.punkt_tokenizer = PunktSentenceTokenizer(params)
        return self

    def segment(self, transcript: str, engine: Optional[str] = None) -> List[str]:
        """
        Detects sentence boundaries and segments the transcript into distinct clinical statements.

        Args:
            transcript (str): Raw or preprocessed clinical dictation text.
            engine (Optional[str]): Override engine ("hybrid", "punkt", "spacy", "regex").

        Returns:
            List[str]: List of segmented clinical sentences.
        """
        selected_engine = engine or self.engine
        return segment_sentences(
            transcript,
            engine=selected_engine,
            punkt_tokenizer=self.punkt_tokenizer
        )

    def segment_with_nltk(self, transcript: str) -> List[str]:
        """
        Directly segments using domain-adapted NLTK PunktSentenceTokenizer.
        """
        return self.segment(transcript, engine="punkt")

    def segment_with_spacy(self, transcript: str) -> List[str]:
        """
        Directly segments using spaCy Sentencizer.
        """
        return self.segment(transcript, engine="spacy")


class ClinicalTokenizer:
    """
    Specialized Tokenizer for clinical text that preserves:
    - Compound measurement units (mg/dL, mmol/L)
    - Vital ratios (120/80, 138/84)
    - Clinical abbreviations with internal dots (p.o., q.i.d.)
    - Hyphenated medical terminology (follow-up)
    - Diagnostic negations during stop-word filtering
    """

    def __init__(
        self,
        preserve_case_for_acronyms: bool = True,
        engine: str = "clinical"
    ):
        """
        Initializes the tokenizer with case-preservation, engine selection, and NLP models.
        Engines: 'clinical' (NLTK RegexpTokenizer with medical patterns),
                 'spacy' (spaCy with medical special cases), 'nltk', or 'regex'.
        """
        self.preserve_case_for_acronyms = preserve_case_for_acronyms
        self.engine = engine
        self.nltk_tokenizer = CLINICAL_NLTK_TOKENIZER
        self.spacy_nlp = CLINICAL_SPACY_NLP

    def tokenize(self, sentence: str, engine: Optional[str] = None) -> List[str]:
        """
        Tokenizes an individual clinical sentence into clinical words, symbols, and entities.

        Args:
            sentence (str): A single clinical sentence.
            engine (Optional[str]): Engine override ('clinical', 'spacy', 'nltk', 'regex').

        Returns:
            List[str]: Extracted list of tokens.
        """
        selected_engine = engine or self.engine
        return tokenize_medical_words(sentence, engine=selected_engine)

    def tokenize_with_spacy(self, sentence: str) -> List[str]:
        """
        Tokenizes sentence using domain-adapted spaCy tokenizer.
        """
        return tokenize_with_spacy(sentence)

    def tokenize_with_nltk(self, sentence: str) -> List[str]:
        """
        Tokenizes sentence using NLTK RegexpTokenizer.
        """
        return tokenize_with_nltk(sentence)

    def get_spacy_doc(self, sentence: str):
        """
        Returns spaCy Doc object for advanced linguistic attributes (lemmas, shapes, POS).
        """
        return self.spacy_nlp(sentence)

    def filter_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Filters non-clinical stopwords while preserving diagnostic negations (no, not, denies).

        Args:
            tokens (List[str]): List of tokens.

        Returns:
            List[str]: Filtered tokens.
        """
        return remove_custom_stopwords(tokens)

    def normalize(self, tokens: List[str]) -> List[str]:
        """
        Normalizes tokens via case folding, canonical expansions, and root mapping.

        Args:
            tokens (List[str]): List of tokens.

        Returns:
            List[str]: Normalized tokens.
        """
        return normalize_tokens(tokens)

    def tokenize_and_normalize(self, sentence: str) -> List[str]:
        """
        Tokenizes, filters non-diagnostic stopwords, and normalizes tokens in a unified pass.

        Args:
            sentence (str): A single clinical sentence.

        Returns:
            List[str]: Processed and normalized tokens.
        """
        raw_tokens = self.tokenize(sentence)
        filtered_tokens = self.filter_stopwords(raw_tokens)
        return self.normalize(filtered_tokens)
