"""
Preprocessing module for Clinical Dictation NLP.

Provides production-ready clinical text preprocessing:
- Medical abbreviation protection
- Regex-based clinical pattern detection
- Sentence segmentation (handling abbreviations, decimals, and unpunctuated text)
- Medical word tokenization (preserving units like mg/dL, compound terms)
- Custom stop-word removal (strictly preserving clinical negations)
- Rule-based token normalization (canonical expansions and case normalization)
"""

import re
from typing import List, Dict, Any, Tuple, Optional

# Core NLP library integrations
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters, PunktTrainer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords as nltk_stopwords

import spacy
from spacy.symbols import ORTH
from spacy.matcher import Matcher
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS

# Titles and clinical prefixes that can never end a sentence
TITLE_ABBREVIATIONS = [
    # Physician and clinical titles
    r"Dr\.", r"Drs\.", r"Prof\.", r"Mr\.", r"Mrs\.", r"Ms\.", r"Pt\.", r"RN\.", r"MD\.",
    # Medication formulation prefixes
    r"Tab\.", r"Cap\.", r"Susp\.", r"Inj\.", r"Sol\.", r"Elix\.", r"Stat\.",
    # Comparative/relative terms and units
    r"Vs\.", r"vs\.", r"approx\.", r"min\.", r"hr\.", r"wk\.", r"mo\.", r"yr\.", r"wt\.", r"ht\.",
    # Academic/Latin prefixes
    r"e\.g\.", r"i\.e\.", r"et al\.", r"etc\."
]

# Latin medical routes & frequencies that can end a sentence
ROUTE_FREQUENCY_ABBREVIATIONS = [
    r"p\.o\.", r"p\.r\.n\.", r"q\.i\.d\.", r"b\.i\.d\.", r"t\.i\.d\.", r"q\.d\.",
    r"q\.h\.s\.", r"q\.h\.", r"q\.2h\.", r"q\.4h\.", r"q\.6h\.", r"q\.8h\.", r"q\.12h\.",
    r"q\.a\.m\.", r"q\.p\.m\.",
    r"i\.v\.", r"i\.m\.", r"s\.c\.", r"s\.q\.", r"p\.r\.",
    r"o\.d\.", r"o\.s\.", r"o\.u\."
]

# Combined comprehensive list of medical abbreviations
CLINICAL_ABBREVIATIONS = TITLE_ABBREVIATIONS + ROUTE_FREQUENCY_ABBREVIATIONS

# Capitalized sentence starters that follow a route/frequency abbreviation to mark a new sentence
SENTENCE_STARTERS = {
    "Patient", "Pt", "She", "He", "They", "The", "Plan", "Follow-up", "Follow",
    "Vital", "Vitals", "Denies", "Denied", "Continue", "Continued", "Discharge", "Discharged",
    "Prescribe", "Prescribed", "Recommend", "Recommended", "Start", "Started",
    "Will", "Next", "Repeat", "Return", "Advised", "Blood", "Heart", "Physical",
    "Assessment", "Current", "No", "Dr", "Dr.", "At", "In", "On", "If",
    "Laboratory", "Labs", "Exam", "Examination", "Review", "Please", "History",
    "Creatinine", "Glucose", "Temperature", "Medication", "Medications",
    "This", "Her", "His", "Their", "These", "Those", "There",
    "Administered", "Adjusted", "Maintained", "Initiated", "Scheduled", "Recheck"
}

ROUTE_FREQUENCY_SET = {abbr.replace(r"\.", ".").lower() for abbr in ROUTE_FREQUENCY_ABBREVIATIONS}

# Named constant for unpunctuated text threshold:
# sentence-ending punctuation (. ! ? ;) appearing less than about once per 25 words
UNPUNCTUATED_WORDS_PER_PUNCT_THRESHOLD = 25

# Standard medical acronyms that do not always contain dots
CLINICAL_ACRONYMS = {
    "BP", "HR", "RR", "PRN", "PO", "QID", "BID", "TID", "QD", "ED", "ER", "ICU",
    "CCU", "OR", "EKG", "ECG", "EEG", "CT", "MRI", "CBC", "BMP", "CMP", "BUN",
    "WBC", "RBC", "HGB", "HCT", "PLT", "INR", "PT", "PTT", "CXR", "SOB", "NPO",
    "HTN", "DM", "CAD", "COPD", "CHF", "CKD", "ESRD", "GERD", "UTI", "DVT", "PE"
}

# Clinical shorthand to canonical expansion dictionary
NORMALIZATION_MAP = {
    "pt": "patient",
    "pts": "patients",
    "hx": "history",
    "rx": "prescription",
    "tx": "treatment",
    "dx": "diagnosis",
    "sx": "symptom",
    "c/o": "complaining_of",
    "s/p": "status_post",
    "w/": "with",
    "w/o": "without",
    "b.i.d.": "twice_daily",
    "b.i.d": "twice_daily",
    "bid": "twice_daily",
    "t.i.d.": "three_times_daily",
    "t.i.d": "three_times_daily",
    "tid": "three_times_daily",
    "q.i.d.": "four_times_daily",
    "q.i.d": "four_times_daily",
    "qid": "four_times_daily",
    "p.o.": "orally",
    "p.o": "orally",
    "po": "orally",
    "q.d.": "once_daily",
    "q.d": "once_daily",
    "qd": "once_daily",
    "p.r.n.": "as_needed",
    "p.r.n": "as_needed",
    "prn": "as_needed",
    "i.v.": "intravenous",
    "i.v": "intravenous",
    "iv": "intravenous",
    "tab.": "tablet",
    "tab": "tablet",
    "tabs": "tablets",
    "cap.": "capsule",
    "cap": "capsule",
    "caps": "capsules",
    "ed": "emergency_department",
    "er": "emergency_room"
}

# Critical diagnostic negations that MUST NOT be removed
PRESERVED_NEGATIONS = {
    "no", "not", "none", "neither", "never", "nobody", "nowhere", "nothing",
    "denies", "denied", "denying", "without", "negative", "non", "ruled out", "free of"
}

# Generic non-diagnostic stopwords
GENERIC_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
}

# Stopwords integration from standard list, NLTK, and spaCy
try:
    NLTK_STOPWORDS = set(nltk_stopwords.words("english"))
except Exception:
    try:
        nltk.download("stopwords", quiet=True)
        NLTK_STOPWORDS = set(nltk_stopwords.words("english"))
    except Exception:
        NLTK_STOPWORDS = set()

SPACY_STOPWORDS = set(SPACY_STOP_WORDS)
COMBINED_STOPWORDS = GENERIC_STOPWORDS.union(NLTK_STOPWORDS)
# Final stopword set: strictly excludes any diagnostic negation words
CLINICAL_STOPWORDS = COMBINED_STOPWORDS - PRESERVED_NEGATIONS

DOT_SENTINEL = "§DOT§"

# Token pattern string for clinical tokenization
ABBR_TITLES = r"\b(?:Dr|Drs|Prof|Mr|Mrs|Ms|Pt|RN|MD|Tab|Cap|Susp|Inj|Sol|Elix|Stat|Vs|vs|approx|min|hr|wk|mo|yr|wt|ht)\."
CLINICAL_TOKEN_PATTERN_STR = (
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"                      # Dates
    r"\b\d{2,3}/\d{2,3}\b|"                                     # Blood pressure ratios (120/80)
    r"\b[a-zA-Z]+/[a-zA-Z]+\b|"                                # Compound units / shorthand (mg/dL, c/o, s/p)
    r"\b(?:[a-zA-Z]\.)+(?:[a-zA-Z])?(?=\s|$|[^\w\.])|"         # Abbreviations with internal/ending dots (p.o., q.i.d.)
    rf"{ABBR_TITLES}|"                                         # Titles with dot (Dr., Tab., Cap.)
    r"\b\d+(?:\.\d+)?%?|"                                      # Numbers & decimals with %
    r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)+\b|"                     # Hyphenated words (follow-up)
    r"\b[a-zA-Z0-9]+\b|"                                       # Standard words and alphanumeric tokens
    r"[^\s\w]"                                                 # Punctuation
)

# NLTK-powered clinical regex tokenizer
CLINICAL_NLTK_TOKENIZER = RegexpTokenizer(CLINICAL_TOKEN_PATTERN_STR, flags=re.IGNORECASE)


def create_clinical_punkt_tokenizer(custom_abbreviations: Optional[List[str]] = None) -> PunktSentenceTokenizer:
    """
    Builds a domain-adapted NLTK PunktSentenceTokenizer initialized with clinical abbreviation types.
    """
    params = PunktParameters()
    for abbr in CLINICAL_ABBREVIATIONS:
        clean = abbr.replace(r"\.", ".").rstrip(".").lower()
        params.abbrev_types.add(clean)
        params.abbrev_types.add(clean.replace(".", ""))
    if custom_abbreviations:
        for abbr in custom_abbreviations:
            clean = abbr.replace(r"\.", ".").rstrip(".").lower()
            params.abbrev_types.add(clean)
            params.abbrev_types.add(clean.replace(".", ""))
    return PunktSentenceTokenizer(params)


CLINICAL_PUNKT_TOKENIZER = create_clinical_punkt_tokenizer()


def create_clinical_spacy_nlp():
    """
    Builds a domain-adapted spaCy English pipeline with special cases for medical units, ratios, and abbreviations.
    """
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    special_cases = [
        "mg/dL", "mmol/L", "mL/min", "mmHg", "mEq/L", "bpm", "mg/dl", "mcg/mL",
        "120/80", "138/84", "140/90", "130/80", "142/88", "118/76", "124/82", "132/86", "128/82", "136/84", "110/70",
        "p.o.", "q.i.d.", "b.i.d.", "t.i.d.", "q.d.", "p.r.n.", "i.v.", "i.m.", "s.c.",
        "Dr.", "Drs.", "Tab.", "Cap.", "Pt.", "RN.", "MD.",
        "follow-up", "sub-acute", "post-operative", "pre-operative", "well-appearing",
        "c/o", "s/p", "w/", "w/o"
    ]
    for sc in special_cases:
        nlp.tokenizer.add_special_case(sc, [{ORTH: sc}])
    return nlp


CLINICAL_SPACY_NLP = create_clinical_spacy_nlp()


def protect_medical_abbreviations(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Protects medical abbreviations, decimal values, and dates from premature sentence splitting
    by substituting periods with a unique sentinel token.

    - Titles that can never end a sentence (Dr., Drs., Tab., Cap., etc.) are always protected.
    - Routes & frequencies that can end a sentence (p.o., b.i.d., q.d., etc.) preserve their
      final period as a true sentence boundary when the following word is capitalized and in
      SENTENCE_STARTERS. Internal dots remain protected.
    - Decimal values (e.g., 2.5, 126.4) and dates (e.g., 14.11.2023) are protected.

    Args:
        text (str): Input clinical text.

    Returns:
        Tuple[str, Dict[str, str]]: (Protected text string, Sentinel restoration mapping).
    """
    mapping = {}
    protected = text

    # 1. Protect clinical abbreviations sorted descending by length
    all_abbrs_sorted = sorted(CLINICAL_ABBREVIATIONS, key=lambda x: len(x.replace(r"\.", ".")), reverse=True)
    pattern = re.compile(rf"\b({'|'.join(all_abbrs_sorted)})", re.IGNORECASE)

    def replace_abbrev(match):
        orig = match.group(0)
        token_key = f"§ABBR_{len(mapping)}§"
        mapping[token_key] = orig
        clean = orig.replace(r"\.", ".").lower()

        # Route/frequency abbreviations can end a sentence if followed by a capitalized sentence starter
        if clean in ROUTE_FREQUENCY_SET:
            following = match.string[match.end():]
            m_next = re.match(r"\s+([A-Za-z0-9\-]+)", following)
            if m_next:
                next_word = m_next.group(1)
                if next_word[0].isupper() and (next_word in SENTENCE_STARTERS or next_word.rstrip(".") in SENTENCE_STARTERS):
                    if orig.endswith("."):
                        return orig[:-1].replace(".", DOT_SENTINEL) + "."
        return orig.replace(".", DOT_SENTINEL)

    protected = pattern.sub(replace_abbrev, protected)

    # 2. Protect decimal numbers (e.g., 2.5, 126.4, 0.75)
    decimal_pattern = re.compile(r"(\b\d+)\.(\d+\b)")
    protected = decimal_pattern.sub(rf"\1{DOT_SENTINEL}\2", protected)

    # 3. Protect dates formatted with dots (e.g., 14.11.2023)
    date_dot_pattern = re.compile(r"(\b\d{1,2})\.(\d{1,2})\.(\d{2,4}\b)")
    protected = date_dot_pattern.sub(rf"\1{DOT_SENTINEL}\2{DOT_SENTINEL}\3", protected)

    return protected, mapping


def restore_protected_text(text: str, mapping: Optional[Dict[str, str]] = None) -> str:
    """
    Restores protected sentinels back to standard periods.
    """
    restored = text.replace(DOT_SENTINEL, ".")
    if mapping:
        for k, v in mapping.items():
            restored = restored.replace(k, v)
    return restored


def detect_clinical_patterns(text: str) -> List[Dict[str, Any]]:
    """
    Detects clinical patterns using regular expressions (dosages, blood pressure,
    dates, decimal values, abbreviations, and negations).

    Args:
        text (str): Input text.

    Returns:
        List[Dict[str, Any]]: Detected pattern entities with category, match, and indices.
    """
    patterns = []

    # Blood Pressure (e.g., 'BP 120/80 mmHg', '138/84 mmHg', '120/80')
    bp_regex = re.compile(
        r"(?:BP\s*)?(\b\d{2,3}\s*/\s*\d{2,3}(?:\s*mmHg)?\b)",
        re.IGNORECASE
    )
    for m in bp_regex.finditer(text):
        matched = m.group(0).strip()
        # Avoid simple dates like 12/04/2023 being captured as BP
        if not re.search(r"/\d{4}", matched):
            patterns.append({
                "category": "blood_pressure",
                "value": matched,
                "start": m.start(),
                "end": m.end()
            })

    # Calendar Dates (e.g., '14/11/2023', '04-12-2023', '14.11.2023')
    date_regex = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",
        re.IGNORECASE
    )
    for m in date_regex.finditer(text):
        patterns.append({
            "category": "date",
            "value": m.group(0),
            "start": m.start(),
            "end": m.end()
        })

    # Dosages and medical units (e.g., '500 mg', '2.5 ml', '12.5 units', '110 mg/dL', '81 mg')
    dosage_regex = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:mg/dL|mmol/L|mcg|mg|g|kg|ml|mL|units|mEq|bpm|mmHg)\b",
        re.IGNORECASE
    )
    for m in dosage_regex.finditer(text):
        patterns.append({
            "category": "dosage_or_measurement",
            "value": m.group(0),
            "start": m.start(),
            "end": m.end()
        })

    # Decimal values not already part of dosages
    decimal_regex = re.compile(r"\b\d+\.\d+(?:%)?\b")
    for m in decimal_regex.finditer(text):
        patterns.append({
            "category": "decimal_value",
            "value": m.group(0),
            "start": m.start(),
            "end": m.end()
        })

    # Clinical abbreviations & titles
    abbrev_regex = re.compile(
        r"\b(?:Dr\.|Drs\.|Pt\.|hx\b|rx\b|c/o\b|s/p\b|p\.o\.|b\.i\.d\.|q\.i\.d\.|q\.d\.|Tab\.|Cap\.|ED\b|BP\b|HR\b)",
        re.IGNORECASE
    )
    for m in abbrev_regex.finditer(text):
        patterns.append({
            "category": "clinical_abbreviation",
            "value": m.group(0),
            "start": m.start(),
            "end": m.end()
        })

    # Diagnostic negations
    negation_regex = re.compile(
        r"\b(?:no|not|denies|denied|without|negative\s+for|ruled\s+out)\b",
        re.IGNORECASE
    )
    for m in negation_regex.finditer(text):
        patterns.append({
            "category": "clinical_negation",
            "value": m.group(0),
            "start": m.start(),
            "end": m.end()
        })

    # Sort patterns by position
    patterns.sort(key=lambda p: p["start"])
    return patterns


def is_unpunctuated(text: str, threshold: int = UNPUNCTUATED_WORDS_PER_PUNCT_THRESHOLD) -> bool:
    """
    Determines whether a clinical text transcript appears unpunctuated.
    A transcript is deemed unpunctuated if sentence-ending punctuation (. ! ? ;)
    appears less than about once per threshold words in the text.
    Punctuated transcripts skip dictation boundary cues entirely.
    """
    if not text or not text.strip():
        return False
    words = text.split()
    if not words:
        return False
    punct_count = sum(text.count(p) for p in [".", "!", "?", ";"])
    if punct_count == 0:
        return True
    return (len(words) / punct_count) > threshold


def segment_sentences(
    text: str,
    engine: str = "hybrid",
    punkt_tokenizer: Optional[PunktSentenceTokenizer] = None
) -> List[str]:
    """
    Segments clinical dictation transcripts into distinct sentences.
    Employs a multi-stage clinical segmentation strategy using:
    1. Medical abbreviation, decimal, and date protection (via regex sentinels).
    2. Dictation boundary cue detection for unpunctuated streams (via regex).
    3. Sentence boundary detection using domain-adapted NLTK PunktSentenceTokenizer,
       spaCy Sentencizer, or standard regex.
    4. Restoration of protected clinical entities and punctuation formatting.

    Args:
        text (str): Raw or preprocessed clinical text.
        engine (str): Segmentation engine: "hybrid" (default, NLTK Punkt + regex sentinels),
                      "punkt" (NLTK Punkt), "spacy" (spaCy Sentencizer), or "regex".
        punkt_tokenizer (Optional[PunktSentenceTokenizer]): Custom Punkt tokenizer instance.

    Returns:
        List[str]: List of segmented clinical sentences.
    """
    if not text or not text.strip():
        return []

    cleaned_text = re.sub(r"\s+", " ", text.strip())

    # Step 1: Protect abbreviations and decimal numbers
    protected_text, mapping = protect_medical_abbreviations(cleaned_text)

    # Step 2: Handle unpunctuated clinical dictation streams
    # Apply cues only to text that looks unpunctuated (sentence-ending punctuation appears
    # less than once per UNPUNCTUATED_WORDS_PER_PUNCT_THRESHOLD words).
    # Punctuated transcripts skip the cues entirely.
    # Scoped case-sensitivity ensures Dr. Name checks capitalization while cues stay case-insensitive.
    # Generic cues ('reports no', 'next appointment', 'started on') are removed so they cannot fire mid-sentence.
    if is_unpunctuated(protected_text, UNPUNCTUATED_WORDS_PER_PUNCT_THRESHOLD):
        dictation_boundary_cues = [
            r"(?<=[^\.\?\!;\n])\s+(?=(?i:pt\s+admitted|patient\s+admitted)\b)",
            r"(?<=[^\.\?\!;\n])\s+(?=(?i:vitals\s+stable|vital\s+signs:?)\b)",
            r"(?<=[^\.\?\!;\n])\s+(?=(?i:Dr§DOT§\s+)(?-i:[A-Z][a-z]+)\s+(?i:consulted|evaluated|noted|advised)\b)",
            r"(?<=[^\.\?\!;\n])\s+(?=(?i:pt\s+denies|patient\s+denies)\b)",
            r"(?<=[^\.\?\!;\n])\s+(?=(?i:discharge\s+planned|follow-up\s+scheduled)\b)"
        ]
        for cue in dictation_boundary_cues:
            protected_text = re.sub(cue, ". ", protected_text)

    # Step 3: Sentence splitting using chosen NLP engine
    tokenizer_to_use = punkt_tokenizer or CLINICAL_PUNKT_TOKENIZER

    if engine in ("hybrid", "punkt"):
        # Primary: Domain-adapted NLTK PunktSentenceTokenizer
        raw_sentences = tokenizer_to_use.tokenize(protected_text)
    elif engine == "spacy":
        # spaCy Sentencizer
        doc = CLINICAL_SPACY_NLP(protected_text)
        raw_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        # Pure regex splitting fallback
        raw_sentences = re.split(r"(?<=[.!?])\s+", protected_text)

    # Step 4: Restore original dots and clean sentences
    sentences = []
    for s in raw_sentences:
        restored = restore_protected_text(s.strip(), mapping)
        if restored:
            # Ensure proper clinical period at the end if it was an unpunctuated stream segment
            if not re.search(r"[.!?:]$", restored):
                restored = restored + "."
            sentences.append(restored)

    return sentences


def tokenize_medical_words(sentence: str, engine: str = "clinical") -> List[str]:
    """
    Tokenizes a clinical sentence into discrete word and symbol tokens.

    Supports multiple tokenization engines:
    - "clinical" / "nltk": Uses NLTK RegexpTokenizer with domain-adapted patterns.
    - "spacy": Uses domain-adapted spaCy tokenizer with clinical special cases.
    - "regex": Standard regex findall.

    Args:
        sentence (str): Input sentence.
        engine (str): Tokenization engine ("clinical", "nltk", "spacy", "regex").

    Returns:
        List[str]: List of extracted clinical tokens.
    """
    if not sentence or not sentence.strip():
        return []

    if engine == "spacy":
        doc = CLINICAL_SPACY_NLP(sentence)
        return [token.text for token in doc if not token.is_space]
    elif engine in ("clinical", "nltk"):
        tokens = CLINICAL_NLTK_TOKENIZER.tokenize(sentence)
        return [t.strip() for t in tokens if t.strip()]
    else:
        tokens = re.findall(CLINICAL_TOKEN_PATTERN_STR, sentence, re.IGNORECASE)
        return [t.strip() for t in tokens if t.strip()]


def tokenize_with_spacy(sentence: str) -> List[str]:
    """
    Tokenizes clinical sentence using domain-adapted spaCy tokenizer.
    """
    return tokenize_medical_words(sentence, engine="spacy")


def tokenize_with_nltk(sentence: str) -> List[str]:
    """
    Tokenizes clinical sentence using NLTK RegexpTokenizer.
    """
    return tokenize_medical_words(sentence, engine="nltk")


def remove_custom_stopwords(tokens: List[str], custom_stopwords: Optional[List[str]] = None) -> List[str]:
    """
    Removes general stopwords while strictly preserving clinically critical
    negations (e.g., 'no', 'not', 'denies', 'without', 'never').

    Args:
        tokens (List[str]): Extracted tokens.
        custom_stopwords (Optional[List[str]]): Optional stopword override list.

    Returns:
        List[str]: Filtered tokens with diagnostic negations intact.
    """
    stop_set = set(custom_stopwords) if custom_stopwords is not None else CLINICAL_STOPWORDS
    # Filter while preserving any token in PRESERVED_NEGATIONS or uppercase clinical acronyms
    filtered = []
    for token in tokens:
        lower_token = token.lower()
        if lower_token in PRESERVED_NEGATIONS:
            filtered.append(token)
        elif lower_token in stop_set:
            continue
        else:
            filtered.append(token)
    return filtered


def normalize_tokens(tokens: List[str]) -> List[str]:
    """
    Normalizes clinical tokens via case folding, abbreviation mapping, and canonical standardization.
    Preserves uppercase medical acronyms (e.g., 'BP', 'HR', 'ED').

    Args:
        tokens (List[str]): List of tokens.

    Returns:
        List[str]: Normalized tokens.
    """
    normalized = []
    for token in tokens:
        clean = token.strip()
        if not clean:
            continue

        lower = clean.lower()

        # Check dictionary of clinical expansions
        if lower in NORMALIZATION_MAP:
            normalized.append(NORMALIZATION_MAP[lower])
        elif clean.isupper() and len(clean) >= 2:
            # Preserve medical acronym casing (e.g. BP, HR, ED, CBC)
            normalized.append(clean)
        else:
            # Standard lowercase normalization
            # Strip trailing period if it is an honorific like Dr. -> dr
            if lower == "dr.":
                normalized.append("doctor")
            else:
                normalized.append(lower)

    return normalized
