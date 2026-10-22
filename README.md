# Clinical Dictation Sentence Boundary Detector & Tokenizer

**Assignment:** MT-AIML Mini-Project (Problem 2)  
**Domain:** Biomedical / Clinical Natural Language Processing  

---

## 1. Project Title
**Clinical Dictation Sentence Boundary Detector & Tokenizer**  
*A Domain-Adapted Rule-Based and Machine Learning Framework for Segmenting and Tokenizing Unstructured Medical Dictation Streams.*

---

## 2. Problem Statement
Clinical audio dictation transcripts produced by physicians and medical scribes frequently lack consistent punctuation and standard grammatical boundaries. Furthermore, conventional off-the-shelf sentence splitters (e.g., standard Punkt or whitespace tokenizers) routinely fail on clinical narratives due to:
- Frequent honorific and clinical abbreviations (e.g., `Dr.`, `Pt.`, `Hx`, `Rx`, `Tab.`, `c/o`, `s/p`).
- Latin medical frequency and route shorthand (e.g., `p.o.`, `q.i.d.`, `b.i.d.`, `q.d.`, `p.r.n.`).
- Numerical expressions and decimal lab values (e.g., `126.4 mg/dL`, `2.5 mg`, `0.75 ml`).
- Vital signs and multi-unit compound readings (e.g., `BP 120/80 mmHg`, `HR 74 bpm`).
- Calendar and examination date formats (e.g., `12/04/2023`, `04-12-2023`).
- Crucial clinical negations (e.g., `denies chest pain`, `no shortness of breath`) that must not be stripped as generic stop words.

Inaccurate sentence boundary detection and naive tokenization propagate cascading errors into downstream clinical decision-support systems, named entity recognition (NER), and automated electronic health record (EHR) coding.

---

## 3. Objectives
1. **Clinical Abbreviation Protection:** Design pattern protection algorithms to shield medical abbreviations and measurement units from erroneous sentence splits.
2. **Robust Sentence Boundary Detection:** Develop a dedicated sentence segmenter capable of parsing both punctuated and unpunctuated clinical dictation transcripts into coherent clinical statements.
3. **Medical Word Tokenization:** Construct a domain-specific tokenizer that isolates medical entities, preserves compound units (e.g., `mg/dL`), and respects clinical hyphenation and symbols.
4. **Negation-Preserving Normalization:** Build custom stop-word filtering and token normalization rules that retain critical diagnostic negations (`no`, `not`, `denies`, `without`).
5. **Empirical Evaluation & Benchmarking:** Quantify segmentation and tokenization performance against an annotated gold standard corpus using Precision, Recall, and F1-score, establishing comparison against baseline standard NLP tools.
6. **Interactive Demonstration:** Provide a clean Flask web interface allowing users to input raw clinical dictations and inspect detected boundaries, extracted tokens, and evaluation metrics.

---

## 4. Planned Features
- **Abbreviation & Entity Shielding:** Regex-based pre-segmentation masking for doctor titles, administration routes, dosages, and vital measurements.
- **Dual Segmentation Engine:** Hybrid rule-based and regular-expression engine tailored for clinical transcription text.
- **Clinical Token Extraction & Normalization:** Preserves medical units, standardizes common dictation spelling variants, and maps clinical shorthand.
- **Negation Guarding:** Specialized clinical stop-word filter that protects diagnostic negation markers.
- **Metrics Dashboard:** In-app computation and visualization of Precision, Recall, and F1-score against gold standard test sets.
- **RESTful Flask API & Web UI:** Clean, responsive web dashboard for clinical transcript exploration and testing.

---

## 5. Technology Stack
- **Language:** Python 3.10+
- **Web Framework:** Flask
- **Core NLP Libraries:** NLTK, spaCy, regex
- **Data & Evaluation:** Pandas, Scikit-learn
- **Visualization:** Matplotlib
- **Testing:** Pytest
- **Frontend:** HTML5, CSS3 (Academic clean design)

---

## 6. Folder Structure
```
Clinical_Dictation_NLP/
│
├── docs/
│   ├── Architecture_and_Methodology.docx # System architecture & methodology documentation
│   └── Evaluation_Report.docx            # Empirical results & benchmark report
│
├── src/
│   ├── __init__.py                       # Package initialization
│   ├── preprocessing.py                  # Abbreviation protection, regex, stop-words
│   ├── models.py                         # ClinicalSentenceSegmenter & ClinicalTokenizer
│   ├── pipeline.py                       # End-to-end NLP orchestration pipeline
│   └── evaluation.py                     # Precision, Recall, F1, and baseline comparison
│
├── data/
│   ├── raw_transcripts/
│   │   ├── transcript1.txt               # Sample transcript with medical abbreviations
│   │   ├── transcript2.txt               # Sample with dosages, BP, dates, and decimals
│   │   └── transcript3.txt               # Sample with unpunctuated dictation streams
│   │
│   └── annotated/
│       └── gold_standard.json            # Ground truth sentences and entity annotations
│
├── tests/
│   ├── __init__.py                       # Test package initialization
│   └── test_tokenizer.py                 # Pytest unit tests for clinical edge cases
│
├── templates/
│   └── index.html                        # Starter web interface for transcript testing
│
├── static/
│   └── style.css                         # Academic UI styling
│
├── requirements.txt                      # Project dependencies
├── README.md                             # Project overview and instructions
└── run.py                                # Starter Flask application entry point
```

---

## 7. Installation Instructions

1. **Navigate to the Project Directory:**
   ```bash
   cd Clinical_Dictation_NLP
   ```

2. **Create and Activate a Virtual Environment (Recommended):**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Windows (CMD)
   python -m venv venv
   .\venv\Scripts\activate.bat

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 8. How to Run the Flask Application

Run the starter application entry point:
```bash
python run.py
```
By default, the server will launch at:
```
http://127.0.0.1:5000
```
Open this address in any modern web browser to access the demonstration interface.

---

## 9. Planned Evaluation Methodology
The evaluation methodology will validate segmentation and tokenization performance against human-annotated clinical dictations stored in `data/annotated/gold_standard.json`.

1. **Boundary-Level Metrics:**
   - **True Positives ($TP$):** Predicted sentence boundaries that coincide exactly with ground truth boundaries.
   - **False Positives ($FP$):** Incorrect sentence splits introduced within clinical abbreviations, decimals, or continuous statements.
   - **False Negatives ($FN$):** Missed sentence boundaries in unpunctuated or shorthand dictation streams.

2. **Formulas:**
   $$\text{Precision} = \frac{TP}{TP + FP}$$
   $$\text{Recall} = \frac{TP}{TP + FN}$$
   $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

3. **Benchmarking:**
   - Baseline: Standard NLTK `sent_tokenize` / standard regex whitespace tokenization.
   - Proposed: `ClinicalSentenceSegmenter` + `ClinicalTokenizer` with abbreviation shielding.

---

## 10. Future Implementation Steps
- [x] Implement regex-based clinical abbreviation detection and shielding in `src/preprocessing.py`.
- [x] Implement `ClinicalSentenceSegmenter` and `ClinicalTokenizer` in `src/models.py`.
- [x] Wire end-to-end execution flow in `src/pipeline.py`.
- [x] Implement metric calculations ($P$, $R$, $F_1$) and baseline benchmark in `src/evaluation.py`.
- [x] Implement unit tests in `tests/test_tokenizer.py` and ensure all test cases pass.
- [x] Connect Flask `/analyze` endpoint to pipeline output and display live results in `templates/index.html`.
- [x] Generate final architecture diagram and evaluation documentation in `docs/`.
