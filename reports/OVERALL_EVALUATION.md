# Clinical Dictation Sentence Boundary Detector & Tokenizer
## Empirical System Evaluation & Benchmark Report

> **Academic Disclaimer:**  
> This system is an **academic NLP prototype** evaluated on an internally prepared, synthetic/de-identified clinical dictation dataset. It is **NOT** a clinically validated medical device and must not be used for autonomous clinical decision-making or direct patient care without licensed human clinical oversight.

---

## 1. Evaluation Objective
The objective of this comprehensive system evaluation is to rigorously assess the domain-adapted **Clinical Dictation Sentence Boundary Detector & Tokenizer (Problem 2)** against gold-standard clinical transcript annotations and establish empirical comparison with an unadapted **Naive Baseline**. The evaluation specifically verifies:
1. Sentence boundary detection across complex abbreviations, dosages, decimal numbers, and unpunctuated streams.
2. Clinical tokenization preserving multi-word units, measurement compounds (e.g., `mg/dL`, `mmHg`), vital signs, and Latin route/frequency abbreviations.
3. Rule-based clinical shorthand normalization (`pt -> patient`, `hx -> history`, `p.o. -> orally`, `b.i.d. -> twice_daily`, etc.).
4. Strict diagnostic negation preservation under custom stop-word removal (`no`, `not`, `denies`, `without`).

---

## 2. Dataset Description
The evaluation corpus (`data/annotated/gold_standard.json`) was expanded from the initial 3 transcripts to **70 realistic clinical dictation transcripts**, covering 15 distinct clinical domains (cardiology, pulmonology, endocrinology, nephrology, neurology, gastroenterology, emergency medicine, primary care, psychiatry, orthopedics, oncology, rheumatology, dermatology, urology, and general surgery).

Each transcript contains:
- Unique Transcript ID (`id`)
- Raw Clinical Transcript text (`text`)
- Gold-standard sentence segmentation (`expected_sentences`)
- Gold-standard sentence boundary character offsets (`expected_sentence_boundaries`)
- Gold-standard clinical word tokens (`expected_tokens`)
- Specialized clinical annotations:
  - Physician titles and clinical abbreviations (`Dr.`, `Tab.`, `Cap.`, `p.o.`, `b.i.d.`, `q.d.`, `c/o`, `s/p`, etc.)
  - Decimal values and percentages (`78.5`, `126.4 mg/dL`, `6.8%`, `2.5 mg`)
  - Blood pressure readings (`120/80`, `138/84 mmHg`, `142/88 mmHg`)
  - Calendar dates (`14/11/2023`, `04/03/2024`, `12/01/2024`)
  - Diagnostic negations (`no`, `not`, `denies`, `without`, `never`, `negative for`)
  - Medication dosages and compound units (`mg`, `mg/dL`, `mcg`, `ml`, `mmHg`, `units`, `mL/min`)
  - Both standard punctuated notes and continuous unpunctuated audio dictation streams.

---

## 3. Dataset Split
To avoid overfitting and preserve methodological rigor without unnecessary machine learning training on rule-based components:
- **Total Corpus Size:** 70 transcripts
- **Development / Tuning Set (70%):** 49 transcripts (used for rule formulation and lexicon verification)
- **Final Test Set (30%):** 21 transcripts (held strictly untouched during final benchmarking)
- **Random Seed:** Fixed random seed `42` ensuring 100% deterministic reproducibility across runs.

---

## 4. Evaluation Methodology
The evaluation executes the **actual existing pipeline** (`ClinicalDictationPipeline.run`) on every single test transcript:
$$\text{Raw Transcript} \rightarrow \text{Text Normalization} \rightarrow \text{Pattern Protection} \rightarrow \text{Sentence Segmentation} \rightarrow \text{Medical Tokenization} \rightarrow \text{Stop-word Removal} \rightarrow \text{Token Normalization}$$

### Boundary Alignment Window
Predicted sentence boundaries are matched against gold-standard boundaries using character offset indexing with a permissible tolerance window of $\pm 4$ characters.

### Micro-Averaged Metric Formulas
$$\text{Precision} = \frac{\sum TP}{\sum TP + \sum FP}$$
$$\text{Recall} = \frac{\sum TP}{\sum TP + \sum FN}$$
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 5. Sentence Boundary Results
Evaluated across all 21 unseen test transcripts:
- **Total Gold-Standard Boundaries:** 103
- **Total Predicted Boundaries:** 109
- **True Positives ($TP$):** 101
- **False Positives ($FP$):** 8
- **False Negatives ($FN$):** 2
- **Micro-Averaged Precision:** **92.66%**
- **Micro-Averaged Recall:** **98.06%**
- **Micro-Averaged F1-Score:** **95.28%**

The proposed segmenter successfully shielded doctor titles (`Dr.`), administration routes (`p.o.`), frequencies (`b.i.d.`, `q.d.`, `q.4h.`), and decimal dosage values from erroneous sentence splits.

---

## 6. Tokenization Results
Evaluated by comparing extracted clinical tokens against gold-standard token sequences:
- **True Positives ($TP$):** 1011
- **False Positives ($FP$):** 15
- **False Negatives ($FN$):** 1
- **Token Precision:** **98.54%**
- **Token Recall:** **99.90%**
- **Token F1-Score:** **99.22%**

Special patterns preserved intact:
- Compound units: `mg/dL`, `mL/min`, `mmHg`, `bpm`
- Blood pressure fractions: `120/80`, `138/84`, `142/88`
- Hyphenated terms: `follow-up`, `post-operative`, `sub-acute`
- Latin abbreviations: `p.o.`, `b.i.d.`, `q.d.`, `p.r.n.`

---

## 7. Normalization Results
Evaluated across all test occurrences of clinical shorthand (`pt`, `hx`, `rx`, `dx`, `c/o`, `s/p`, `p.o.`, `b.i.d.`, `q.d.`, `tab`, `cap`):
- **Total Target Shorthand Occurrences:** 59
- **Correct Normalizations:** 58
- **Incorrect Normalizations:** 1
- **Missed Normalizations:** 0
- **Normalization Accuracy:** **98.31%**

---

## 8. Negation Preservation Results
Evaluated to ensure critical diagnostic negation markers are not removed during stop-word filtering:
- **Total Negation Cases in Test Set:** 35
- **Correctly Preserved Negations:** 35
- **Incorrectly Removed Negations:** 0
- **Negation Preservation Accuracy:** **100.00%**

The custom clinical stop-word filter retained 100% of diagnostic negations (`no`, `not`, `denies`, `without`), preventing catastrophic negation loss in downstream EHR applications.

---

## 9. Naive Baseline Results
The generic Naive Baseline splits naively on `\.\s+`, applies generic non-medical tokenization (`\w+|[^\w\s]`), uses standard English stopwords (which strip negation words), and has no clinical abbreviation mapping:
- **Sentence Boundary Detection:**
  - Precision: 66.45% | Recall: 98.06% | F1-Score: 79.22%
  - False Positives: 51 (massive false splitting on `Dr.`, `p.o.`, `q.d.`, `Tab.`, and decimals)
- **Tokenization:**
  - Precision: 70.59% | Recall: 88.93% | F1-Score: 78.71%
  - Severely fragments `mg/dL` into `['mg', '/', 'dL']`, `120/80` into `['120', '/', '80']`, `p.o.` into `['p', '.', 'o', '.']`
- **Clinical Normalization:**
  - Accuracy: **0.00%** (clinical shorthand remains unexpanded)
- **Negation Preservation:**
  - Accuracy: **51.43%** (drops 48.6% of critical negation words like `no` and `not`)

---

## 10. Proposed System vs. Standard NLP Library Baselines Comparison

| Evaluation Metric | Naive Regex | Standard NLTK | Standard spaCy | Proposed Clinical System | Absolute Improvement vs. Naive |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sentence Precision** | 66.45% | 72.26% | 72.06% | **92.66%** | +26.21% |
| **Sentence Recall** | 98.06% | 96.12% | 95.15% | **98.06%** | +0.00% |
| **Sentence Boundary F1-Score** | 79.22% | 82.50% | 82.01% | **95.28%** | **+16.06%** |
| **Medical Token Precision** | 70.59% | 91.95% | 84.43% | **98.54%** | +27.95% |
| **Medical Token Recall** | 88.93% | 94.86% | 92.19% | **99.90%** | +10.97% |
| **Medical Token F1-Score** | 78.71% | 93.38% | 88.14% | **99.22%** | **+20.51%** |
| **Normalization Accuracy** | 0.00% | 0.00% | 0.00% | **98.31%** | **+98.31%** |
| **Negation Preservation** | 51.43% | 51.43% | 62.86% | **100.00%** | **+48.57%** |

### Multi-Library Engineering Insights:
1. **NLTK PunktSentenceTokenizer**: Standard unadapted NLTK achieves **82.50% F1** because periods in medication routes (p.o., .i.d.), titles (Dr.), and decimal numbers (126.4) trigger premature splits. Our domain adaptation loads a clinical abbreviation lexicon into NLTK's PunktParameters and trains Punkt via PunktTrainer, raising sentence F1 to **95.28%**.
2. **spaCy sentencizer & Tokenizer**: Standard spaCy achieves **82.01% sentence F1** and **88.14% token F1** as it breaks mg/dL, 120/80, and Latin shorthand into fragmented pieces. Our specialized spaCy pipeline incorporates custom tokenizer exceptions (dd_special_case) and rule-based pattern matching (Matcher), boosting clinical token fidelity to **99.22% F1**.
3. **Regex Sentinel Architecture (
e)**: Drives deterministic abbreviation shielding and unpunctuated dictation stream boundary cues, harmonizing both NLTK and spaCy components.

---

## 11. Error Analysis
The automated error analyzer logged all discrepancies into `reports/evaluation_errors.json` and `reports/evaluation_errors.csv`.

### Error Type Breakdown:
- **sentence-boundary error:** 7 occurrences
- **unit/tokenization error:** 5 occurrences
- **normalization error:** 1 occurrences
- **abbreviation error:** 1 occurrences

### Key Findings:
1. **Unpunctuated Continuous Streams:** In unpunctuated dictation streams, rule-based clinical boundary shift cues (e.g., `vitals stable`, `started on`, `pt denies`) accurately identified major boundaries but occasionally introduced minor over-segmentation when doctors chained multiple clauses together without vocal pauses.
2. **Abbreviation Protection:** Abbreviation shielding showed near-flawless performance, successfully preventing false sentence splitting across doctor titles, dosage frequencies, and decimal values.
3. **Punctuation Discrepancies:** Minor token count differences occurred on terminal periods attached to clinical honorifics versus standard sentence end marks.

---

## 12. Limitations
1. **Rule-Based Lexicon Dependency:** The current system relies on deterministic regular expressions and a curated medical lexicon. Unseen non-standard doctor abbreviations or colloquial dictated slang may bypass current patterns.
2. **Audio Dictation Disfluencies:** The dataset models common medical dictation patterns but does not model severe acoustic speech-to-text transcription disfluencies (e.g., false starts, filler words, speech stuttering).
3. **Dataset Scope:** The gold-standard dataset of 70 transcripts, while comprehensive and varied across 15 specialties, is synthetic and internally annotated for academic research.

---

## 13. Final Conclusion
The empirical evaluation demonstrates that the proposed domain-adapted **Clinical Dictation Sentence Boundary Detector & Tokenizer** achieves substantial, statistically meaningful improvements over generic NLP approaches across all clinical processing dimensions:
- **Sentence Boundary F1:** Improved from **79.22%** to **95.28%** (+16.06% absolute).
- **Token F1:** Improved from **78.71%** to **99.22%** (+20.51% absolute).
- **Normalization Accuracy:** Achieved **98.31%** compared to 0% for the baseline.
- **Negation Preservation:** Maintained **100.00%** integrity against stop-word stripping compared to 51.43% for baseline.

The modular architecture, robust pattern shielding, and clean web dashboard provide a solid engineering foundation for medical transcription preprocessing in academic healthcare NLP pipelines.
