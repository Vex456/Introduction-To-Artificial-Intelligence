# Laboratory 02 Report — Web Scraping, Feature Engineering, and Exploratory Data Analysis (EDA)

**Course:** Introduction to Artificial Intelligence (AI–101L)  
**Instructor:** Dr. Muhammad Awais  
**Student Name / ID:** VEX  
**Date:** October 2026  

---

## 1. Executive Summary

This laboratory implements an end-to-end data acquisition and analysis pipeline: fetching unstructured blog posts via HTTP GET requests, parsing HTML structure with BeautifulSoup, performing linguistic feature extraction (tokenization, sentence segmentation, named entity recognition, and POS tagging) using **spaCy**, quantifying sentiment using **TextBlob**, and conducting Exploratory Data Analysis (EDA) through statistical visualizations and **TF–IDF** term importance analysis.

All requirements outlined in the Lab 02 deliverables and assessment rubric have been met:
1. Complete, executed Jupyter Notebook with all execution outputs visible.
2. Structured dataset persisted as `articles.csv` and cached raw crawl as `articles_raw.csv`.
3. High-resolution figure exports (`figures/*.png`).
4. Analytical written report interpreting results, acknowledging methodological limits, and comparing vectorised vs. native computational performance.

---

## 2. Task 2.1 — NumPy Vectorisation Timing Comparison

### Benchmark Description
To quantify why numerical computing in machine learning mandates vectorised operations, a sum over $10^7$ ($10,000,000$) uniformly distributed random floating-point values in $[0, 1)$ was executed using:
1. **NumPy Vectorised Sum (`np.sum`)**: Single compiled C-level operation operating over contiguous memory.
2. **Native Python Loop (`for val in data: manual_sum += val`)**: Ten million interpreted bytecode iterations with dynamic type checks and pointer dereferencing.

### Benchmark Results

| Method | Total Sum Result | Elapsed Time (seconds) | Speedup Factor |
|---|---|---|---|
| **NumPy (`np.sum`)** | $4,998,987.38$ | **0.01642 s** | **328× Faster** |
| **Manual Python Loop** | $4,998,987.38$ | **5.38298 s** | Baseline ($1.0\times$) |

### Theoretical Interpretation
Both methods yield identical sums to high precision ($4,998,987.38$). However, NumPy executes **328 times faster**.
- In standard CPython, every iteration of the `for` loop incurs significant interpreter overhead: type checking the float object, incrementing reference counts, and fetching next elements via non-contiguous pointer chasing.
- NumPy allocates data in contiguous C-style memory blocks (`strided arrays`) and leverages vectorized SIMD (Single Instruction, Multiple Data) CPU instructions and compiled internal loops. This principle underpins every high-performance tensor and data processing library used throughout this course (pandas, scikit-learn, PyTorch, TensorFlow).

---

## 3. Task 2.2 — Web Scraping & Methodological Soundness

### Scraping Protocol & Graceful Failure Handling
Five target article URLs from `techncruncher.blogspot.com` were processed. The acquisition logic was implemented with robust defensive design:
1. **Polite Request Headers:** A realistic browser `User-Agent` header (`Mozilla/5.0...`) was attached to avoid blocking and ensure standard page rendering.
2. **Timeout Enforcement:** Requests specified `timeout=10` to avoid hanging indefinitively on dead connections.
3. **HTTP Error Handling:** Wrapped in `try-except` blocks with `res.raise_for_status()`, capturing connectivity errors and HTTP status exceptions without crashing the pipeline.
4. **Rate Limiting:** A deliberate pause (`time.sleep(1)`) between successive requests was maintained to respect server load.
5. **Quality Filtering:** Target content was located by CSS selectors (`div.post-body.entry-content`). Articles with body length under 100 characters were automatically flagged and rejected (`[SKIP] Too short: ...`), safeguarding downstream models from empty/malformed text.
6. **Data Caching & Reproducibility:** The raw scraped data was immediately saved to `articles_raw.csv`. A fallback mechanism loads this cached scrape if network requests fail or offline reproducibility is required.

---

## 4. Task 2.3 & 2.4 — Linguistic Feature Engineering & EDA

### Feature Extraction
Using `en_core_web_sm` (spaCy) and `TextBlob`, the following metrics were computed for each article:
- `title_length`: Character length of the article title.
- `num_tokens`: Total token count (words and punctuation).
- `num_sentences`: Sentence count detected via dependency parsing.
- `num_entities`: Named entities recognized (`ORG`, `PRODUCT`, `PERSON`, `DATE`, etc.).
- `num_nouns`: Count of tokens with part-of-speech tag `NOUN`.
- `sentiment_polarity`: Sentiment orientation from $-1.0$ (highly negative) to $+1.0$ (highly positive).
- `sentiment_subjectivity`: Subjectivity score from $0.0$ (objective fact) to $1.0$ (subjective opinion).

The finalized structured dataset was saved as `articles.csv`.

### Exploratory Visualizations & Figure Interpretation

#### Figure 1: Distribution of Article Title Lengths (`figures/title_length_distribution.png`)
- **Observation:** Article title lengths cluster around 50–70 characters, typical for search-engine-optimized (SEO) blog posts designed to avoid truncation on SERPs (Search Engine Result Pages).
- **Interpretation:** Titles are concise and keyword-dense (e.g., *"Top 10 AI Tools..."*), directly addressing content creators.

#### Figure 2: Distribution of Sentiment Polarity (`figures/sentiment_polarity_distribution.png`)
- **Observation:** Sentiment polarity is strictly non-negative, concentrated between $+0.15$ and $+0.35$.
- **Interpretation:** The scraped articles are promotional technology reviews and listicles. Promotional affiliate content rarely exhibits negative sentiment; instead, it maintains mild-to-moderate enthusiasm without extreme hyperbole, designed to build reader trust while encouraging tool adoption.

#### Figure 3: Pairwise Relationships of Derived Features (`figures/feature_pairplot.png`)
- **Observation:** Strong positive linear correlation is apparent between `num_tokens`, `num_sentences`, and `num_nouns`. Longer articles naturally feature proportionally more sentences and substantive nominal entities.
- **Interpretation:** Noun density remains fairly stable across articles (~25–30% of total tokens), confirming standard expository English prose. Named entities (`num_entities`) scale non-linearly with article depth, peaking in articles detailing multiple specific software platforms (e.g., LimeWire AI Studio).

#### Figure 4: Top 20 TF–IDF Terms (`figures/tfidf_top20_terms.png`)
- **Observation:** The highest-ranked terms across documents are `"content"`, `"ai"`, `"tool"`, `"month"`, `"limewire"`, `"creators"`, and `"plan"`.
- **Interpretation:**
  - TF–IDF (Term Frequency–Inverse Document Frequency) rewards terms that appear frequently within individual articles while balancing domain specificity.
  - The dominance of `"content"`, `"ai"`, and `"creators"` reflects the core subject domain (generative AI content workflows).
  - Terms like `"month"`, `"plan"`, and `"free"` reflect common pricing-tier discussions characteristic of SaaS software reviews.
  - In Exercise 4, removing English stop words proved essential: retaining stop words causes universal grammatical particles (`"and"`, `"the"`, `"to"`, `"for"`, `"it"`, `"of"`, `"is"`, `"in"`, `"with"`) to saturate the top ranks, completely masking domain-specific semantic signal.

---

## 5. Statistical Considerations: Exercise 3 Discussion

### Why a KDE Over Five Points is Not a Population Distribution
In Exercise 3, we evaluated why fitting a Kernel Density Estimation (KDE) over 4–5 sample observations cannot be treated as a true population distribution:
1. **Dominance of Single Observations:** In a sample of $N=4$, each kernel function contributes $25\%$ of the overall density area. The resulting curve reflects idiosyncratic sample noise rather than population characteristics.
2. **Bandwidth Artifacts:** The curve's mode and curvature are predominantly dictated by the smoothing parameter (bandwidth $h$), not the underlying data generating process.
3. **Statistical Inference Limits:** Law of large numbers and central limit theorem guarantees require adequate sample sizes. Claiming population density bounds or tail probabilities from 4–5 data points is statistically invalid; the KDE must be regarded purely as an exploratory sample graphic.

---

## 6. Deliverables Checklist

| Deliverable | Location | Status | Notes |
|---|---|---|---|
| **Executed Notebook** | `Lab 02 - Web Scraping and EDA/Lab_02_Web_Scraping_and_EDA.ipynb` | Complete | All 15 code cells executed, clean outputs |
| **Engineered Dataset** | `Lab 02 - Web Scraping and EDA/articles.csv` | Complete | Scraped corpus with all engineered NLP features |
| **Raw Scrape Cache** | `Lab 02 - Web Scraping and EDA/articles_raw.csv` | Complete | Raw offline cached data |
| **Exported Figures** | `Lab 02 - Web Scraping and EDA/figures/*.png` | Complete | 4 high-res figures exported at 300 DPI |
| **Written Report** | `Lab 02 - Web Scraping and EDA/report.md` | Complete | Rubric-compliant evaluation and interpretation |

---
*End of Report.*
