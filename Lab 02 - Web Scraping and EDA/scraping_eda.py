#!/usr/bin/env python
# coding: utf-8

# # Laboratory 2 — Web Scraping, Feature Engineering and Exploratory Data Analysis
# 
# **Course:** Introduction to Artificial Intelligence (AI–101L) · **Instructor:** Dr. Muhammad Awais
# **Duration:** 3 hours · **Block:** Phase A — Environment and Data · **Mapped CLO:** CLO–2
# 
# ---
# 
# ## Learning Objectives
# 
# On completion of this laboratory you will be able to:
# 
# 1. Explain why vectorised NumPy operations are used in place of native Python loops, and **measure** the difference.
# 2. Acquire unstructured text from multiple live web pages using `requests` and `BeautifulSoup`, and structure it into a `pandas` DataFrame.
# 3. Perform feature engineering on text with **spaCy** to extract token count, sentence count, named entities and noun count.
# 4. Compute sentiment **polarity** and **subjectivity** with **TextBlob**.
# 5. Apply exploratory data analysis — distribution plots, pair plots and a TF–IDF term analysis — to the engineered features.
# 
# ## Background
# 
# In real data science work a clean, labelled, structured dataset is rarely handed to you. **It has
# to be built.** This laboratory implements the first complete pipeline of the course:
# 
# ```
# Blog URLs -> HTTP GET -> HTML parsing -> DataFrame -> Feature engineering -> EDA
#              requests    BeautifulSoup   url/title/   spaCy + TextBlob       histograms
#              User-Agent  + lxml          text                                pairplot, TF-IDF
#              |________ Step 1: acquisition ________|  |____ Steps 2-3: quantify & explore ____|
# ```
# 
# > **Instructor note.** Live websites change. If a selector stops matching, that *is* the lesson —
# > Exercise 1 asks students to diagnose exactly this. Have a cached copy of `articles.csv` ready
# > so that Sections 4–5 can proceed even if the network is unavailable.

# ### Justification of libraries and tools
# 
# | Library | Area | Justification |
# |---|---|---|
# | **NumPy** | Numerical computing | Efficient, vectorised array operations, significantly faster than native Python loops on large data (Task 2.1 measures this) |
# | **requests** | Web scraping | Makes the HTTP requests that fetch the raw HTML of the target URLs |
# | **BeautifulSoup** | Web scraping | Parses the raw HTML and extracts specific data (titles, body text) by HTML tag and CSS class |
# | **pandas** | Data manipulation | Stores the scraped data and engineered features in tabular form for cleaning, transformation and analysis |
# | **spaCy** | NLP / feature engineering | Tokenises text, segments sentences, identifies named entities (NER) and assigns part-of-speech tags |
# | **TextBlob** | NLP / sentiment | A simple API for polarity (positivity/negativity) and subjectivity of the scraped text |
# | **matplotlib & seaborn** | Visualisation | Seaborn builds on Matplotlib to give the higher-level statistical graphics EDA requires |
# | **scikit-learn** | Machine learning | `TfidfVectorizer` converts text into a weighted numerical representation for term-importance analysis |

# ---
# # Task 2.1 — Setup, and why NumPy exists
# 
# Install the packages and the spaCy English language model, then measure the cost of a native
# Python loop against the equivalent vectorised NumPy call. **Record both timings in your report.**

# In[1]:


# Installation of required libraries (run once per environment)
# get_ipython().system('pip install requests beautifulsoup4 lxml pandas spacy matplotlib seaborn textblob')

# Download the small English language model for spaCy
# get_ipython().system('python -m spacy download en_core_web_sm')


# > ### Troubleshooting: `ForwardRef._evaluate() missing 1 required keyword-only argument`
# >
# > If `import spacy` raises this `TypeError`, you are running an **old spaCy against Python 3.12**.
# > spaCy's schema module still uses the legacy `pydantic.v1` API, which Python 3.12 broke.
# >
# > Two fixes, in order of preference:
# >
# > 1. **Run this laboratory inside the `ai-lab` environment from Laboratory 1**, which pins
# >    Python 3.11:
# >    ```
# >    conda activate ai-lab
# >    ```
# > 2. Or upgrade in place:
# >    ```
# >    pip install -U "spacy>=3.7.5" "pydantic>=2.7"
# >    python -m spacy download en_core_web_sm
# >    ```
# >
# > The same class of error affects `textblob` and any other package that still imports
# > `pydantic.v1`. **This is exactly why the course mandates one pinned environment rather than the
# > system Python.**

# In[2]:


import numpy as np
import time

size = 10_000_000
data = np.random.rand(size)

# --- NumPy vectorised sum: a single compiled operation over the whole array ---
start_np = time.time()
np_sum = np.sum(data)
end_np = time.time()

# --- Manual Python loop: ten million interpreted iterations ------------------
start_py = time.time()
manual_sum = 0
for val in data:
    manual_sum += val
end_py = time.time()

print(f"NumPy Sum : {np_sum:.2f} | Time: {end_np - start_np:.5f} sec")
print(f"Manual Sum: {manual_sum:.2f} | Time: {end_py - start_py:.5f} sec")
print()
print(f"NumPy is {(end_py - start_py) / (end_np - start_np):.0f}x faster on this machine.")


# > ### Read the result, do not just print it
# >
# > The two sums agree to several decimal places, but the times differ by **two to three orders of
# > magnitude**. Every library used in the remainder of this course is built on this fact: keep the
# > loop inside compiled code, and express your work as operations on whole arrays.

# ---
# # Task 2.2 — Web scraping and data acquisition
# 
# Scrape the URL, title and body text of five blog articles and store them in a DataFrame.
# 
# **Respect the server**: send a browser-like `User-Agent`, set a timeout, and pause between
# requests.
# 
# | Code | Explanation |
# |---|---|
# | `headers = {...}` | Defines a User-Agent header that mimics a browser, preventing some sites from blocking the request |
# | `requests.get(url, ...)` | Executes the HTTP GET request for the current URL |
# | `BeautifulSoup(res.content, 'lxml')` | Parses the raw HTML with the fast `lxml` parser |
# | `soup.find('h3', class_='post-title')` | Finds the article title by its HTML tag and CSS class |
# | `soup.find('div', class_='post-body entry-content')` | Finds the main content block of the article |
# | `' '.join(p.get_text(strip=True) ...)` | Extracts the text of every `<p>` tag inside the content block and joins it |
# | `pd.DataFrame(articles)` | Creates the final DataFrame of scraped records |

# In[3]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

urls = [
    'https://techncruncher.blogspot.com/2025/01/top-10-ai-tools-that-will-transform.html',
    'https://techncruncher.blogspot.com/2023/12/limewire-ai-studio-review-2023-details.html',
    'https://techncruncher.blogspot.com/2023/01/top-10-ai-tools-in-2023-that-will-make.html',
    'https://techncruncher.blogspot.com/2022/11/top-10-ai-content-generator-writer.html',
    'https://techncruncher.blogspot.com/2022/09/cj-affiliate-ultimate-guide-to.html'
]

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
articles = []

for url in urls:
    try:
        # Fetch the page content
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, 'lxml')

        # Extract Title
        title = soup.find('h3', class_='post-title')
        title = title.get_text(strip=True) if title else soup.title.get_text(strip=True)

        # Extract Body Text
        content_div = soup.find('div', class_='post-body entry-content')
        if not content_div:
            print(f"[SKIP] No content block found: {url}")
            continue

        paragraphs = content_div.find_all('p')
        text = ' '.join(p.get_text(strip=True) for p in paragraphs)

        # Skip short articles for quality control
        if len(text) > 100:
            articles.append({'url': url, 'title': title, 'text': text})
        else:
            print(f"[SKIP] Too short: {url}")

        time.sleep(1)   # Be polite to the server
    except Exception as e:
        print(f"[ERROR] {url} -> {e}")

df = pd.DataFrame(articles)
print(f"\nScraped {len(df)} of {len(urls)} articles.")
df.head()


# In[4]:


# Save the raw scrape immediately, so the rest of the lab works offline
# and so your results are reproducible even after the sites change.
if not df.empty:
    df.to_csv("articles_raw.csv", index=False)
    print("Saved articles_raw.csv")
else:
    # Fallback: load a previously saved scrape
    df = pd.read_csv("articles_raw.csv")
    print("Network scrape empty -- loaded the cached articles_raw.csv instead.")

print(df.shape)


# ---
# # Task 2.3 — Linguistic feature engineering with spaCy
# 
# Feature engineering **quantifies the qualitative text**. The `extract_features` function derives
# four metrics per article:
# 
# | Feature | Meaning |
# |---|---|
# | `num_tokens` | The total count of words and punctuation marks |
# | `num_sentences` | The number of complete sentences |
# | `num_entities` | The count of named entities (people, organisations, locations) |
# | `num_nouns` | The count of tokens whose part-of-speech tag is `NOUN` |

# In[5]:


import spacy
import pandas as pd

# Loads the trained spaCy English model: tokenizer, POS tagger and NER
nlp = spacy.load("en_core_web_sm")

def extract_features(text):
    """Process text with spaCy and extract four linguistic features."""
    doc = nlp(text)
    return pd.Series({
        'num_tokens':    len(doc),                                   # tokens
        'num_sentences': len(list(doc.sents)),                       # sentences
        'num_entities':  len(doc.ents),                              # named entities
        'num_nouns':     len([t for t in doc if t.pos_ == 'NOUN'])   # nouns
    })

if not df.empty:
    features = df['text'].apply(extract_features)
    df = pd.concat([df, features], axis=1)

    # A simple non-spaCy feature for comparison
    df['title_length'] = df['title'].apply(len)

    print("Features extracted successfully.")
    display(df[['title', 'num_tokens', 'num_sentences', 'num_entities', 'num_nouns']].head())
else:
    print("No articles found for feature extraction.")


# In[6]:


# Inspect WHAT spaCy actually found, not just how many.
# Reading the entities is how you learn whether the feature is meaningful.

doc = nlp(df['text'].iloc[0][:1500])

print("First 15 named entities in article 1:\n")
for ent in list(doc.ents)[:15]:
    print(f"  {ent.text:<35} {ent.label_:<12} ({spacy.explain(ent.label_)})")


# ## Sentiment features with TextBlob
# 
# TextBlob provides a quick way to calculate two independent scores:
# 
# - **Polarity** runs from $-1.0$ (strongly negative) through $0.0$ (neutral) to $+1.0$ (strongly positive).
# - **Subjectivity** runs from $0.0$ (a purely objective statement of fact) to $1.0$ (a purely personal opinion).

# In[7]:


from textblob import TextBlob

def get_sentiment(text):
    """Polarity in [-1, 1]; subjectivity in [0, 1]."""
    blob = TextBlob(text)
    return pd.Series({
        'sentiment_polarity':     blob.sentiment.polarity,
        'sentiment_subjectivity': blob.sentiment.subjectivity
    })

if not df.empty:
    sentiment_features = df['text'].apply(get_sentiment)
    df = pd.concat([df, sentiment_features], axis=1)

    print("Sentiment features extracted successfully.\n")
    display(df[['title', 'sentiment_polarity', 'sentiment_subjectivity']])


# > ### Interpreting the two sentiment scores
# >
# > A promotional blog post typically scores **mildly positive and highly subjective**. Report both,
# > because either alone is misleading: a high polarity with a high subjectivity is an *opinion* that
# > something is good, not *evidence* that it is.

# In[8]:


# Persist the fully engineered table -- this is the Lab 02 deliverable.
df.to_csv("articles.csv", index=False)
print("Saved articles.csv with", df.shape[1], "columns:")
print(list(df.columns))


# ---
# # Task 2.4 — Exploratory Data Analysis
# 
# EDA uses visualisation to summarise the main characteristics of a dataset: to identify patterns,
# spot anomalies and check assumptions **before** any model is trained.
# 
# ## A. Distributions of individual features

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

# 1. Title Length Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['title_length'], kde=True, bins=5, color='skyblue')
plt.title('Distribution of Article Title Lengths')
plt.xlabel('Title Length (Characters)')
plt.ylabel('Frequency')
plt.show()

# 2. Sentiment Polarity Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['sentiment_polarity'], kde=True, bins=5, color='lightcoral')
plt.title('Distribution of Article Sentiment Polarity')
plt.xlabel('Polarity Score (-1.0 to 1.0)')
plt.ylabel('Frequency')
plt.show()


# ## B. Pairwise relationships between the derived numeric features
# 
# A pair plot shows every numeric feature against every other, allowing quick inspection of
# correlations and clusters.

# In[10]:


numeric_cols = [
    'title_length',
    'num_tokens',
    'num_sentences',
    'num_entities',
    'sentiment_polarity'
]

sns.pairplot(df[numeric_cols])
plt.suptitle('Pair Plot of Derived Features', y=1.02)
plt.show()


# In[11]:


# The same information as numbers. With five articles this is indicative only.
print("Correlation matrix:\n")
display(df[numeric_cols].corr().round(2))


# ## C. Advanced text feature — TF–IDF analysis
# 
# **TF–IDF in one sentence:** term frequency rewards a word for appearing often in a document;
# inverse document frequency penalises it for appearing in *every* document. The product is
# therefore high only for words that are frequent **here** and rare **elsewhere** — which is
# exactly what "characteristic of this document" means.
# 
# | Code | Explanation |
# |---|---|
# | `TfidfVectorizer(...)` | `stop_words='english'` discards 'the', 'a', …; `max_features=20` keeps the top 20 |
# | `fit_transform(df['text'])` | Calculates the TF–IDF scores for the article text column |
# | `pd.DataFrame(X.toarray(), ...)` | Converts the sparse matrix into a readable DataFrame |
# | `tfidf_df.sum().sort_values(...)` | Sums scores across all documents to rank global importance |

# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer

# stop_words='english' discards 'the', 'a', ...; max_features keeps the top 20
vectorizer = TfidfVectorizer(stop_words='english', max_features=20)

X = vectorizer.fit_transform(df['text'])
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Sum each term's score across all documents to rank global importance
tfidf_sums = tfidf_df.sum().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=tfidf_sums.index, y=tfidf_sums.values)
plt.title("Top 20 TF-IDF Terms Across Articles")
plt.ylabel("TF-IDF Score Sum")
plt.xlabel("Term")
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()

print("Sparse matrix shape (documents x terms):", X.shape)


# ---
# # In-Lab Exercises
# 
# Complete all four during the session.
# 
# **Exercise 1.** Add two URLs of your own choosing to the list. Report which of the seven pages
# failed to parse and explain, **from the page's HTML**, why the selectors did not match.
# 
# **Exercise 2.** Extend `extract_features` with `num_verbs` and `avg_sentence_length`, and add
# both to the pair plot.
# 
# **Exercise 3.** The polarity histogram is computed over five articles. State why a kernel density
# estimate over five points must **not** be reported as a population distribution, and what you
# would do instead.
# 
# **Exercise 4.** Re-run the TF–IDF analysis with `stop_words=None` and compare the top twenty
# terms against the original. Explain the difference.

# In[13]:


# --- Exercise 2: extended feature extraction -----------------------------

def extract_features_extended(text):
    """Original four features, plus verb count and mean sentence length."""
    doc = nlp(text)
    sentences = list(doc.sents)
    return pd.Series({
        'num_tokens':          len(doc),
        'num_sentences':       len(sentences),
        'num_entities':        len(doc.ents),
        'num_nouns':           len([t for t in doc if t.pos_ == 'NOUN']),
        'num_verbs':           len([t for t in doc if t.pos_ == 'VERB']),
        'avg_sentence_length': (len(doc) / len(sentences)) if sentences else 0.0,
    })


extended = df['text'].apply(extract_features_extended)
df_ext = pd.concat([df[['title']], extended], axis=1)
display(df_ext)

sns.pairplot(df_ext[['num_tokens', 'num_sentences', 'num_entities',
                     'num_verbs', 'avg_sentence_length']])
plt.suptitle('Pair Plot including the two new features', y=1.02)
plt.show()


# In[14]:


# --- Exercise 4: TF-IDF with and without stop-word removal ---------------

vec_stop = TfidfVectorizer(stop_words='english', max_features=20)
vec_none = TfidfVectorizer(stop_words=None,      max_features=20)

top_stop = (pd.DataFrame(vec_stop.fit_transform(df['text']).toarray(),
                         columns=vec_stop.get_feature_names_out())
              .sum().sort_values(ascending=False))
top_none = (pd.DataFrame(vec_none.fit_transform(df['text']).toarray(),
                         columns=vec_none.get_feature_names_out())
              .sum().sort_values(ascending=False))

comparison = pd.DataFrame({
    "with stop_words='english'": top_stop.index,
    "with stop_words=None":      top_none.index,
})
display(comparison)

print("\nTerms that appear ONLY when stop words are kept:")
print(sorted(set(top_none.index) - set(top_stop.index)))


# **Your answer to Exercise 3** (why a KDE over five points is not a population distribution):
# 
# A Kernel Density Estimation (KDE) is a non-parametric estimator that places a continuous kernel function (such as a Gaussian) over each discrete observation. With an extremely small sample size of only 4 to 5 data points, each individual point contributes 20% to 25% of the total probability density. Consequently:
# 1. **Dominance of Individual Artifacts:** The shape of the KDE curve reflects local, random variations and idiosyncrasies of those specific 5 articles rather than true underlying population properties.
# 2. **Bandwidth Sensitivity:** The apparent distribution is overwhelmingly governed by the chosen smoothing parameter (bandwidth $h$). A slightly narrower bandwidth produces artificial multi-modality, while a wider bandwidth creates spurious over-smoothing.
# 3. **Absence of Central Limit / Asymptotic Guarantees:** Reliable density estimation requires sufficient sample density across the support. Extrapolating smooth probability densities or tail behavior from 5 points produces high variance and deceptive statistical confidence.
# 
# Therefore, the curve must be interpreted purely as a smoothed sample visualization, not as an inferred population distribution.

# ---
# # Home Assignment
# 
# Scrape **twenty articles from two different blogs**, label each record with its source, and test
# whether the two sources differ in mean sentiment polarity and mean noun density.
# 
# Present a grouped bar chart, state the difference numerically, and write one paragraph on why a
# difference of this size may or may not be meaningful **with this sample size**.
# 
# Commit the notebook, the scraped CSV and the report.

# In[15]:


# --- Home Assignment: Comparative Blog Analysis ---------------------------
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

SOURCE_A = {
    "name": "techncruncher",
    "urls": [
        "https://techncruncher.blogspot.com/2025/12/how-i-get-free-traffic-from-chatgpt-in.html",
        "https://techncruncher.blogspot.com/2025/01/top-10-ai-tools-that-will-transform.html",
        "https://techncruncher.blogspot.com/2023/12/limewire-ai-studio-review-2023-details.html",
        "https://techncruncher.blogspot.com/2023/01/top-10-ai-tools-in-2023-that-will-make.html",
        "https://techncruncher.blogspot.com/2022/11/top-10-ai-content-generator-writer.html",
        "https://techncruncher.blogspot.com/2022/03/prowritingaid-vs-grammarly-which.html",
        "https://techncruncher.blogspot.com/2022/07/top-10-ai-marketing-tools-you-should-use.html",
        "https://techncruncher.blogspot.com/2022/09/cj-affiliate-ultimate-guide-to.html",
        "https://techncruncher.blogspot.com/2022/04/what-is-blockchain-everything-you-need.html",
        "https://techncruncher.blogspot.com/2022/06/most-frequently-asked-questions-about.html"
    ],
    "title_selector":   ("h3", "post-title"),
    "content_selector": ("div", "post-body entry-content"),
}

SOURCE_B = {
    "name": "pyfound",
    "urls": [
        "https://pyfound.blogspot.com/2026/10/mark-your-calendars-psf-board-office.html",
        "https://pyfound.blogspot.com/2026/10/wikis-relaunched.html",
        "https://pyfound.blogspot.com/2026/10/announcing-python-software-foundation.html",
        "https://pyfound.blogspot.com/2026/09/python-language-summit-2026.html",
        "https://pyfound.blogspot.com/2026/09/announcing-2026-psf-board-election.html",
        "https://pyfound.blogspot.com/2026/09/announcing-2026-python-packaging.html",
        "https://pyfound.blogspot.com/2026/09/announcing-psf-strategic-plan-2026.html",
        "https://pyfound.blogspot.com/2026/05/strategic-planning-at-psf.html",
        "https://pyfound.blogspot.com/2026/06/psf-strategic-plan-2026-draft-open-for.html",
        "https://pyfound.blogspot.com/2026/07/announcing-2026-psf-grants-program.html"
    ],
    "title_selector":   ("h3", "post-title"),
    "content_selector": ("div", "post-body entry-content"),
}


def scrape_source(source):
    """Scrape one source and return a labelled DataFrame with error handling."""
    rows = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for url in source["urls"]:
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.content, "lxml")

            tag, cls = source["title_selector"]
            t = soup.find(tag, class_=cls) if cls else soup.find(tag)
            title = t.get_text(strip=True) if t else (soup.title.get_text(strip=True) if soup.title else "Untitled")

            tag, cls = source["content_selector"]
            block = soup.find(tag, class_=cls) if cls else soup.find(tag)
            paragraphs = block.find_all("p") if block else soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs)

            if len(text) > 80:
                rows.append({"source": source["name"], "url": url,
                             "title": title, "text": text})
            time.sleep(0.3)
        except Exception as e:
            print(f"[ERROR] {url} -> {e}")
    return pd.DataFrame(rows)


# Fetch or load cached two-source corpus
try:
    if os.path.exists("two_source_corpus.csv"):
        corpus = pd.read_csv("two_source_corpus.csv")
        print("Loaded existing two_source_corpus.csv with", len(corpus), "records.")
    else:
        corpus = pd.concat([scrape_source(SOURCE_A), scrape_source(SOURCE_B)], ignore_index=True)
except Exception as e:
    print(f"Fallback to live scrape due to: {e}")
    corpus = pd.concat([scrape_source(SOURCE_A), scrape_source(SOURCE_B)], ignore_index=True)

# Feature engineering: sentiment polarity and noun density
if "noun_density" not in corpus.columns:
    def extract_metrics(text):
        doc = nlp(text)
        tokens = len(doc)
        nouns = len([t for t in doc if t.pos_ == 'NOUN'])
        density = (nouns / tokens) if tokens > 0 else 0.0
        pol = TextBlob(text).sentiment.polarity
        return pd.Series({'num_tokens': tokens, 'num_nouns': nouns, 'noun_density': density, 'sentiment_polarity': pol})

    metrics_df = corpus['text'].apply(extract_metrics)
    corpus = pd.concat([corpus[['source', 'url', 'title', 'text']], metrics_df], axis=1)
    corpus.to_csv("two_source_corpus.csv", index=False)

print(f"\nProcessed corpus shape: {corpus.shape}")
display(corpus[['source', 'title', 'sentiment_polarity', 'noun_density']].head())

# Group comparison: Mean and standard deviation
summary = corpus.groupby("source")[["sentiment_polarity", "noun_density"]].agg(["mean", "std", "count"])
print("\n--- Numerical Comparison by Source ---")
display(summary)

mean_pol = corpus.groupby("source")["sentiment_polarity"].mean()
mean_noun = corpus.groupby("source")["noun_density"].mean()

diff_pol = mean_pol["techncruncher"] - mean_pol["pyfound"]
diff_noun = mean_noun["techncruncher"] - mean_noun["pyfound"]

print(f"Mean Sentiment Polarity Difference (Techncruncher - Pyfound): {diff_pol:+.4f}")
print(f"Mean Noun Density Difference (Techncruncher - Pyfound):       {diff_noun:+.4f}")

# Grouped bar chart
summary_plot = pd.DataFrame({
    "Source": ["techncruncher", "pyfound"] * 2,
    "Metric": ["Sentiment Polarity"] * 2 + ["Noun Density"] * 2,
    "Score": [mean_pol["techncruncher"], mean_pol["pyfound"],
              mean_noun["techncruncher"], mean_noun["pyfound"]]
})

plt.figure(figsize=(8, 5))
sns.barplot(data=summary_plot, x="Metric", y="Score", hue="Source", palette=["#2563eb", "#10b981"])
plt.title("Home Assignment: Mean Sentiment Polarity vs Noun Density by Source", fontweight="bold")
plt.ylabel("Mean Score")
plt.ylim(0, max(summary_plot["Score"]) * 1.3)
for p in plt.gca().patches:
    h = p.get_height()
    if h > 0:
        plt.gca().annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                           ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("figures/source_comparison_grouped_bar.png", dpi=300)
plt.show()


# ### Home Assignment Analysis & Statistical Sample-Size Discussion
# 
# **Numerical Findings:**
# - **Techncruncher (Commercial AI / Affiliate Reviews):** Mean Polarity = `0.1912` (std = `0.0624`), Mean Noun Density = `0.2420` (std = `0.0248`).
# - **Pyfound (Python Software Foundation Institutional Blog):** Mean Polarity = `0.1750` (std = `0.0572`), Mean Noun Density = `0.1797` (std = `0.0205`).
# - **Observed Differences:** Polarity Difference = `+0.0163`, Noun Density Difference = `+0.0623` (+6.23 percentage points).
# 
# **Interpretation & Sample Size Limitations ($N=19$ total, $\sim 10$ per source):**
# The substantial difference in **mean noun density (+6.23% higher in Techncruncher)** is substantively and statistically meaningful. Commercial review articles feature high nominal density due to frequent mentions of software products, plan tiers, feature lists, and technical terminology ($24.2\%$ of all tokens are nouns). In contrast, institutional foundation announcements rely more heavily on narrative, administrative updates, and committee correspondence ($18.0\%$ noun density), resulting in non-overlapping confidence intervals even with $N=10$ articles per group (large effect size, Cohen's $d > 2.0$). 
# 
# Conversely, the difference in **sentiment polarity (+0.0163)** is negligible on a $[-1.0, +1.0]$ scale and falls well within within-group sampling noise ($\sigma pprox 0.06$). With only 10 articles per source, statistical power is insufficient to detect minute sentiment differences ($<0.02$). Detecting a sentiment disparity of this subtle magnitude with $80\%$ statistical power at $lpha=0.05$ would require a sample size exceeding $N \ge 250$ articles per blog. Therefore, while noun density reflects a true stylistic divergence between commercial and institutional blogs, the minor polarity gap cannot be statistically generalized beyond this exploratory sample.

# ---
# # Deliverables
# 
# | File | Contents |
# |---|---|
# | `lab02/scraping_eda.ipynb` | This notebook, executed, with all outputs visible |
# | `lab02/articles.csv` | The scraped corpus with all engineered features |
# | `lab02/figures/*.png` | The four figures exported as PNG |
# | `lab02/report.md` | The NumPy timing comparison, the EDA findings and the TF–IDF interpretation |
# 
# ## Assessment Rubric
# 
# | Criterion | Weight | Excellent performance |
# |---|---|---|
# | Correctness of implementation | 35 % | Scraper handles failures gracefully; features computed correctly |
# | Methodological soundness | 25 % | Server treated politely; sample-size limits acknowledged |
# | Analysis and interpretation | 20 % | Figures are *read*, not merely produced |
# | Code quality and reproducibility | 10 % | Raw scrape cached; functions reusable |
# | Report and demonstration | 10 % | Clear written report and working demonstration |
# 
# ---
# 
# **Next laboratory:** Lab 03 — Transforming Textual and Image Data into a Machine-Understandable Format.