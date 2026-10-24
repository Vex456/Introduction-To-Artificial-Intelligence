# Introduction to Artificial Intelligence (AI–101L) — Weekly Lab Portfolio

**Instructor:** Dr. Muhammad Awais  
**Student:** VEX  
**Course Code:** AI–101L  
**Session:** Fall 2026  

---

## 📁 Repository Overview

This repository contains the weekly laboratory work and deliverables for Introduction to Artificial Intelligence, structured cleanly with verified, working code.

| Laboratory | Date | Topic | Key Deliverables / Files |
|---|---|---|---|
| **Lab 02** | Oct 2026 | Web Scraping, NLP Feature Engineering & EDA | [`Lab_02_Web_Scraping_and_EDA.ipynb`](./Lab%2002%20-%20Web%20Scraping%20and%20EDA/Lab_02_Web_Scraping_and_EDA.ipynb), [`report.md`](./Lab%2002%20-%20Web%20Scraping%20and%20EDA/report.md), [`articles.csv`](./Lab%2002%20-%20Web%20Scraping%20and%20EDA/articles.csv), [`figures/`](./Lab%2002%20-%20Web%20Scraping%20and%20EDA/figures/) |
| **Lab 04** | Oct 2026 | Supervised Learning: Linear & Logistic Regression | [`Simple Linear Regression.ipynb`](./Lab4/Simple%20Linear%20Regression.ipynb), [`Multi Linear Regression.ipynb`](./Lab4/Multi%20Linear%20Regression.ipynb), [`diabetes-data-exploration-and-logistic-regressi.ipynb`](./Lab4/diabetes-data-exploration-and-logistic-regressi.ipynb) |
| **Lab 06** | Oct 2026 | Model Deployment: FastAPI Web App & UI | [`main.py`](./LAB06/main.py), [`index.html`](./LAB06/index.html), [`Mymdoel.pkl`](./LAB06/Mymdoel.pkl) |
| **Lab 08** | Oct 2026 | Deep Learning: Handwritten Digit Recognition (CNN) | [`Digits Recognition.ipynb`](./Lab08/Digits%20Recognition.ipynb), [`images.png`](./Lab08/images.png) |

---

## 🛠️ Laboratory Summaries

### Lab 02: Web Scraping, NLP Feature Engineering & EDA
- **Pipeline:** HTTP GET via `requests` + `BeautifulSoup` HTML parsing $\rightarrow$ tabular structuring $\rightarrow$ `spaCy` & `TextBlob` feature extraction $\rightarrow$ EDA & TF–IDF.
- **Deliverables:**
  - `articles.csv`: Full scraped corpus with engineered linguistic metrics.
  - `figures/*.png`: 4 high-resolution exported visualizations (distributions, feature pairplot, top-20 TF–IDF).
  - `report.md`: Complete analytical report evaluating vectorised speedup (328×), EDA findings, TF–IDF interpretation, and small-sample KDE limits.

### Lab 04: Supervised Machine Learning
- **Simple Linear Regression:** Experience-to-salary modeling ($y = \beta_0 + \beta_1 x$), train/test splitting, and evaluation.
- **Multiple Linear Regression:** Multi-variable house price prediction ($R^2$, MAE, MSE, RMSE).
- **Logistic Regression:** Binary diabetes classification on clinical metrics, confusion matrix visualization, and serialization to `Mymdoel.pkl`.

### Lab 06: Model Deployment with FastAPI
- Production REST API exposing `/predict` and `/health` endpoints.
- Real-time HTML/JS client interface for interactive patient diagnosis.

### Lab 08: Deep Learning with CNNs
- Sequential Convolutional Neural Network (Conv2D $\rightarrow$ ReLU $\rightarrow$ MaxPooling2D $\rightarrow$ Dropout $\rightarrow$ Softmax) implemented in TensorFlow/Keras.
- Trained on MNIST digits achieving **~98.9% validation accuracy**.
