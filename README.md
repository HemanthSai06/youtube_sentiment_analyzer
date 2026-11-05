# 🎬 YouTube Comments Sentiment Analyzer using Machine Learning

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/seaborn-3776AB.svg?logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) and Supervised Machine Learning system designed to analyze and classify YouTube video comments into **Positive**, **Neutral**, or **Negative** sentiment categories.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Objectives](#-project-objectives)
- [Dataset Architecture](#-dataset-architecture)
  - [Dataset Overview](#dataset-overview)
  - [Sentiment Class Balance](#sentiment-class-balance)
  - [Available Metadata](#available-metadata)
- [Text Preprocessing & NLP Pipeline](#-text-preprocessing--nlp-pipeline)
  - [Regex Cleaning](#regex-cleaning)
  - [TF-IDF Feature Extraction](#tf-idf-feature-extraction)
- [Machine Learning Workflow](#-machine-learning-workflow)
  - [Algorithms Implemented](#algorithms-implemented)
  - [Model Training & Evaluation](#model-training--evaluation)
- [Visualizations & Diagnostics](#-visualizations--diagnostics)
- [Interactive Live Inference](#-interactive-live-inference)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Execution Guide](#-execution-guide)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 📖 Overview

User comments on YouTube offer rich, real-time insights into audience engagement, brand perception, and content reception. However, unstructured comment text is noisy—rife with URLs, HTML fragments, varied casing, and colloquial phrasing. 

This project implements a complete multiclass sentiment analysis pipeline using Scikit-Learn. It cleans raw comment text, extracts high-dimensional n-gram features using TF-IDF vectorization, benchmarks three foundational NLP classification algorithms, visualizes model performance diagnostics, and exposes an interactive terminal session for real-time comment inference.

---

## 🎯 Project Objectives

1. **Automated Sentiment Classification**: Categorize user comments into `Positive`, `Neutral`, and `Negative` sentiment tiers.
2. **Text Cleaning Pipeline**: Strip HTML artifacts, hyperlinks, irregular whitespace, and duplicate entries.
3. **TF-IDF Feature Representation**: Extract salient unigram and bigram features with sublinear term-frequency scaling.
4. **Algorithm Benchmarking**: Evaluate and contrast three distinct model families:
   - **Multinomial Naive Bayes** (Probabilistic)
   - **Logistic Regression** (Discriminative Linear Log-odds)
   - **Linear Support Vector Machine / LinearSVC** (Maximal-Margin Hyperplane)
5. **Multi-Faceted Diagnostics**: Assess models using Accuracy, Macro-averaged Precision, Recall, and F1-Score alongside confusion matrix heatmaps.
6. **Real-Time Comment Inference**: Interactive console loop enabling immediate prediction on user-supplied comments.

---

## 📊 Dataset Architecture

### Dataset Overview
The project is built upon a large-scale cleaned YouTube comments dataset (`youtube_comments_cleaned.csv`) containing over **1,032,000 annotated comments**.

### Sentiment Class Balance
The dataset features a balanced tri-class distribution across the three sentiment categories:

| Sentiment Class | Sample Count | Percentage |
| :--- | :--- | :--- |
| **Negative** | ~346,075 | 33.5% |
| **Positive** | ~343,317 | 33.3% |
| **Neutral** | ~342,833 | 33.2% |

### Available Metadata
While text classification focuses on `CommentText` and `Sentiment`, the dataset contains comprehensive engagement and publishing attributes:
- `CommentID`, `VideoID`, `VideoTitle`
- `AuthorName`, `AuthorChannelID`
- `CommentText`, `Sentiment`
- `Likes`, `Replies`, `PublishedAt`
- `CountryCode`, `CategoryID`

---

## 🧹 Text Preprocessing & NLP Pipeline

```
[Raw YouTube Comments]
          │
          ▼
[Case Folding & Normalization: lower()]
          │
          ▼
[HTML Tag Stripping: re.sub(r"<[^>]+>", " ")]
          │
          ▼
[URL Removal: re.sub(r"http\S+|www\.\S+", " ")]
          │
          ▼
[Whitespace Normalization & Empty String Pruning]
          │
          ▼
[Duplicate Comment-Label Pair De-duplication]
          │
          ▼
[Stratified 80/20 Train-Test Split]
          │
          ▼
[TF-IDF Feature Extraction]
  - max_features = 10,000
  - ngram_range = (1, 2) [Unigrams & Bigrams]
  - sublinear_tf = True [1 + log(tf)]
          │
          ▼
[Classification Models Training & Evaluation]
```

### Regex Cleaning
The text cleaning routine cleans noise while preserving emotional tokens:
- **HTML Cleanup**: Removes leftover markup from API extractions.
- **URL Stripping**: Eliminates spam links and referral URLs.
- **Whitespace Collapsing**: Trims multi-space sequences to single spaces.

### TF-IDF Feature Extraction
- **Unigram & Bigram Modeling** (`ngram_range=(1, 2)`): Captures individual words as well as key phrases (e.g., *"not good"*, *"highly recommend"*).
- **Sublinear Term Frequency** (`sublinear_tf=True`): Scales term frequency logarithmically ($1 + \log(\text{tf})$) to dampen the dominance of repetitive words.
- **Vocabulary Constraint** (`max_features=10,000`): Constrains feature dimensionality to prevent overfitting and optimize inference latency.

---

## ⚙️ Machine Learning Workflow

### Algorithms Implemented

1. **Multinomial Naive Bayes (`MultinomialNB`)**:
   - Classic generative probabilistic text classifier applying Bayes' Theorem under feature conditional independence.
   - Highly efficient training and inference on sparse text representations.

2. **Logistic Regression (`LogisticRegression`)**:
   - Robust discriminative linear model maximizing class conditional log-likelihood.
   - `max_iter=1000`, `random_state=42`.
   - Offers well-calibrated class boundaries for text sentiment tasks.

3. **Linear Support Vector Machine (`LinearSVC`)**:
   - Constructs optimal maximum-margin hyperplanes separating sentiment classes in 10,000-dimensional TF-IDF space.
   - `max_iter=5000`, `random_state=42`.
   - Superior performance on high-dimensional sparse representations.

### Model Training & Evaluation
Each model is bundled into a unified Scikit-Learn `Pipeline([('tfidf', vectorizer), ('classifier', model)])` and evaluated on the 20% holdout test set using:
- **Accuracy**: Overall fraction of correct predictions.
- **Macro Precision**: Average precision across classes without majority bias.
- **Macro Recall**: True positive rate averaged across all sentiment classes.
- **Macro F1-Score**: Harmonic mean of macro precision and recall (used to pick the champion model).
- **Confusion Matrices**: Detailed error distribution across classes.

---

## 📈 Visualizations & Diagnostics

The script generates four distinct analytical visualizations:
1. **Sentiment Count Distribution**: Seaborn categorical countplot of the dataset labels.
2. **Multi-Model Performance Comparison**: Side-by-side grouped bar chart comparing Accuracy, Precision, Recall, and F1-Score across Naive Bayes, Logistic Regression, and SVM.
3. **Confusion Matrix Comparison**: Side-by-side heatmap grid ($1 \times 3$) displaying predicted vs. actual classifications for all three models.
4. **Overall Sentiment Breakdown**: Matplotlib pie chart detailing percentage distribution of sentiments.

---

## 💬 Interactive Live Inference

After training and benchmark selection, the script automatically deploys the champion model into an interactive console loop:

```plaintext
=================================================================
YOUTUBE COMMENT SENTIMENT PREDICTION
=================================================================

Selected Model: Logistic Regression (or SVM)

Enter a comment to predict its sentiment.
Type 'exit' to stop prediction.

Enter YouTube Comment: This tutorial saved me hours of debugging, fantastic explanation!
Original Comment: This tutorial saved me hours of debugging, fantastic explanation!
Predicted Sentiment: positive
-----------------------------------------------------------------

Enter YouTube Comment: complete waste of time, video doesn't work at all
Original Comment: complete waste of time, video doesn't work at all
Predicted Sentiment: negative
-----------------------------------------------------------------
```

---

## 📁 Project Structure

```plaintext
youtube_sentiment_analyzer/
│
├── youtube_sentiment_analyzer.py      # Main pipeline script (cleaning, training, viz, CLI)
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git exclusion rules (ignores large CSV & cache)
└── README.md                          # Comprehensive project documentation
```

*Note: The dataset file (`youtube_comments_cleaned.csv`) is excluded from the Git repository due to its large size (~296 MB). When executing locally, ensure `youtube_comments_cleaned.csv` is placed in the project directory.*

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Git and [Git LFS](https://git-lfs.com/)

### 1. Clone the Repository
```bash
git clone https://github.com/HemanthSai06/youtube_sentiment_analyzer.git
cd youtube_sentiment_analyzer
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Execution Guide

Run the pipeline from your terminal:

```bash
python youtube_sentiment_analyzer.py
```

*Note: The script contains fallback file dialog resolution for execution within interactive IDEs such as Spyder, VS Code, or PyCharm.*

---

## 👤 Author

- **Hemanth Sai**  
  GitHub: [@HemanthSai06](https://github.com/HemanthSai06)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - feel free to use and modify for educational and portfolio purposes.
