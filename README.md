# From-Game-Patch-Notes-to-Predicted-Player-Sentiment
A multi-staged NLP pipeline that predicts the direction of community sentiment shifts after game updates, using topic modeling (LDA), aspect-based sentiment analysis, and ML classifiers on 100K+ Reddit posts and 100+ game patch notes. This project is built as a Master's Thesis project.

## Overview

Live-service games release frequent patches that can heavily impact community sentiment. This project builds a supervised pipeline to predict whether sentiment around specific discussion topics will improve or decline after a patch, using only the patch's characteristics and pre-patch discussion data.

The pipeline was applied to *The Finals* (Embark Studios), analyzing 100,000+ Reddit posts from r/TheFinals collected via Arctic Shift.

## Methodology

1. **Topic Modeling** — LDA (Gensim) to segment community discourse into 4 recurring themes: Combat Mechanics & Balance, Technical Performance, Cosmetics & Monetization, and Progression & Matchmaking.
2. **Aspect-Based Sentiment Analysis** — PyABSA's model (LCF architecture) scores sentiment per aspect as P(positive) − P(negative), using vocabulary-based aspect tagging and spaCy lemmatization.
3. **Feature Engineering** — 118 patches manually annotated and feature-engineered (with hotfixes merged into parent patches), producing 11 features representing the changes that happened to the game during the respective update.
4. **Classification** — Five models (Logistic Regression, Random Forest, SVM, XGBoost, CatBoost) predict the direction of sentiment shift (pre-patch vs. post-patch) per topic, using pre-patch sentiment score + 11 patch notes features as input.
5. **Evaluation** — Accuracy, macro-precision/recall/F1 from confusion matrices, with Wilson 90% confidence intervals for robustness.

## Results

- Best-performing model: **XGBoost**, reaching up to **79.2% accuracy** on the Combat Mechanics & Balance topic.
- Performance varied by discussion topic, reflecting varying patch impact and predictability per theme.

## Tech Stack

`Python` · `Gensim` (LDA) · `PyABSA` · `scikit-learn` · `pandas` . `Jupyter Notebook`

## Project Structure

```
├── data/           # Reddit posts and patch notes results
├── notebooks/      # Jupyter notebook with the full pipeline
└── README.md
```

## Author

Dhia — Master's in Business Analytics, Tunis Business School
