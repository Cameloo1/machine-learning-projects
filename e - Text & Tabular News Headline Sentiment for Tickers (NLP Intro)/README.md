# News Headline Sentiment

NLP project that classifies financial headlines as bullish, neutral, or bearish using TF-IDF features and ticker metadata.

## Models

- Logistic Regression
- Linear SVC

## Data

The source code supports two collection paths:

- NewsAPI.org with `NEWS_API_KEY`
- manual fallback scraping when `USE_MANUAL_SCRAPE=1`

Raw and labeled headline CSVs are not committed. Current review artifacts are committed under `reports/`.

## Run

```bash
pip install -r requirements.txt
set NEWS_API_KEY=your_key
python scripts/run_end_to_end.py
```

Or run the notebook:

```bash
jupyter notebook notebooks/01_news_headline_sentiment.ipynb
```

## Outputs

- `reports/model_comparison.md`
- `reports/model_comparison.csv`
- figures under `reports/figures/`
- token-importance CSVs under `reports/`

## Reproducibility Notes

The full pipeline depends on headline collection and manual labels. Treat the committed reports as a saved portfolio snapshot, not a guaranteed current benchmark.
