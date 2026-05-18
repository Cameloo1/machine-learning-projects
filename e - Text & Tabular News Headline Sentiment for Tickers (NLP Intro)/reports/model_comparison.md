# News Sentiment Model Comparison

Saved portfolio snapshot comparing Logistic Regression and Linear SVC on the labeled headline dataset used for this run.

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.500 | 0.502 | 0.500 |
| Linear SVC | 0.574 | 0.577 | 0.575 |

## Takeaways

- Linear SVC performed best in this saved run.
- Accuracy is modest and the dataset is small, so this is a prototype result.
- Bullish vs bearish ambiguity remains the main weakness.

## Artifacts

- `model_comparison.csv`
- `top_tokens_logreg.csv`
- `top_tokens_linearsvc.csv`
- figures under `reports/figures/`

## Use

Treat this as an explainable baseline for financial-headline sentiment, not a production classifier. A stronger version needs more labeled data, time-aware validation, and clearer labeling guidelines.
