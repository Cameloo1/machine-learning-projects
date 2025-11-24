# Model Comparison

| model_name | accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- |
| Logistic Regression | 0.500 | 0.502 | 0.500 |
| Linear SVC | 0.574 | 0.577 | 0.575 |

## Highlights
- Macro F1 leader: Linear SVC (0.577).
- Weighted F1 leader: Linear SVC (0.575).
- Largest confusion (LogReg): ('Neutral', 'Bullish', 7).
- Largest confusion (Linear SVC): ('Bullish', 'Bearish', 7).
- Both models remain interpretable thanks to linear weights over TF-IDF + ticker features.
