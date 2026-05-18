# SHAP Plot Explanation

SHAP plots explain how the trained model used input features for its predictions on the saved dataset.

## Files

- `artifacts/shap/global_summary.png`: global feature influence.
- `artifacts/shap/local_example_bar.png`: one local prediction example.
- `artifacts/explanations_sample.csv`: sampled text explanations.

## How To Read Them

- Larger absolute SHAP values mean a feature had more influence on the model output.
- Positive and negative directions are model-class specific.
- Global plots summarize many rows.
- Local plots explain one prediction at a time.

## Caution

SHAP explains the trained model. It does not prove that a feature causes alert priority in the real world. Use it to inspect model behavior, find suspicious shortcuts, and explain demo predictions.
