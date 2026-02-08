# Challenge

## Part I: Model Implementation

### Model Selection

4 models were evaluated, all trained with the top 10 features by importance:

| Model | Class Balance | Recall class "1" | F1 class "1" |
|-------|---------------|------------------|--------------|
| XGBoost | With balance | > 0.60 | > 0.30 |
| XGBoost | Without balance | Low | Low |
| Logistic Regression | With balance | > 0.60 | > 0.30 |
| Logistic Regression | Without balance | Low | Low |

**Chosen model: XGBoost with class balancing (`scale_pos_weight`)**

Rationale:
- Class balancing is required to achieve recall > 0.60 for class "1" (delays), since the dataset is imbalanced (~81% class 0, ~19% class 1).
- There is no noticeable performance difference between XGBoost and Logistic Regression. XGBoost was chosen because it provides native feature importance, which was used to select the top 10 features.
- Reducing from all features to the top 10 most important does not decrease model performance.

### Bugs Found and Fixed

1. **Type hint in `preprocess()`**: Fixed `Union(Tuple[...], ...)` to `Union[Tuple[...], ...]`. Python type hints use square brackets `[]`, not parentheses `()`.

### Implementation

- **`preprocess(data, target_column)`**: Generates dummy variables for `OPERA`, `TIPOVUELO` and `MES` using `pd.get_dummies`. Filters the top 10 features using `reindex` with `fill_value=0` to handle missing columns. If `target_column` is provided, computes `min_diff` and `delay`, returning `(features, target)`.

- **`fit(features, target)`**: Computes `scale_pos_weight = n_y0 / n_y1` for class balancing. Trains an `XGBClassifier` and persists the model to disk using `pickle`.

- **`predict(features)`**: Loads the model from disk if not already in memory. Returns predictions as `List[int]`.

### Tests

Run from the `tests/` folder:

```bash
cd tests && python -m pytest model/test_model.py -v
```

Result: 4/4 tests passed.
