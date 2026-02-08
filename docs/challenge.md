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

## Part II: API Implementation

### Endpoint

- **POST `/predict`**: Receives a list of flights and returns delay predictions.

Request body:
```json
{
  "flights": [
    {"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 3}
  ]
}
```

Response:
```json
{"predict": [0]}
```

### Input Validation

Pydantic models (`FlightData`) validate each flight before processing:
- **OPERA**: Must be a valid airline from the dataset (loaded dynamically from `data.csv`).
- **TIPOVUELO**: Must be "I" (international) or "N" (national).
- **MES**: Must be between 1 and 12.

Invalid inputs return HTTP 400. FastAPI's default 422 validation error is overridden with a custom exception handler to return 400 as expected by the tests.

### Flow

1. Validate input with Pydantic
2. Convert flights to DataFrame
3. Preprocess with `DelayModel.preprocess()`
4. Predict with `DelayModel.predict()`
5. Return predictions

### Tests

Run from the `tests/` folder:

```bash
cd tests && python -m pytest api/test_api.py -v
```

Result: 4/4 tests passed.

## Part III: Cloud Deployment

### Cloud Provider

The API was deployed on **Google Cloud Run** in the `us-central1` region.

- **URL:** `https://flight-delay-prediction-739046741257.us-central1.run.app`
- **Endpoint:** `POST /predict`

### Cloud Run Configuration

| Parameter | Value |
|---|---|
| CPU | 1 vCPU |
| Memory | 2 GiB |
| Min instances (`minScale`) | 1 |
| Max instances (`maxScale`) | 5 |
| Container concurrency | 80 |
| Request timeout | 300s |
| Startup CPU boost | Enabled |

Setting `minScale: 1` ensures one instance is always warm, eliminating cold start latency. `maxScale: 5` allows the service to scale up under load while keeping costs controlled.

### Tests

The API was validated at multiple levels before and after deployment:

- **Unit/integration tests** (`make api-test`): 4/4 passed — validates correct predictions, input validation (invalid OPERA, TIPOVUELO, MES), and HTTP status codes (200/400).
- **Model tests** (`make model-test`): 4/4 passed — validates preprocessing, training, and prediction pipeline.
- **Stress test** (`make stress-test`): Passed — validates performance under load against the deployed service.

### Stress Test Results

The stress test was executed with `make stress-test` (Locust, 100 concurrent users, 60 seconds):

| Metric | Result |
|---|---|
| Total requests | 11,492 |
| Failed requests | 0 (0.00%) |
| Average response time | 157 ms |
| Median response time | 130 ms |
| P95 response time | 270 ms |
| P99 response time | 320 ms |
| Max response time | 1,144 ms |
| Throughput | ~192 req/s |

All requests returned successfully with 0% failure rate. The test passed with exit code 0.

