# Fraud Detection Library

## Overview
A scalable machine learning library for detecting fraudulent transactions in financial data. This project implements an end-to-end pipeline including data preprocessing, feature engineering, model training, and API deployment.

## Project Structure
```
fraud_detection/
├── data/                # Data loading and preprocessing
│   ├── __init__.py
│   └── data_loader.py
├── preprocessing/       # Data preprocessing modules
│   ├── __init__.py
│   └── preprocessors.py
├── features/           # Feature engineering
│   ├── __init__.py
│   └── feature_engineering.py
├── models/             # Model implementation
│   ├── __init__.py
│   ├── model.py
│   └── cross_validation.py
├── evaluation/         # Evaluation metrics
│   ├── __init__.py
│   └── metrics.py
└── api/               # API implementation
    ├── __init__.py
    └── app.py
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## How to Use

### Data Loading
```python
from fraud_detection.data.data_loader import FraudDataLoader

# Load data
loader = FraudDataLoader("path/to/data.csv")
df = loader.load_data()

# Split data
train_df, val_df, test_df = loader.split_data(df)
```

### Feature Engineering
```python
from fraud_detection.features.feature_engineering import FeatureGenerator

# Generate features
generator = FeatureGenerator()
features_df = generator.generate_features(df)
```

### Model Training
```python
from fraud_detection.models.model import FraudDetectionModel
from sklearn.ensemble import RandomForestClassifier

# Initialize and train model
model = FraudDetectionModel(RandomForestClassifier())
model.train(X_train, y_train)
```

## Extending the Library

### Adding New Preprocessors
1. Create a new class in `preprocessing/preprocessors.py`
2. Inherit from `BasePreprocessor`
3. Implement `fit` and `transform` methods
4. Add unit tests

Example:
```python
class NewPreprocessor(BasePreprocessor):
    def fit(self, df: pd.DataFrame) -> 'NewPreprocessor':
        # Add fitting logic
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add transformation logic
        return df_transformed
```

### Adding New Features
1. Add new feature generation methods to `features/feature_engineering.py`
2. Add corresponding unit tests

Example:
```python
def _generate_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    # Add feature generation logic
    return df_new
```

### Adding New Models
1. Create new model class in `models/`
2. Implement training and prediction methods
3. Add evaluation metrics

## API Usage

Start the API:
```bash
uvicorn fraud_detection.api.app:app --reload
```

Make a prediction:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "amount": 1000.0,
         "merchant_category": "retail",
         "transaction_hour": 14,
         "distance_from_home": 0,
         "high_risk_merchant": false,
         "weekend_transaction": false,
         "velocity_num_transactions": 5,
         "velocity_total_amount": 2000.0,
         "velocity_unique_merchants": 3
     }'
```

## Testing

Run the tests:
```bash
python -m pytest tests/
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## License
MIT License