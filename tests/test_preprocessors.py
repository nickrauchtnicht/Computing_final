# tests/test_preprocessors.py

import unittest
import pandas as pd
import numpy as np
from fraud_detection.preprocessing.preprocessors import TransactionPreprocessor

class TestPreprocessors(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'amount': [100, 200, 300],
            'merchant_category': ['retail', 'food', 'retail'],
            'velocity_num_transactions': [5, 10, 15],
            'timestamp': pd.date_range(start='2024-01-01', periods=3)
        })
    
    def test_transaction_preprocessor(self):
        """Test transaction preprocessor functionality."""
        preprocessor = TransactionPreprocessor()
        transformed_df = preprocessor.fit_transform(self.sample_data)
        
        self.assertIn('amount_scaled', transformed_df.columns)
        self.assertEqual(len(transformed_df), len(self.sample_data))

# tests/test_features.py

import unittest
import pandas as pd
import numpy as np
from fraud_detection.features.feature_engineering import FeatureGenerator

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'amount': [100, 200, 300],
            'transaction_hour': [9, 14, 20],
            'timestamp': pd.date_range(start='2024-01-01', periods=3),
            'velocity_num_transactions': [5, 10, 15],
            'velocity_total_amount': [1000, 2000, 3000],
            'high_risk_merchant': [False, True, False],
            'distance_from_home': [0, 1, 2]
        })
    
    def test_feature_generation(self):
        """Test feature generation functionality."""
        generator = FeatureGenerator()
        features_df = generator.generate_features(self.sample_data)
        
        self.assertIn('hour_sin', features_df.columns)
        self.assertIn('amount_log', features_df.columns)
        self.assertIn('risk_score', features_df.columns)
        self.assertEqual(len(features_df), len(self.sample_data))
