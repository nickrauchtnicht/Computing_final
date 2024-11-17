# fraud_detection/features/feature_engineering.py

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime

class FeatureGenerator:
    """Generates features for fraud detection."""
    
    def __init__(self):
        self.time_features = True
        self.amount_features = True
        self.interaction_features = True
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features for the fraud detection model.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df_features = df.copy()
        
        if self.time_features:
            df_features = self._generate_time_features(df_features)
        
        if self.amount_features:
            df_features = self._generate_amount_features(df_features)
        
        if self.interaction_features:
            df_features = self._generate_interaction_features(df_features)
        
        return df_features
    
    def _generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate time-based features."""
        # Hour of day cyclic encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour'] / 24)
        
        # Day of week
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Is weekend
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Is business hour (9-17)
        df['is_business_hour'] = ((df['transaction_hour'] >= 9) & 
                                (df['transaction_hour'] <= 17)).astype(int)
        
        return df
    
    def _generate_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate amount-based features."""
        # Amount transformations
        df['amount_log'] = np.log1p(df['amount'])
        
        # Amount ratios
        df['amount_to_mean_velocity'] = df['amount'] / (df['velocity_total_amount'] / 
                                                       df['velocity_num_transactions'])
        
        # High amount flag
        amount_threshold = df['amount'].quantile(0.95)
        df['is_high_amount'] = (df['amount'] > amount_threshold).astype(int)
        
        return df
    
    def _generate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features."""
        # Risk score combining multiple factors
        df['risk_score'] = (
            df['high_risk_merchant'].astype(int) * 2 +
            df['distance_from_home'] * 1.5 +
            df['is_high_amount'] * 2 +
            (df['velocity_num_transactions'] > df['velocity_num_transactions'].mean()).astype(int)
        )
        
        # Transaction density
        df['transaction_density'] = (df['velocity_num_transactions'] / 
                                   (df['velocity_unique_merchants'] + 1))
        
        return df