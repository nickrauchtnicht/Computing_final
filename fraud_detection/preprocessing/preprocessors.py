# fraud_detection/preprocessing/preprocessors.py

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict

class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors."""
    
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'BasePreprocessor':
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

class TransactionPreprocessor(BasePreprocessor):
    """Preprocesses transaction-specific features."""
    
    def __init__(self):
        self.amount_scaler = StandardScaler()
        self.velocity_scalers = {}
        self.categorical_mappings = {}
    
    def fit(self, df: pd.DataFrame) -> 'TransactionPreprocessor':
        # Fit amount scaler
        self.amount_scaler.fit(df[['amount']])
        
        # Fit velocity scalers
        velocity_cols = [col for col in df.columns if col.startswith('velocity_')]
        for col in velocity_cols:
            self.velocity_scalers[col] = StandardScaler().fit(df[[col]])
        
        # Create categorical mappings
        categorical_columns = ['merchant_category', 'merchant_type', 'card_type', 'device', 'channel']
        for col in categorical_columns:
            if col in df.columns:
                self.categorical_mappings[col] = {
                    val: idx for idx, val in enumerate(df[col].unique())
                }
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_transformed = df.copy()
        
        # Transform amount
        df_transformed['amount_scaled'] = self.amount_scaler.transform(df[['amount']])
        
        # Transform velocity features
        for col, scaler in self.velocity_scalers.items():
            df_transformed[f'{col}_scaled'] = scaler.transform(df[[col]])
        
        # Transform categorical features
        for col, mapping in self.categorical_mappings.items():
            df_transformed[f'{col}_encoded'] = df[col].map(mapping)
        
        return df_transformed