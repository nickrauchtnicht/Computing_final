# fraud_detection/data/data_loader.py

import pandas as pd
import numpy as np
from typing import Tuple

class FraudDataLoader:
    """Loads and preprocesses fraud detection data."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the fraud detection dataset
        """
        self.data_path = data_path
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and perform initial preprocessing of the fraud detection dataset.
        
        Returns:
            DataFrame with processed data
        """
        df = pd.read_csv(self.data_path)
        
        # Process timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Process velocity_last_hour from string to dict
        df['velocity_last_hour'] = df['velocity_last_hour'].apply(eval)
        
        # Extract velocity features
        velocity_features = pd.json_normalize(df['velocity_last_hour'])
        velocity_features.columns = [f'velocity_{col}' for col in velocity_features.columns]
        
        # Drop original velocity column and join with extracted features
        df = df.drop('velocity_last_hour', axis=1)
        df = pd.concat([df, velocity_features], axis=1)
        
        return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   validation_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=42,
            stratify=df['is_fraud']
        )
        
        # Second split: separate validation set
        val_size_adjusted = validation_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=train_val_df['is_fraud']
        )
        
        return train_df, val_df, test_df