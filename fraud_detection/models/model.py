# fraud_detection/models/model.py

from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.model_selection import GridSearchCV

class FraudDetectionModel:
    """Main model class for fraud detection."""
    
    def __init__(self, model: BaseEstimator):
        """
        Initialize the fraud detection model.
        
        Args:
            model: A scikit-learn compatible model
        """
        self.model = model
        self.feature_columns: List[str] = []
    
    def train(self, X: pd.DataFrame, y: pd.Series, feature_columns: List[str] = None):
        """
        Train the model on the provided data.
        
        Args:
            X: Feature DataFrame
            y: Target series
            feature_columns: List of columns to use for training
        """
        if feature_columns is None:
            # Use all numeric columns by default
            feature_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        self.feature_columns = feature_columns
        self.model.fit(X[feature_columns], y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the input data."""
        return self.model.predict(X[self.feature_columns])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions for the input data."""
        return self.model.predict_proba(X[self.feature_columns])
    
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                           param_grid: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.
        
        Args:
            X: Feature DataFrame
            y: Target series
            param_grid: Dictionary of parameters to tune
            
        Returns:
            Dictionary of best parameters
        """
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X[self.feature_columns], y)
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
