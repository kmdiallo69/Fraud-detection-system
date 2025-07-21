#!/usr/bin/env python3
"""
Prediction Module
================

This module handles model loading and prediction for the fraud detection system.
It provides both batch and real-time prediction capabilities.

Author: ML Engineer
Date: 2024
"""

import pandas as pd
import numpy as np
import joblib
import pickle
import os
import json
from datetime import datetime
from typing import Dict, List, Union, Tuple
import warnings

warnings.filterwarnings("ignore")


class FraudPredictor:
    """
    A class for making fraud predictions using trained models.
    """

    def __init__(self, model_dir="data/raw", model_name=None):
        """
        Initialize the fraud predictor.

        Args:
            model_dir (str): Directory containing model files
            model_name (str): Specific model name to load (optional)
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_metadata = None

        # Load the most recent model if no specific model is specified
        if model_name is None:
            self._load_latest_model()
        else:
            self._load_specific_model(model_name)

    def _load_latest_model(self):
        """Load the most recent model from the model directory."""
        try:
            # Find all model files
            model_files = [
                f
                for f in os.listdir(self.model_dir)
                if f.startswith("fraud_detection_model_") and f.endswith(".pkl")
            ]

            if not model_files:
                raise FileNotFoundError(
                    "No model files found in the specified directory"
                )

            # Get the most recent model
            latest_model = sorted(model_files)[-1]
            self._load_model_components(latest_model)

        except Exception as e:
            print(f"❌ Error loading latest model: {e}")
            raise

    def _load_specific_model(self, model_name: str):
        """Load a specific model by name."""
        try:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self._load_model_components(f"{model_name}.pkl")

        except Exception as e:
            print(f"❌ Error loading specific model: {e}")
            raise

    def _load_model_components(self, model_filename: str):
        """Load all model components (model, scaler, features)."""
        print(f"📂 Loading model: {model_filename}")

        # Extract timestamp from model filename
        # Handle both formats: fraud_detection_model_xgboost_20250720_160929.pkl and fraud_detection_model.pkl
        if "xgboost" in model_filename:
            # Extract timestamp from: fraud_detection_model_xgboost_20250720_160929.pkl
            parts = model_filename.split("_")
            timestamp = f"{parts[-2]}_{parts[-1].replace('.pkl', '')}"
        else:
            # For simple model names, use empty timestamp
            timestamp = ""

        # Load model
        model_path = os.path.join(self.model_dir, model_filename)
        self.model = joblib.load(model_path)
        print(f"✅ Model loaded successfully")
        print(f"📊 Model type: {type(self.model)}")

        # Load scaler
        scaler_filename = f"scaler_{timestamp}.pkl"
        scaler_path = os.path.join(self.model_dir, scaler_filename)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded: {scaler_filename}")
        else:
            print(f"⚠️ Scaler not found: {scaler_filename}")

        # Load feature columns
        features_filename = f"feature_columns_{timestamp}.pkl"
        features_path = os.path.join(self.model_dir, features_filename)
        if os.path.exists(features_path):
            with open(features_path, "rb") as f:
                self.feature_columns = pickle.load(f)
            print(f"✅ Feature columns loaded: {features_filename}")
            print(f"📋 Number of features: {len(self.feature_columns)}")
        else:
            print(f"⚠️ Feature columns not found: {features_filename}")

        # Load model metadata
        metadata_filename = f"model_metadata_{timestamp}.json"
        metadata_path = os.path.join(self.model_dir, metadata_filename)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.model_metadata = json.load(f)
            print(f"✅ Model metadata loaded: {metadata_filename}")
        else:
            print(f"⚠️ Model metadata not found: {metadata_filename}")

    def preprocess_transaction(self, transaction_data: Dict) -> np.ndarray:
        """
        Preprocess a single transaction for prediction.

        Args:
            transaction_data (Dict): Transaction data with features

        Returns:
            np.ndarray: Preprocessed features
        """
        if self.feature_columns is None:
            raise ValueError("Feature columns not loaded. Cannot preprocess data.")

        # Create a DataFrame with the expected features
        df = pd.DataFrame([transaction_data])

        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Select only the required features in the correct order
        features = df[self.feature_columns]

        # Apply feature engineering (same as training)
        features = self._apply_feature_engineering(features)

        # Scale features if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)

        return features

    def _apply_feature_engineering(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same feature engineering as during training.

        Args:
            features (pd.DataFrame): Input features

        Returns:
            pd.DataFrame: Engineered features
        """
        # This should match the feature engineering from the training pipeline
        # For now, we'll assume the features are already engineered
        # In a real implementation, you would apply the same transformations

        return features

    def predict_single(self, transaction_data: Dict) -> Dict:
        """
        Make a prediction for a single transaction.

        Args:
            transaction_data (Dict): Transaction data

        Returns:
            Dict: Prediction results
        """
        try:
            # Preprocess the transaction
            features = self.preprocess_transaction(transaction_data)

            # Make prediction
            prediction = self.model.predict(features)[0]
            prediction_proba = self.model.predict_proba(features)[0]

            # Prepare result
            result = {
                "prediction": int(prediction),
                "prediction_label": "Fraud" if prediction == 1 else "Normal",
                "confidence": float(max(prediction_proba)),
                "probabilities": {
                    "normal": float(prediction_proba[0]),
                    "fraud": float(prediction_proba[1]),
                },
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "model_type": type(self.model).__name__,
                    "features_used": (
                        len(self.feature_columns) if self.feature_columns else None
                    ),
                },
            }

            return result

        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Make predictions for a batch of transactions.

        Args:
            transactions (List[Dict]): List of transaction data

        Returns:
            List[Dict]: List of prediction results
        """
        results = []

        for i, transaction in enumerate(transactions):
            try:
                result = self.predict_single(transaction)
                result["transaction_id"] = i
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "transaction_id": i,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return results

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions from a pandas DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with transaction data

        Returns:
            pd.DataFrame: DataFrame with predictions added
        """
        if self.feature_columns is None:
            raise ValueError("Feature columns not loaded. Cannot preprocess data.")

        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Select only the required features
        features = df[self.feature_columns]

        # Apply feature engineering
        features = self._apply_feature_engineering(features)

        # Scale features if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Make predictions
        predictions = self.model.predict(features)
        prediction_probas = self.model.predict_proba(features)

        # Add predictions to the original DataFrame
        result_df = df.copy()
        result_df["prediction"] = predictions
        result_df["prediction_label"] = [
            "Fraud" if p == 1 else "Normal" for p in predictions
        ]
        result_df["confidence"] = np.max(prediction_probas, axis=1)
        result_df["fraud_probability"] = prediction_probas[:, 1]
        result_df["normal_probability"] = prediction_probas[:, 0]

        return result_df

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dict: Model information
        """
        info = {
            "model_type": type(self.model).__name__ if self.model else None,
            "feature_count": (
                len(self.feature_columns) if self.feature_columns else None
            ),
            "has_scaler": self.scaler is not None,
            "model_metadata": self.model_metadata,
        }

        if self.model and hasattr(self.model, "get_params"):
            info["model_parameters"] = self.model.get_params()

        return info

    def save_predictions(self, predictions: List[Dict], filename: str = None):
        """
        Save predictions to a JSON file.

        Args:
            predictions (List[Dict]): List of predictions
            filename (str): Output filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}.json"

        output_path = os.path.join(self.model_dir, filename)

        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2, default=str)

        print(f"✅ Predictions saved to: {output_path}")


def create_sample_transaction() -> Dict:
    """
    Create a sample transaction for testing.

    Returns:
        Dict: Sample transaction data
    """
    # This is a sample transaction with the expected features
    # In a real application, you would get this from your data source
    sample = {
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Amount": 149.62,
        "hour": 0,
        "day": 0,
        "amount_log": 5.008,
        "amount_sqrt": 12.23,
        "v_mean": 0.0,
        "v_std": 1.0,
        "v_max": 2.54,
        "v_min": -1.99,
        "v_range": 4.53,
        "amount_time_interaction": 0.0,
        "Amount_capped": 149.62,
    }

    return sample


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize predictor
        predictor = FraudPredictor()

        # Get model info
        model_info = predictor.get_model_info()
        print("\n📊 Model Information:")
        print(json.dumps(model_info, indent=2))

        # Create sample transaction
        sample_transaction = create_sample_transaction()

        # Make prediction
        prediction = predictor.predict_single(sample_transaction)
        print("\n🎯 Sample Prediction:")
        print(json.dumps(prediction, indent=2))

    except Exception as e:
        print(f"❌ Error: {e}")
