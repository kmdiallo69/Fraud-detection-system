#!/usr/bin/env python3
"""
Data Preprocessing Module
========================

This module handles all data preprocessing tasks for the fraud detection system:
- Data loading from Kaggle
- Data exploration and quality checks
- Feature engineering
- Data cleaning and outlier handling
- Train-test splitting
- Class imbalance handling

Author: ML Engineer
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import joblib
import pickle
import os

warnings.filterwarnings('ignore')

# Visualization settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class DataPreprocessor:
    """
    A comprehensive data preprocessing class for fraud detection.
    """
    
    def __init__(self, data_dir='../data'):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir (str): Directory to store/load data files
        """
        self.data_dir = data_dir
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_columns = None
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/raw", exist_ok=True)
        os.makedirs(f"{data_dir}/processed", exist_ok=True)
    
    def load_dataset(self, force_download=False):
        """
        Load the credit card fraud dataset from Kaggle.
        
        Args:
            force_download (bool): Force re-download of dataset
            
        Returns:
            pandas.DataFrame: The loaded dataset
        """
        print("🔄 Loading dataset...")
        start_time = time.time()
        
        try:
            import kagglehub
            
            # Download latest version of the dataset
            path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
            print(f"📁 Dataset path: {path}")
            
            # Load the dataset
            df = pd.read_csv(f"{path}/creditcard.csv")
            
            load_time = time.time() - start_time
            print(f"✅ Dataset loaded successfully in {load_time:.2f} seconds")
            print(f"📊 Dataset shape: {df.shape}")
            
            self.raw_data = df
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def explore_data(self, save_plots=True):
        """
        Perform comprehensive data exploration.
        
        Args:
            save_plots (bool): Whether to save EDA plots
            
        Returns:
            dict: Exploration results
        """
        if self.raw_data is None:
            print("❌ No data loaded. Call load_dataset() first.")
            return None
        
        print("\n" + "="*60)
        print("🔍 DATA EXPLORATION")
        print("="*60)
        
        df = self.raw_data
        
        # Basic information
        print(f"📊 Dataset Shape: {df.shape}")
        print(f"📋 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        print("\n📋 Data Types:")
        print(df.dtypes.value_counts())
        
        # Missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            print("\n✅ No missing values found!")
        else:
            print("\n⚠️ Missing values detected:")
            print(missing_values[missing_values > 0])
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        print(f"\n🔄 Duplicate rows: {duplicates}")
        
        # Target variable distribution
        print("\n🎯 Target Variable Distribution:")
        target_dist = df['Class'].value_counts()
        print(target_dist)
        print(f"Fraud percentage: {target_dist[1]/len(df)*100:.3f}%")
        
        # Create EDA plots
        if save_plots:
            self._create_eda_plots()
        
        # Statistical summary
        print("\n📈 Statistical Summary:")
        print(df.describe())
        
        # Feature correlations
        correlations = df.corr()['Class'].abs().sort_values(ascending=False)
        print("\n🔍 Top 10 Features by Correlation with Target:")
        print(correlations.head(10))
        
        return {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': missing_values,
            'duplicates': duplicates,
            'target_distribution': target_dist,
            'correlations': correlations
        }
    
    def _create_eda_plots(self):
        """Create and save EDA plots."""
        df = self.raw_data
        
        # Create figure for all plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive EDA - Credit Card Fraud Detection', fontsize=16, fontweight='bold')
        
        # 1. Target Distribution
        ax1 = axes[0, 0]
        sns.countplot(data=df, x='Class', ax=ax1, palette=['#2E8B57', '#DC143C'])
        ax1.set_title('Target Variable Distribution', fontweight='bold')
        ax1.set_xlabel('Class (0: Normal, 1: Fraud)')
        ax1.set_ylabel('Count')
        
        # Add percentage labels
        total = len(df)
        for i, v in enumerate(df['Class'].value_counts().sort_index()):
            ax1.text(i, v + total*0.01, f'{v/total*100:.2f}%', ha='center', fontweight='bold')
        
        # 2. Transaction Amount Distribution
        ax2 = axes[0, 1]
        sns.histplot(data=df, x='Amount', bins=50, ax=ax2, color='skyblue', alpha=0.7)
        ax2.set_title('Transaction Amount Distribution', fontweight='bold')
        ax2.set_xlabel('Amount ($)')
        ax2.set_ylabel('Frequency')
        
        # 3. Time Distribution
        ax3 = axes[0, 2]
        sns.histplot(data=df, x='Time', bins=50, ax=ax3, color='lightgreen', alpha=0.7)
        ax3.set_title('Transaction Time Distribution', fontweight='bold')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Frequency')
        
        # 4. Amount by Class
        ax4 = axes[1, 0]
        sns.boxplot(data=df, x='Class', y='Amount', ax=ax4, palette=['#2E8B57', '#DC143C'])
        ax4.set_title('Transaction Amount by Class', fontweight='bold')
        ax4.set_xlabel('Class (0: Normal, 1: Fraud)')
        ax4.set_ylabel('Amount ($)')
        
        # 5. Correlation Heatmap (top features)
        ax5 = axes[1, 1]
        correlations = df.corr()['Class'].abs().sort_values(ascending=False)
        top_features = correlations.head(10).index.tolist()
        correlation_matrix = df[top_features].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    ax=ax5, fmt='.2f', square=True)
        ax5.set_title('Top Features Correlation Matrix', fontweight='bold')
        
        # 6. Feature Statistics
        ax6 = axes[1, 2]
        feature_stats = df.iloc[:, 1:29].describe()
        ax6.axis('off')
        ax6.text(0.1, 0.9, 'Feature Statistics Summary:', fontsize=12, fontweight='bold')
        ax6.text(0.1, 0.8, f'Mean range: {feature_stats.loc["mean"].min():.3f} to {feature_stats.loc["mean"].max():.3f}', fontsize=10)
        ax6.text(0.1, 0.7, f'Std range: {feature_stats.loc["std"].min():.3f} to {feature_stats.loc["std"].max():.3f}', fontsize=10)
        ax6.text(0.1, 0.6, f'Min range: {feature_stats.loc["min"].min():.3f} to {feature_stats.loc["min"].max():.3f}', fontsize=10)
        ax6.text(0.1, 0.5, f'Max range: {feature_stats.loc["max"].min():.3f} to {feature_stats.loc["max"].max():.3f}', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{self.data_dir}/processed/eda_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 EDA plots saved to: {plot_path}")
        plt.show()
    
    def engineer_features(self):
        """
        Perform feature engineering on the dataset.
        
        Returns:
            tuple: Processed features and target
        """
        if self.raw_data is None:
            print("❌ No data loaded. Call load_dataset() first.")
            return None, None
        
        print("\n" + "="*60)
        print("🧼 FEATURE ENGINEERING")
        print("="*60)
        
        df = self.raw_data.copy()
        
        # 1. Feature Engineering
        print("🔧 Creating engineered features...")
        
        # Time-based features
        df['hour'] = (df['Time'] // 3600) % 24
        df['day'] = (df['Time'] // (3600 * 24)) % 7
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['Amount'])
        df['amount_sqrt'] = np.sqrt(df['Amount'])
        
        # Statistical features for V1-V28
        v_columns = [f'V{i}' for i in range(1, 29)]
        df['v_mean'] = df[v_columns].mean(axis=1)
        df['v_std'] = df[v_columns].std(axis=1)
        df['v_max'] = df[v_columns].max(axis=1)
        df['v_min'] = df[v_columns].min(axis=1)
        df['v_range'] = df['v_max'] - df['v_min']
        
        # Interaction features
        df['amount_time_interaction'] = df['Amount'] * df['Time']
        
        print(f"✅ Created {len(df.columns) - len(self.raw_data.columns)} new features")
        
        # 2. Outlier Detection and Treatment
        print("\n🔍 Detecting outliers...")
        
        # Use IQR method for Amount
        Q1 = df['Amount'].quantile(0.25)
        Q3 = df['Amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = ((df['Amount'] < lower_bound) | 
                         (df['Amount'] > upper_bound)).sum()
        print(f"📊 Found {outliers_count} outliers in Amount ({outliers_count/len(df)*100:.2f}%)")
        
        # Cap outliers instead of removing them
        df['Amount_capped'] = df['Amount'].clip(lower=lower_bound, upper=upper_bound)
        
        # 3. Feature Selection
        print("\n🎯 Performing feature selection...")
        
        # Remove original Time and Amount columns, keep engineered versions
        columns_to_drop = ['Time', 'Amount']
        df = df.drop(columns=columns_to_drop)
        
        # Select features for modeling
        self.feature_columns = [col for col in df.columns if col != 'Class']
        X = df[self.feature_columns]
        y = df['Class']
        
        print(f"📋 Final feature set: {X.shape[1]} features")
        print(f"📊 Final dataset shape: {X.shape}")
        
        self.processed_data = df
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            tuple: Training and testing sets
        """
        print("\n" + "="*60)
        print("✂️ TRAIN-TEST SPLIT")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"📊 Training set: {X_train.shape}")
        print(f"📊 Test set: {X_test.shape}")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        """
        Handle class imbalance using various techniques.
        
        Args:
            X_train: Training features
            y_train: Training target
            method: Resampling method
            
        Returns:
            tuple: Resampled features and target
        """
        print(f"\n⚖️ Handling class imbalance using {method.upper()}...")
        
        print(f"📊 Before resampling:")
        print(f"   Normal transactions: {sum(y_train == 0)}")
        print(f"   Fraud transactions: {sum(y_train == 1)}")
        print(f"   Imbalance ratio: {sum(y_train == 0) / sum(y_train == 1):.2f}:1")
        
        if method == 'smote':
            sampler = SMOTE(random_state=42, k_neighbors=5)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        print(f"📊 After resampling:")
        print(f"   Normal transactions: {sum(y_resampled == 0)}")
        print(f"   Fraud transactions: {sum(y_resampled == 1)}")
        print(f"   Balance ratio: {sum(y_resampled == 0) / sum(y_resampled == 1):.2f}:1")
        
        return X_resampled, y_resampled
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using RobustScaler.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            tuple: Scaled training and test features
        """
        print("\n🔧 Scaling features...")
        
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for better handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        return X_train_scaled, X_test_scaled
    
    def save_preprocessing_artifacts(self):
        """Save preprocessing artifacts for later use."""
        if self.scaler is None or self.feature_columns is None:
            print("❌ No preprocessing artifacts to save. Run preprocessing first.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save scaler
        scaler_path = f"{self.data_dir}/processed/scaler_{timestamp}.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler saved: {scaler_path}")
        
        # Save feature columns
        features_path = f"{self.data_dir}/processed/feature_columns_{timestamp}.pkl"
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print(f"✅ Feature columns saved: {features_path}")
        
        # Save processed data
        if self.processed_data is not None:
            data_path = f"{self.data_dir}/processed/processed_data_{timestamp}.csv"
            self.processed_data.to_csv(data_path, index=False)
            print(f"✅ Processed data saved: {data_path}")
    
    def run_full_preprocessing(self, save_artifacts=True):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            save_artifacts (bool): Whether to save preprocessing artifacts
            
        Returns:
            tuple: Preprocessed training and testing data
        """
        print("🚀 Starting full preprocessing pipeline...")
        
        # 1. Load data
        self.load_dataset()
        
        # 2. Explore data
        self.explore_data()
        
        # 3. Engineer features
        X, y = self.engineer_features()
        
        # 4. Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # 5. Handle class imbalance
        X_train_resampled, y_train_resampled = self.handle_class_imbalance(X_train, y_train)
        
        # 6. Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train_resampled, X_test)
        
        # 7. Save artifacts
        if save_artifacts:
            self.save_preprocessing_artifacts()
        
        print("\n✅ Preprocessing pipeline completed successfully!")
        
        return X_train_scaled, X_test_scaled, y_train_resampled, y_test

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.run_full_preprocessing() 