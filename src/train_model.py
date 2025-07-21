#!/usr/bin/env python3
"""
Fraud Detection System - Enhanced Version
=========================================

This script implements a comprehensive fraud detection system using machine learning
techniques to identify fraudulent credit card transactions.

Key Features:
- Advanced EDA with statistical analysis
- Feature engineering and selection
- Multiple model comparison
- Hyperparameter tuning
- Model interpretability with SHAP
- Performance optimization
- Comprehensive evaluation metrics

Author: ML Engineer
Date: 2024
"""

# =============================================================================
# 1. 📥 IMPORT REQUIRED LIBRARIES
# =============================================================================

# Core data manipulation and analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score, RocCurveDisplay,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor

# Advanced ML libraries
import xgboost as xgb
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Model interpretability
import shap

# Model persistence
import joblib
import pickle

# Performance monitoring
import time
import psutil
import gc

# Visualization settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# =============================================================================
# 2. 📂 DATA LOADING AND INITIAL EXPLORATION
# =============================================================================

def load_dataset():
    """
    Load the credit card fraud dataset from Kaggle.
    
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
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def initial_data_exploration(df):
    """
    Perform initial data exploration and quality checks.
    
    Args:
        df (pandas.DataFrame): Input dataset
    """
    print("\n" + "="*60)
    print("🔍 INITIAL DATA EXPLORATION")
    print("="*60)
    
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
    
    return {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': missing_values,
        'duplicates': duplicates,
        'target_distribution': target_dist
    }

# =============================================================================
# 3. 🔍 EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def perform_eda(df):
    """
    Comprehensive Exploratory Data Analysis.
    
    Args:
        df (pandas.DataFrame): Input dataset
    """
    print("\n" + "="*60)
    print("📊 EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
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
    # Select top correlated features with target
    correlations = df.corr()['Class'].abs().sort_values(ascending=False)
    top_features = correlations.head(10).index.tolist()
    correlation_matrix = df[top_features].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=ax5, fmt='.2f', square=True)
    ax5.set_title('Top Features Correlation Matrix', fontweight='bold')
    
    # 6. Feature Statistics
    ax6 = axes[1, 2]
    # Calculate statistics for V1-V28 features
    feature_stats = df.iloc[:, 1:29].describe()
    ax6.axis('off')
    ax6.text(0.1, 0.9, 'Feature Statistics Summary:', fontsize=12, fontweight='bold')
    ax6.text(0.1, 0.8, f'Mean range: {feature_stats.loc["mean"].min():.3f} to {feature_stats.loc["mean"].max():.3f}', fontsize=10)
    ax6.text(0.1, 0.7, f'Std range: {feature_stats.loc["std"].min():.3f} to {feature_stats.loc["std"].max():.3f}', fontsize=10)
    ax6.text(0.1, 0.6, f'Min range: {feature_stats.loc["min"].min():.3f} to {feature_stats.loc["min"].max():.3f}', fontsize=10)
    ax6.text(0.1, 0.5, f'Max range: {feature_stats.loc["max"].min():.3f} to {feature_stats.loc["max"].max():.3f}', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Additional statistical analysis
    print("\n📈 Statistical Summary:")
    print(df.describe())
    
    # Feature importance analysis
    print("\n🔍 Top 10 Features by Correlation with Target:")
    correlations = df.corr()['Class'].abs().sort_values(ascending=False)
    print(correlations.head(10))
    
    return correlations

# =============================================================================
# 4. 🧼 DATA PREPROCESSING AND FEATURE ENGINEERING
# =============================================================================

def preprocess_data(df):
    """
    Comprehensive data preprocessing and feature engineering.
    
    Args:
        df (pandas.DataFrame): Input dataset
        
    Returns:
        tuple: Processed features and target
    """
    print("\n" + "="*60)
    print("🧼 DATA PREPROCESSING & FEATURE ENGINEERING")
    print("="*60)
    
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # 1. Feature Engineering
    print("🔧 Creating engineered features...")
    
    # Time-based features
    df_processed['hour'] = (df_processed['Time'] // 3600) % 24
    df_processed['day'] = (df_processed['Time'] // (3600 * 24)) % 7
    
    # Amount-based features
    df_processed['amount_log'] = np.log1p(df_processed['Amount'])
    df_processed['amount_sqrt'] = np.sqrt(df_processed['Amount'])
    
    # Statistical features for V1-V28
    v_columns = [f'V{i}' for i in range(1, 29)]
    df_processed['v_mean'] = df_processed[v_columns].mean(axis=1)
    df_processed['v_std'] = df_processed[v_columns].std(axis=1)
    df_processed['v_max'] = df_processed[v_columns].max(axis=1)
    df_processed['v_min'] = df_processed[v_columns].min(axis=1)
    df_processed['v_range'] = df_processed['v_max'] - df_processed['v_min']
    
    # Interaction features
    df_processed['amount_time_interaction'] = df_processed['Amount'] * df_processed['Time']
    
    print(f"✅ Created {len(df_processed.columns) - len(df.columns)} new features")
    
    # 2. Outlier Detection and Treatment
    print("\n🔍 Detecting outliers...")
    
    # Use IQR method for Amount
    Q1 = df_processed['Amount'].quantile(0.25)
    Q3 = df_processed['Amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_count = ((df_processed['Amount'] < lower_bound) | 
                     (df_processed['Amount'] > upper_bound)).sum()
    print(f"📊 Found {outliers_count} outliers in Amount ({outliers_count/len(df_processed)*100:.2f}%)")
    
    # Cap outliers instead of removing them
    df_processed['Amount_capped'] = df_processed['Amount'].clip(lower=lower_bound, upper=upper_bound)
    
    # 3. Feature Selection
    print("\n🎯 Performing feature selection...")
    
    # Remove original Time and Amount columns, keep engineered versions
    columns_to_drop = ['Time', 'Amount']
    df_processed = df_processed.drop(columns=columns_to_drop)
    
    # Select features for modeling
    feature_columns = [col for col in df_processed.columns if col != 'Class']
    X = df_processed[feature_columns]
    y = df_processed['Class']
    
    print(f"📋 Final feature set: {X.shape[1]} features")
    print(f"📊 Final dataset shape: {X.shape}")
    
    return X, y, df_processed

# =============================================================================
# 5. ⚖️ HANDLING CLASS IMBALANCE
# =============================================================================

def handle_class_imbalance(X_train, y_train, method='smote'):
    """
    Handle class imbalance using various techniques.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        method (str): Resampling method ('smote', 'adasyn', 'smoteenn', 'undersample')
        
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
    elif method == 'adasyn':
        sampler = ADASYN(random_state=42, n_neighbors=5)
    elif method == 'smoteenn':
        sampler = SMOTEENN(random_state=42)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    print(f"📊 After resampling:")
    print(f"   Normal transactions: {sum(y_resampled == 0)}")
    print(f"   Fraud transactions: {sum(y_resampled == 1)}")
    print(f"   Balance ratio: {sum(y_resampled == 0) / sum(y_resampled == 1):.2f}:1")
    
    return X_resampled, y_resampled

# =============================================================================
# 6. 🤖 MODEL TRAINING AND EVALUATION
# =============================================================================

def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models and compare their performance.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        dict: Trained models and their performance metrics
    """
    print("\n" + "="*60)
    print("🤖 MODEL TRAINING AND EVALUATION")
    print("="*60)
    
    models = {}
    results = {}
    
    # Define models to train
    model_configs = {
        'XGBoost': {
            'model': XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [8, 10, 12],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
    }
    
    # Train and evaluate each model
    for name, config in model_configs.items():
        print(f"\n🔧 Training {name}...")
        start_time = time.time()
        
        # Hyperparameter tuning
        print(f"   🔍 Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        models[name] = best_model
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'best_params': grid_search.best_params_,
            'training_time': time.time() - start_time
        }
        
        results[name] = metrics
        
        print(f"   ✅ {name} trained successfully!")
        print(f"   📊 Best parameters: {grid_search.best_params_}")
        print(f"   ⏱️ Training time: {metrics['training_time']:.2f} seconds")
        print(f"   🎯 F1 Score: {metrics['f1']:.4f}")
        print(f"   📈 AUC: {metrics['auc']:.4f}")
    
    return models, results

def plot_model_comparison(results):
    """
    Plot comparison of model performances.
    
    Args:
        results (dict): Model results dictionary
    """
    print("\n📊 Plotting model comparison...")
    
    # Prepare data for plotting
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    models = list(results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i//3, i%3]
        values = [results[model][metric] for model in models]
        
        bars = ax.bar(models, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_title(f'{metric.upper()} Score', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Training time comparison
    ax = axes[1, 2]
    times = [results[model]['training_time'] for model in models]
    bars = ax.bar(models, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_title('Training Time (seconds)', fontweight='bold')
    ax.set_ylabel('Time (s)')
    
    # Add value labels on bars
    for bar, value in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{value:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("\n📋 Model Performance Summary:")
    print("-" * 80)
    print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 80)
    for model, metrics in results.items():
        print(f"{model:<15} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['auc']:<10.4f}")

# =============================================================================
# 7. 📊 DETAILED MODEL EVALUATION
# =============================================================================

def detailed_evaluation(best_model, X_test, y_test, model_name):
    """
    Perform detailed evaluation of the best model.
    
    Args:
        best_model: Trained model
        X_test, y_test: Test data
        model_name (str): Name of the model
    """
    print(f"\n" + "="*60)
    print(f"📊 DETAILED EVALUATION - {model_name.upper()}")
    print("="*60)
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # 1. Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'])
    plt.title('Confusion Matrix', fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # 3. ROC Curve
    plt.subplot(1, 2, 2)
    RocCurveDisplay.from_predictions(y_test, y_pred_proba)
    plt.title('ROC Curve', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 4. Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ap_score = average_precision_score(y_test, y_pred_proba)
    
    plt.plot(recall, precision, linewidth=2, label=f'AP = {ap_score:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 5. Feature Importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        print("\n🎯 Feature Importance Analysis:")
        
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title('Top 15 Feature Importances', fontweight='bold')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        print("Top 10 most important features:")
        print(feature_importance.head(10))

# =============================================================================
# 8. 🔍 MODEL INTERPRETABILITY (SHAP)
# =============================================================================

def model_interpretability(model, X_test, y_test, model_name):
    """
    Perform model interpretability analysis using SHAP.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        model_name (str): Name of the model
    """
    print(f"\n" + "="*60)
    print(f"🔍 MODEL INTERPRETABILITY - {model_name.upper()}")
    print("="*60)
    
    try:
        # Create SHAP explainer
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_test)
        
        # Calculate SHAP values
        print("🔄 Calculating SHAP values...")
        shap_values = explainer.shap_values(X_test)
        
        # For tree-based models, shap_values is a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class values
        
        # 1. Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {model_name}', fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 2. Detailed Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f'SHAP Summary Plot - {model_name}', fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 3. Individual prediction explanation
        print("\n🔍 Individual Prediction Analysis:")
        
        # Find a fraud case
        fraud_indices = np.where(y_test == 1)[0]
        if len(fraud_indices) > 0:
            sample_idx = fraud_indices[0]
            print(f"Analyzing fraud case at index {sample_idx}")
            
            plt.figure(figsize=(12, 8))
            shap.force_plot(
                explainer.expected_value if not isinstance(explainer.expected_value, list) 
                else explainer.expected_value[1],
                shap_values[sample_idx],
                X_test.iloc[sample_idx],
                show=False
            )
            plt.title(f'SHAP Force Plot - Fraud Case - {model_name}', fontweight='bold')
            plt.tight_layout()
            plt.show()
        
    except Exception as e:
        print(f"⚠️ SHAP analysis failed: {e}")
        print("This might be due to model type or computational constraints.")

# =============================================================================
# 9. 💾 MODEL PERSISTENCE AND DEPLOYMENT
# =============================================================================

def save_model(model, scaler, feature_columns, model_name, results):
    """
    Save the trained model and related artifacts for deployment.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_columns: List of feature column names
        model_name (str): Name of the model
        results (dict): Model performance results
    """
    print(f"\n" + "="*60)
    print("💾 MODEL PERSISTENCE AND DEPLOYMENT")
    print("="*60)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_filename = f'fraud_detection_model_{model_name.lower()}_{timestamp}.pkl'
    joblib.dump(model, model_filename)
    print(f"✅ Model saved as: {model_filename}")
    
    # Save scaler
    scaler_filename = f'scaler_{timestamp}.pkl'
    joblib.dump(scaler, scaler_filename)
    print(f"✅ Scaler saved as: {scaler_filename}")
    
    # Save feature columns
    features_filename = f'feature_columns_{timestamp}.pkl'
    with open(features_filename, 'wb') as f:
        pickle.dump(feature_columns, f)
    print(f"✅ Feature columns saved as: {features_filename}")
    
    # Save model metadata
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'performance_metrics': results[model_name],
        'feature_count': len(feature_columns),
        'model_type': type(model).__name__
    }
    
    metadata_filename = f'model_metadata_{timestamp}.json'
    import json
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✅ Model metadata saved as: {metadata_filename}")
    
    # Create deployment summary
    print(f"\n📋 Deployment Summary:")
    print(f"   Model: {model_name}")
    print(f"   Performance: F1={results[model_name]['f1']:.4f}, AUC={results[model_name]['auc']:.4f}")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Files created: {model_filename}, {scaler_filename}, {features_filename}, {metadata_filename}")

# =============================================================================
# 10. 🚀 MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function orchestrating the entire fraud detection pipeline.
    """
    print("🚀 FRAUD DETECTION SYSTEM - ENHANCED VERSION")
    print("="*60)
    print("Starting comprehensive fraud detection analysis...")
    
    # Track overall execution time
    start_time = time.time()
    
    try:
        # 1. Load dataset
        df = load_dataset()
        if df is None:
            return
        
        # 2. Initial exploration
        exploration_results = initial_data_exploration(df)
        
        # 3. EDA
        correlations = perform_eda(df)
        
        # 4. Preprocessing
        X, y, df_processed = preprocess_data(df)
        
        # 5. Train-test split
        print("\n" + "="*60)
        print("✂️ TRAIN-TEST SPLIT")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {X_train.shape}")
        print(f"📊 Test set: {X_test.shape}")
        
        # 6. Handle class imbalance
        X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train, method='smote')
        
        # 7. Scale features
        print("\n🔧 Scaling features...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame for better handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_resampled.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # 8. Train models
        models, results = train_models(X_train_scaled, y_train_resampled, X_test_scaled, y_test)
        
        # 9. Plot model comparison
        plot_model_comparison(results)
        
        # 10. Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
        best_model = models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   F1 Score: {results[best_model_name]['f1']:.4f}")
        print(f"   AUC: {results[best_model_name]['auc']:.4f}")
        
        # 11. Detailed evaluation
        detailed_evaluation(best_model, X_test_scaled, y_test, best_model_name)
        
        # 12. Model interpretability
        model_interpretability(best_model, X_test_scaled, y_test, best_model_name)
        
        # 13. Save model
        save_model(best_model, scaler, X_train_scaled.columns.tolist(), best_model_name, results)
        
        # 14. Performance summary
        total_time = time.time() - start_time
        print(f"\n" + "="*60)
        print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print(f"🏆 Best model: {best_model_name}")
        print(f"📊 Best F1 Score: {results[best_model_name]['f1']:.4f}")
        print(f"📈 Best AUC: {results[best_model_name]['auc']:.4f}")
        
        # Memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        print(f"💾 Peak memory usage: {memory_usage:.2f} MB")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 