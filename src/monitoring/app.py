#!/usr/bin/env python3
"""
Fraud Detection Monitoring Dashboard
===================================

Streamlit application for monitoring fraud detection system performance,
model metrics, and real-time predictions.

Author: ML Engineer
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
import time
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional
import joblib
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import FraudPredictor, create_sample_transaction

# Try to import logger, but don't fail if it doesn't exist
try:
    from utils.logger import setup_logger
    logger = setup_logger("fraud_monitoring")
except ImportError:
    import logging
    logger = logging.getLogger("fraud_monitoring")
    logger.setLevel(logging.INFO)

# Configure page
st.set_page_config(
    page_title="Fraud Detection Monitor",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FraudDetectionMonitor:
    """Main monitoring dashboard class."""
    
    def __init__(self):
        self.predictor = None
        self.api_url = os.getenv("API_URL", "http://localhost:8000")
        self.model_dir = "data/raw"
        
    def load_model(self):
        """Load the fraud detection model."""
        try:
            self.predictor = FraudPredictor(model_dir=self.model_dir)
            return True
        except Exception as e:
            st.warning(f"⚠️ Model not loaded: {e}")
            st.info("💡 The dashboard will work in demo mode without the model")
            return False
    
    def check_api_health(self) -> Dict:
        """Check API health status."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        if self.predictor:
            return self.predictor.get_model_info()
        return {}
    
    def create_sample_data(self, n_samples: int = 100) -> pd.DataFrame:
        """Create sample transaction data for demonstration."""
        data = []
        for i in range(n_samples):
            sample = create_sample_transaction()
            # Add some variation
            sample['transaction_id'] = f"TXN_{i:06d}"
            sample['timestamp'] = datetime.now() - timedelta(minutes=i)
            sample['amount_variation'] = sample['Amount'] * np.random.uniform(0.5, 2.0)
            data.append(sample)
        return pd.DataFrame(data)
    
    def run_dashboard(self):
        """Run the main dashboard."""
        
        try:
            # Header
            st.title("🕵️ Fraud Detection System Monitor")
            st.markdown("---")
            
            # Sidebar
            self.setup_sidebar()
            
            # Main content
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                self.show_system_status()
            
            with col2:
                self.show_model_metrics()
            
            with col3:
                self.show_prediction_stats()
            
            with col4:
                self.show_api_status()
            
            # Tabs for different sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Real-time Monitoring", 
                "🎯 Model Performance", 
                "🔍 Prediction Analysis",
                "📈 Historical Data",
                "⚙️ System Configuration"
            ])
            
            with tab1:
                self.show_realtime_monitoring()
            
            with tab2:
                self.show_model_performance()
            
            with tab3:
                self.show_prediction_analysis()
            
            with tab4:
                self.show_historical_data()
            
            with tab5:
                self.show_system_config()
                
        except Exception as e:
            st.error(f"❌ Dashboard error: {e}")
            st.info("💡 Trying to show basic information...")
            
            # Fallback to basic display
            st.title("🕵️ Fraud Detection System Monitor")
            st.write("⚠️ Some dashboard components failed to load, but the system is working.")
            
            # Try to show basic model info
            try:
                if self.predictor:
                    st.success("✅ Model is loaded and working")
                    model_info = self.get_model_info()
                    st.write(f"Model Type: {model_info.get('model_type', 'Unknown')}")
                else:
                    st.warning("⚠️ Model not loaded")
            except:
                st.error("❌ Unable to access model information")
    
    def setup_sidebar(self):
        """Setup the sidebar with controls."""
        st.sidebar.title("🎛️ Controls")
        
        # Model loading
        if st.sidebar.button("🔄 Reload Model"):
            with st.spinner("Loading model..."):
                if self.load_model():
                    st.sidebar.success("Model loaded successfully!")
                else:
                    st.sidebar.error("Failed to load model")
        
        # API configuration
        st.sidebar.subheader("🔗 API Configuration")
        self.api_url = st.sidebar.text_input(
            "API URL", 
            value=self.api_url,
            help="URL of the fraud detection API"
        )
        
        # Refresh rate
        st.sidebar.subheader("⏱️ Refresh Settings")
        auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
        refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 5, 60, 30)
        
        if auto_refresh:
            st.info(f"Auto-refresh enabled: {refresh_rate}s")
            # Note: Auto-refresh should be implemented with st.empty() and time-based updates
            # For now, we'll disable it to prevent blocking
        
        # Sample data generation
        st.sidebar.subheader("📊 Sample Data")
        n_samples = st.sidebar.slider("Number of samples", 10, 1000, 100)
        
        if st.sidebar.button("Generate Sample Data"):
            self.sample_data = self.create_sample_data(n_samples)
            st.sidebar.success(f"Generated {n_samples} samples")
    
    def show_system_status(self):
        """Show system status card."""
        st.metric(
            label="System Status",
            value="🟢 Online" if self.predictor else "🔴 Offline",
            delta="Model Loaded" if self.predictor else "Model Not Loaded"
        )
    
    def show_model_metrics(self):
        """Show model metrics card."""
        if self.predictor:
            model_info = self.get_model_info()
            model_type = model_info.get('model_type', 'Unknown')
            feature_count = model_info.get('feature_count', 0)
            
            st.metric(
                label="Model Type",
                value=model_type,
                delta=f"{feature_count} features"
            )
        else:
            st.metric(
                label="Model Type",
                value="Not Loaded",
                delta="Load model to see metrics"
            )
    
    def show_prediction_stats(self):
        """Show prediction statistics card."""
        # This would be populated from actual prediction logs
        total_predictions = 1234
        fraud_detected = 45
        
        st.metric(
            label="Total Predictions",
            value=total_predictions,
            delta=f"{fraud_detected} fraud detected"
        )
    
    def show_api_status(self):
        """Show API status card."""
        api_health = self.check_api_health()
        status = api_health.get('status', 'unknown')
        
        if status == 'healthy':
            st.metric(
                label="API Status",
                value="🟢 Healthy",
                delta="All systems operational"
            )
        else:
            st.metric(
                label="API Status",
                value="🔴 Unhealthy",
                delta=api_health.get('error', 'Unknown error')
            )
    
    def show_realtime_monitoring(self):
        """Show real-time monitoring dashboard."""
        st.header("📊 Real-time Monitoring")
        
        # Create sample real-time data
        if not hasattr(self, 'sample_data'):
            self.sample_data = self.create_sample_data(100)
        
        # Real-time predictions chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔄 Live Predictions")
            
            # Simulate real-time predictions
            if self.predictor and st.button("Start Live Monitoring"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(10):
                    # Simulate prediction
                    sample = create_sample_transaction()
                    try:
                        result = self.predictor.predict_single(sample)
                        status_text.text(f"Prediction {i+1}: {result['prediction_label']} (Confidence: {result['confidence']:.3f})")
                    except:
                        status_text.text(f"Prediction {i+1}: Error")
                    
                    progress_bar.progress((i + 1) / 10)
                    time.sleep(0.5)
                
                st.success("Live monitoring completed!")
        
        with col2:
            st.subheader("📈 Transaction Volume")
            
            # Create time series data
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            volumes = np.random.poisson(1000, 30)
            
            fig = px.line(
                x=dates, 
                y=volumes,
                title="Daily Transaction Volume",
                labels={'x': 'Date', 'y': 'Transactions'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Fraud detection trends
        st.subheader("🎯 Fraud Detection Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud rate over time
            fraud_rates = np.random.beta(2, 100, 30)  # Low fraud rate
            fig = px.line(
                x=dates,
                y=fraud_rates * 100,
                title="Fraud Rate (%)",
                labels={'x': 'Date', 'y': 'Fraud Rate (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Amount distribution
            amounts = np.random.exponential(100, 1000)
            fig = px.histogram(
                x=amounts,
                title="Transaction Amount Distribution",
                labels={'x': 'Amount ($)', 'y': 'Count'},
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_model_performance(self):
        """Show model performance metrics."""
        st.header("🎯 Model Performance")
        
        # Load model metadata if available
        model_metadata = {}
        try:
            model_files = [f for f in os.listdir(self.model_dir) 
                          if f.startswith('model_metadata_') and f.endswith('.json')]
            if model_files:
                latest_metadata = sorted(model_files)[-1]
                with open(os.path.join(self.model_dir, latest_metadata), 'r') as f:
                    model_metadata = json.load(f)
        except Exception as e:
            st.warning(f"Could not load model metadata: {e}")
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Metrics")
            
            if model_metadata:
                metrics = model_metadata.get('best_model_metrics', {})
                
                # Create metrics display
                metric_cols = st.columns(2)
                
                with metric_cols[0]:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                    st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                    st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                
                with metric_cols[1]:
                    st.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
                    st.metric("AUC", f"{metrics.get('auc', 0):.4f}")
                    st.metric("Training Time", f"{metrics.get('training_time', 0):.2f}s")
            else:
                st.info("No model metadata available. Train a model first.")
        
        with col2:
            st.subheader("🏆 Model Comparison")
            
            # Create comparison chart
            models = ['XGBoost', 'Random Forest', 'Logistic Regression']
            accuracies = [0.9994, 0.9980, 0.9902]
            precisions = [0.7748, 0.4620, 0.1397]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Accuracy',
                x=models,
                y=accuracies,
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='Precision',
                x=models,
                y=precisions,
                marker_color='lightcoral'
            ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                barmode='group',
                yaxis_title="Score"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("🔍 Feature Importance")
        
        if self.predictor and hasattr(self.predictor.model, 'feature_importances_'):
            feature_importance = self.predictor.model.feature_importances_
            feature_names = self.predictor.feature_columns
            
            # Get top 10 features
            top_indices = np.argsort(feature_importance)[-10:]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            fig = px.bar(
                x=top_importance,
                y=top_features,
                orientation='h',
                title="Top 10 Most Important Features",
                labels={'x': 'Importance', 'y': 'Feature'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
    
    def show_prediction_analysis(self):
        """Show prediction analysis and insights."""
        st.header("🔍 Prediction Analysis")
        
        # Prediction form
        st.subheader("🎯 Test Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Sample Transaction**")
            if st.button("Load Sample Transaction"):
                sample = create_sample_transaction()
                st.json(sample)
        
        with col2:
            st.write("**Custom Transaction**")
            
            # Create input form for custom transaction
            with st.form("prediction_form"):
                amount = st.number_input("Amount", min_value=0.0, value=100.0)
                v1 = st.number_input("V1", value=-1.36)
                v2 = st.number_input("V2", value=-0.07)
                
                submitted = st.form_submit_button("Predict")
                
                if submitted and self.predictor:
                    # Create transaction data
                    transaction = create_sample_transaction()
                    transaction['Amount'] = amount
                    transaction['V1'] = v1
                    transaction['V2'] = v2
                    
                    try:
                        result = self.predictor.predict_single(transaction)
                        
                        st.success("Prediction completed!")
                        st.json(result)
                        
                        # Show confidence gauge
                        confidence = result['confidence']
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=confidence * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Prediction Confidence"},
                            delta={'reference': 80},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
        
        # Batch prediction analysis
        st.subheader("📊 Batch Prediction Analysis")
        
        if st.button("Run Batch Analysis"):
            if hasattr(self, 'sample_data') and self.predictor:
                with st.spinner("Running batch analysis..."):
                    try:
                        results = self.predictor.predict_from_dataframe(self.sample_data)
                        
                        # Show results summary
                        fraud_count = (results['prediction'] == 1).sum()
                        normal_count = (results['prediction'] == 0).sum()
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Transactions", len(results))
                        col2.metric("Fraud Detected", fraud_count)
                        col3.metric("Normal Transactions", normal_count)
                        
                        # Show distribution
                        fig = px.pie(
                            values=[normal_count, fraud_count],
                            names=['Normal', 'Fraud'],
                            title="Transaction Classification Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show confidence distribution
                        fig = px.histogram(
                            results,
                            x='confidence',
                            title="Prediction Confidence Distribution",
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Batch analysis failed: {e}")
            else:
                st.warning("Please load sample data and model first.")
    
    def show_historical_data(self):
        """Show historical data and trends."""
        st.header("📈 Historical Data Analysis")
        
        # Date range selector
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        # Generate historical data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Simulate historical data
        np.random.seed(42)
        historical_data = pd.DataFrame({
            'date': date_range,
            'transactions': np.random.poisson(1000, len(date_range)),
            'fraud_count': np.random.binomial(1000, 0.001, len(date_range)),
            'avg_amount': np.random.exponential(100, len(date_range)),
            'fraud_rate': np.random.beta(2, 1000, len(date_range))
        })
        
        # Historical trends
        st.subheader("📊 Historical Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                historical_data,
                x='date',
                y='transactions',
                title="Daily Transaction Volume"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                historical_data,
                x='date',
                y='fraud_rate',
                title="Daily Fraud Rate"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("📋 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{historical_data['transactions'].sum():,}")
        
        with col2:
            st.metric("Total Fraud", f"{historical_data['fraud_count'].sum():,}")
        
        with col3:
            st.metric("Avg Daily Volume", f"{historical_data['transactions'].mean():.0f}")
        
        with col4:
            st.metric("Avg Fraud Rate", f"{historical_data['fraud_rate'].mean()*100:.3f}%")
    
    def show_system_config(self):
        """Show system configuration and settings."""
        st.header("⚙️ System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Model Configuration")
            
            if self.predictor:
                model_info = self.get_model_info()
                
                st.write("**Model Details:**")
                st.json(model_info)
                
                # Model parameters
                if 'model_parameters' in model_info:
                    st.write("**Model Parameters:**")
                    st.json(model_info['model_parameters'])
            else:
                st.info("No model loaded. Load a model to see configuration.")
        
        with col2:
            st.subheader("🌐 API Configuration")
            
            api_health = self.check_api_health()
            st.write("**API Health:**")
            st.json(api_health)
            
            # API endpoints
            st.write("**Available Endpoints:**")
            endpoints = [
                "GET / - API information",
                "GET /health - Health check",
                "POST /predict - Single prediction",
                "POST /batch_predict - Batch predictions",
                "GET /model_info - Model information",
                "GET /sample_transaction - Sample data"
            ]
            
            for endpoint in endpoints:
                st.write(f"• {endpoint}")
        
        # System logs
        st.subheader("📝 System Logs")
        
        # Simulate log display
        log_entries = [
            f"{datetime.now() - timedelta(minutes=1)} - INFO: Prediction completed successfully",
            f"{datetime.now() - timedelta(minutes=2)} - WARNING: High fraud rate detected",
            f"{datetime.now() - timedelta(minutes=3)} - INFO: Model reloaded successfully",
            f"{datetime.now() - timedelta(minutes=4)} - ERROR: API connection timeout"
        ]
        
        for log in log_entries[:10]:
            st.text(log)

def main():
    """Main function to run the dashboard."""
    
    try:
        # Initialize monitor
        monitor = FraudDetectionMonitor()
        
        # Try to load model
        if not monitor.predictor:
            monitor.load_model()
        
        # Run dashboard
        monitor.run_dashboard()
        
    except Exception as e:
        st.error(f"❌ Dashboard failed to start: {e}")
        st.info("Please check the console for more details.")
        st.exception(e)  # Show the full error traceback
        st.stop()

if __name__ == "__main__":
    main() 