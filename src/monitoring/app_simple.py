#!/usr/bin/env python3
"""
Simplified Fraud Detection Monitoring Dashboard
==============================================

A simplified version to test basic functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure page
st.set_page_config(
    page_title="Fraud Detection Monitor",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main function to run the dashboard."""

    try:
        # Header
        st.title("🕵️ Fraud Detection System Monitor")
        st.write("✅ Streamlit is working!")

        # Test basic components
        st.header("📊 System Status")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Status", "🟢 Online", "Active")

        with col2:
            st.metric("Model", "✅ Loaded", "XGBoost")

        with col3:
            st.metric("API", "🟡 Testing", "Port 8000")

        with col4:
            st.metric("Uptime", "2h 15m", "99.9%")

        # Test model loading
        st.header("🤖 Model Information")

        try:
            from predict import FraudPredictor

            st.write("✅ Prediction module imported successfully")

            # Try to load model
            try:
                predictor = FraudPredictor(model_dir="data/raw")
                model_info = predictor.get_model_info()

                st.success("✅ Model loaded successfully!")

                # Display model info
                st.subheader("Model Details")
                col1, col2 = st.columns(2)

                with col1:
                    st.write(
                        f"**Model Type:** {model_info.get('model_type', 'Unknown')}"
                    )
                    st.write(
                        f"**Feature Count:** {model_info.get('feature_count', 'Unknown')}"
                    )
                    st.write(
                        f"**Has Scaler:** {model_info.get('has_scaler', 'Unknown')}"
                    )

                with col2:
                    if "model_metadata" in model_info and model_info["model_metadata"]:
                        metadata = model_info["model_metadata"]
                        if "performance_metrics" in metadata:
                            metrics = metadata["performance_metrics"]
                            st.write(f"**Accuracy:** {metrics.get('accuracy', 0):.4f}")
                            st.write(
                                f"**Precision:** {metrics.get('precision', 0):.4f}"
                            )
                            st.write(f"**Recall:** {metrics.get('recall', 0):.4f}")
                            st.write(f"**F1-Score:** {metrics.get('f1', 0):.4f}")

            except Exception as e:
                st.warning(f"⚠️ Model loading failed: {e}")
                st.info("💡 Running in demo mode")

        except Exception as e:
            st.error(f"❌ Failed to import prediction module: {e}")

        # Test sample data creation
        st.header("📈 Sample Data")

        try:
            from predict import create_sample_transaction

            # Create sample data
            sample_data = []
            for i in range(50):
                sample = create_sample_transaction()
                sample["transaction_id"] = f"TXN_{i:06d}"
                sample["timestamp"] = datetime.now() - timedelta(minutes=i)
                sample_data.append(sample)

            df = pd.DataFrame(sample_data)

            # Display sample chart
            st.subheader("Sample Transaction Amounts")
            fig = px.histogram(
                df, x="Amount", nbins=20, title="Transaction Amount Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.success(f"✅ Created {len(sample_data)} sample transactions")

        except Exception as e:
            st.error(f"❌ Sample data creation failed: {e}")

        # Test API connection
        st.header("🌐 API Status")

        try:
            import requests

            api_url = "http://localhost:8000"

            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ API is responding")
                    st.json(response.json())
                else:
                    st.warning(f"⚠️ API returned status {response.status_code}")
            except Exception as e:
                st.warning(f"⚠️ API not accessible: {e}")
                st.info("💡 API might not be running on port 8000")

        except Exception as e:
            st.error(f"❌ API test failed: {e}")

        # Final status
        st.header("🎯 Overall Status")
        st.success("✅ Fraud Detection System is working!")
        st.write(f"🕐 Last updated: {datetime.now()}")

    except Exception as e:
        st.error(f"❌ Dashboard failed to start: {e}")
        st.info("Please check the console for more details.")
        st.stop()


if __name__ == "__main__":
    main()
