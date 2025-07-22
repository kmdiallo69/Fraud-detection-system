#!/usr/bin/env python3
"""
Streamlit Monitoring App Launcher
=================================

Simple launcher script for the fraud detection monitoring dashboard.

Usage:
    python run_monitoring.py
    streamlit run src/monitoring/app.py
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit monitoring app."""
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Check if plotly is installed
    try:
        import plotly
        print("✅ Plotly is installed")
    except ImportError:
        print("❌ Plotly not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    
    # Launch Streamlit app
    app_path = os.path.join("src", "monitoring", "app.py")
    
    if not os.path.exists(app_path):
        print(f"❌ App not found at {app_path}")
        return
    
    print("🚀 Launching Fraud Detection Monitoring Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8601")
    print("🔄 Press Ctrl+C to stop the server")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", "8601",
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == "__main__":
    main() 