# 🚀 Fraud Detection System

A comprehensive machine learning system for detecting fraudulent transactions using advanced ML algorithms, real-time API, and interactive monitoring dashboard.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Monitoring Dashboard](#-monitoring-dashboard)
- [Docker Deployment](#-docker-deployment)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🤖 Machine Learning
- **Advanced Feature Engineering**: Automated feature creation and selection
- **Multiple Algorithms**: XGBoost, Random Forest, Logistic Regression
- **Hyperparameter Tuning**: Automated optimization using GridSearchCV
- **Model Interpretability**: SHAP analysis for feature importance
- **Class Imbalance Handling**: SMOTE and other techniques
- **Model Persistence**: Save and load trained models

### 🌐 API & Services
- **FastAPI REST API**: High-performance prediction endpoints
- **Real-time Predictions**: Single and batch processing
- **Health Monitoring**: System status and model health checks
- **Auto-scaling Ready**: Containerized for easy scaling

### 📊 Monitoring & Visualization
- **Streamlit Dashboard**: Interactive real-time monitoring
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Interactive Charts**: Plotly visualizations
- **System Health**: Resource usage and API status
- **Historical Data**: Model performance tracking

### 🔧 DevOps & CI/CD
- **Docker Containerization**: Production-ready containers
- **GitHub Actions**: Automated testing and deployment
- **Multi-service Architecture**: API + Dashboard + Monitoring
- **Health Checks**: Automated system monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │    │  Preprocessing  │    │  Model Training │
│   (Kaggle)      │───▶│   Pipeline      │───▶│   & Validation  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Monitoring     │    │  FastAPI        │    │  Model Storage  │
│  Dashboard      │◀───│  REST API       │◀───│  (.pkl files)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker (optional)
- Kaggle API credentials

### 1. Clone and Setup
```bash
cd fraud-detection-system

# Activate virtual environment
source venv/bin/activate

# Install dependencies (use python -m pip)
python -m pip install -r requirements.txt
```

### 2. Data and Model Setup
```bash
# Download data and train model
python src/data_preprocessing.py
python src/train_model.py
```

### 3. Start Services
```bash
# Start API server
cd src/api && python main.py

# In another terminal, start monitoring dashboard
python run_monitoring.py
```

### 4. Access Services
- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## 📦 Installation

### Local Development

1. **Clone Repository**
```bash
git clone <repository-url>
cd fraud-detection-system
```

2. **Setup Virtual Environment**
```bash
# Activate existing environment
source venv/bin/activate

# Or create new environment
python -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
python -m pip install -r requirements.txt
```

4. **Setup Kaggle API** (for data download)
```bash
# Install kaggle CLI
python -m pip install kaggle

# Configure API credentials
mkdir ~/.kaggle
# Add your kaggle.json file to ~/.kaggle/
```

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build single container
docker build -t fraud-detection-system .
docker run -p 8000:8000 fraud-detection-system
```

## 🔧 Usage

### Data Preprocessing
```bash
python src/data_preprocessing.py
```
This will:
- Download the fraud detection dataset from Kaggle
- Perform exploratory data analysis
- Engineer features
- Handle class imbalance
- Save processed data and models

### Model Training
```bash
# Full training with all algorithms
python src/train_model.py

# Fast training (XGBoost only)
python src/train_model_fast.py
```

### Making Predictions
```python
from src.predict import FraudDetectionPredictor

# Initialize predictor
predictor = FraudDetectionPredictor()

# Single prediction
result = predictor.predict_single(transaction_data)

# Batch predictions
results = predictor.predict_batch(transactions_list)
```

## 🌐 API Documentation

### Endpoints

#### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-07-20T16:00:00Z"
}
```

#### 2. Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "amount": 100.50,
  "oldbalanceOrg": 1000.00,
  "newbalanceOrig": 899.50,
  "oldbalanceDest": 0.00,
  "newbalanceDest": 100.50,
  "type": "CASH_OUT"
}
```
**Response:**
```json
{
  "prediction": 0,
  "probability": 0.15,
  "is_fraud": false,
  "confidence": 0.85
}
```

#### 3. Batch Predictions
```http
POST /predict/batch
Content-Type: application/json

{
  "transactions": [
    {
      "amount": 100.50,
      "oldbalanceOrg": 1000.00,
      "newbalanceOrig": 899.50,
      "oldbalanceDest": 0.00,
      "newbalanceDest": 100.50,
      "type": "CASH_OUT"
    }
  ]
}
```

#### 4. Model Information
```http
GET /model/info
```
**Response:**
```json
{
  "model_name": "XGBoost",
  "version": "1.0.0",
  "accuracy": 0.9987,
  "precision": 0.9876,
  "recall": 0.9543,
  "f1_score": 0.9708,
  "training_date": "2024-07-20T10:00:00Z"
}
```

### API Usage Examples

#### Python
```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", json={
    "amount": 100.50,
    "oldbalanceOrg": 1000.00,
    "newbalanceOrig": 899.50,
    "oldbalanceDest": 0.00,
    "newbalanceDest": 100.50,
    "type": "CASH_OUT"
})
result = response.json()
print(f"Fraud: {result['is_fraud']}, Confidence: {result['confidence']}")
```

#### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "amount": 100.50,
       "oldbalanceOrg": 1000.00,
       "newbalanceOrig": 899.50,
       "oldbalanceDest": 0.00,
       "newbalanceDest": 100.50,
       "type": "CASH_OUT"
     }'
```

## 📊 Monitoring Dashboard

The Streamlit dashboard provides real-time monitoring with the following features:

### Dashboard Tabs

1. **🏠 Overview**
   - System status and health
   - Recent predictions
   - Key performance indicators

2. **📈 Model Performance**
   - Accuracy metrics over time
   - Confusion matrix
   - ROC curve and AUC score
   - Feature importance visualization

3. **🔍 Prediction Analysis**
   - Real-time prediction monitoring
   - Fraud detection trends
   - Transaction type analysis
   - Amount distribution

4. **📊 Historical Data**
   - Historical performance data
   - Model comparison charts
   - Training history
   - Performance trends

5. **⚙️ System Configuration**
   - Model parameters
   - System settings
   - API configuration
   - Health check status

### Access Dashboard
```bash
# Start dashboard
python run_monitoring.py

# Access at: http://localhost:8501
```

## 🐳 Docker Deployment

### Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Individual Containers
```bash
# Build API container
docker build -t fraud-detection-api .

# Run API
docker run -p 8000:8000 fraud-detection-api

# Run dashboard (separate container)
docker run -p 8501:8501 -v $(pwd)/src/monitoring:/app fraud-detection-dashboard
```

### Docker Services
- **API**: Port 8000
- **Dashboard**: Port 8501
- **Health Checks**: Automatic monitoring
- **Logging**: Structured logging

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
python -m pip install pytest pytest-cov

# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/ --cov-report=html
```

### Test Coverage
- Unit tests for prediction module
- API endpoint testing
- Model validation tests
- Integration tests

### Manual Testing
```bash
# Test API health
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"amount": 100.50, "type": "CASH_OUT"}'
```

## 📁 Project Structure

```
fraud-detection-system/
├── .github/workflows/          # CI/CD pipelines
├── data/                       # Data and models
│   ├── raw/                    # Raw data and trained models
│   └── processed/              # Processed data
├── notebooks/                  # Jupyter notebooks
├── src/                        # Source code
│   ├── api/                    # FastAPI application
│   ├── monitoring/             # Streamlit dashboard
│   ├── utils/                  # Utility modules
│   ├── data_preprocessing.py   # Data pipeline
│   ├── predict.py              # Prediction module
│   └── train_model*.py         # Training scripts
├── tests/                      # Unit tests
├── venv/                       # Virtual environment
├── docker-compose.yml          # Multi-service deployment
├── Dockerfile                  # Container configuration
├── requirements.txt            # Python dependencies
└── run_monitoring.py           # Dashboard launcher
```

For detailed structure, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

1. **Test Workflow** (`.github/workflows/test.yml`)
   - Runs on every push
   - Executes unit tests
   - Checks code quality
   - Reports test coverage

2. **Build Workflow** (`.github/workflows/build.yml`)
   - Runs on releases
   - Builds Docker images
   - Pushes to GitHub Container Registry
   - Deploys to staging

### Automated Tasks
- ✅ Code linting and formatting
- ✅ Unit test execution
- ✅ Security scanning
- ✅ Docker image building
- ✅ Container registry push

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guide
- Add docstrings to functions
- Include type hints
- Write unit tests
- Update documentation

### Testing Guidelines
- Maintain >90% test coverage
- Test all new endpoints
- Validate model predictions
- Check error handling

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle for the fraud detection dataset
- FastAPI for the excellent web framework
- Streamlit for the interactive dashboard
- XGBoost for the powerful ML algorithm

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)
- **Email**: support@your-domain.com

---

**Version**: 1.0.0  
**Last Updated**: July 20, 2024  
**Status**: Production Ready 🚀 