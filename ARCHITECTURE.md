# 🏗️ Fraud Detection System Architecture

## 📋 Overview

The Fraud Detection System is designed as a production-ready, scalable machine learning application with a microservices architecture. It combines advanced ML algorithms, real-time API services, and comprehensive monitoring capabilities.

## 🎯 System Goals

- **High Accuracy**: Detect fraudulent transactions with >99% accuracy
- **Real-time Processing**: Sub-second response times for predictions
- **Scalability**: Handle thousands of transactions per second
- **Reliability**: 99.9% uptime with automatic failover
- **Monitoring**: Comprehensive observability and alerting
- **Maintainability**: Modular design with clear separation of concerns

## 🏛️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  Web App │ Mobile App │ Third-party API │ Batch Processing     │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  Load Balancer │ Rate Limiting │ Authentication │ SSL/TLS       │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Server │ Streamlit Dashboard │ Health Monitoring      │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Prediction Service │ Model Management │ Data Processing        │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  Model Storage │ Feature Store │ Logs │ Metrics │ Config       │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Core Components

### 1. Data Processing Pipeline (`src/data_preprocessing.py`)

**Purpose**: Transform raw transaction data into ML-ready features

**Key Features**:
- Automated data validation and cleaning
- Feature engineering and selection
- Class imbalance handling (SMOTE)
- Data scaling and normalization
- Artifact persistence

**Data Flow**:
```
Raw Data → Validation → Cleaning → Feature Engineering → Scaling → Storage
```

**Technologies**:
- Pandas for data manipulation
- Scikit-learn for preprocessing
- Imbalanced-learn for SMOTE
- Joblib for serialization

### 2. Model Training Engine (`src/train_model.py`)

**Purpose**: Train and validate multiple ML models

**Key Features**:
- Multiple algorithm support (XGBoost, Random Forest, Logistic Regression)
- Hyperparameter optimization (GridSearchCV)
- Cross-validation and evaluation
- Model comparison and selection
- Model persistence and versioning

**Training Pipeline**:
```
Data Split → Model Training → Hyperparameter Tuning → Evaluation → Model Selection → Persistence
```

**Technologies**:
- XGBoost for gradient boosting
- Scikit-learn for traditional ML
- SHAP for model interpretability
- Pickle/Joblib for model storage

### 3. Prediction Service (`src/predict.py`)

**Purpose**: Provide real-time fraud predictions

**Key Features**:
- Model loading and caching
- Single and batch predictions
- Feature preprocessing
- Confidence scoring
- Error handling and logging

**Prediction Flow**:
```
Input Data → Feature Preprocessing → Model Inference → Post-processing → Response
```

**Technologies**:
- NumPy for numerical operations
- Pandas for data handling
- Joblib for model loading
- Logging for monitoring

### 4. REST API (`src/api/main.py`)

**Purpose**: Expose prediction services via HTTP endpoints

**Key Features**:
- FastAPI framework for high performance
- Automatic API documentation
- Request validation and error handling
- Health monitoring endpoints
- Rate limiting and security

**API Endpoints**:
- `GET /health` - System health check
- `POST /predict` - Single transaction prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Model information
- `GET /docs` - Interactive API documentation

**Technologies**:
- FastAPI for web framework
- Pydantic for data validation
- Uvicorn for ASGI server
- Prometheus for metrics

### 5. Monitoring Dashboard (`src/monitoring/app.py`)

**Purpose**: Real-time system monitoring and visualization

**Key Features**:
- Real-time performance metrics
- Interactive visualizations
- Model performance tracking
- System health monitoring
- Historical data analysis

**Dashboard Tabs**:
- Overview: System status and KPIs
- Model Performance: Accuracy, precision, recall
- Prediction Analysis: Real-time monitoring
- Historical Data: Trends and patterns
- System Configuration: Settings and parameters

**Technologies**:
- Streamlit for web interface
- Plotly for interactive charts
- Pandas for data manipulation
- Requests for API communication

## 🔄 Data Flow Architecture

### 1. Training Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Kaggle    │───▶│  Raw Data   │───▶│ Preprocess  │───▶│   Train     │
│   Dataset   │    │   Storage   │    │  Pipeline   │    │   Models    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Model     │◀───│ Evaluation  │◀───│ Validation  │◀───│   Model     │
│  Storage    │    │  Metrics    │    │   Results   │    │  Training   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 2. Inference Pipeline

```
┌─────────────┐    ┌────────────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │───▶│   API (9000→8000) │───▶│ Prediction  │───▶│   Model     │
│  Request    │    │   Server          │    │  Service    │    │   Cache     │
└─────────────┘    └────────────────────┘    └─────────────┘    └─────────────┘
                                                             │
                                                             ▼
┌─────────────┐    ┌────────────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │◀───│   API (9000→8000) │◀───│ Prediction  │◀───│   Model     │
│  Response   │    │   Server          │    │  Results    │    │  Inference  │
└─────────────┘    └────────────────────┘    └─────────────┘    └─────────────┘
```

### 3. Monitoring Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   System    │───▶│   Metrics   │───▶│  Streamlit  │───▶│   Real-time │
│  Events     │    │  Collector  │    │  Dashboard  │    │  Display    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Logs      │    │ Historical  │    │ Interactive │    │   Alerts    │
│  Storage    │    │   Data      │    │   Charts    │    │  & Reports  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 🗄️ Data Architecture

### Data Storage Strategy

#### 1. Model Artifacts
```
data/raw/
├── fraud_detection_model_*.pkl      # Trained models
├── scaler_*.pkl                     # Feature scalers
├── feature_columns_*.pkl            # Feature names
└── model_metadata_*.json            # Model metadata
```

#### 2. Processed Data
```
data/processed/
├── X_train_*.pkl                    # Training features
├── X_test_*.pkl                     # Test features
├── y_train_*.pkl                    # Training labels
├── y_test_*.pkl                     # Test labels
└── feature_importance_*.json        # Feature importance
```

#### 3. Logs and Metrics
```
logs/
├── training.log                     # Training logs
├── api.log                          # API request logs
├── predictions.log                  # Prediction logs
└── system.log                       # System logs
```

### Data Flow Patterns

#### 1. Batch Processing
- **Frequency**: Daily model retraining
- **Data Volume**: Full dataset processing
- **Processing Time**: 10-30 minutes
- **Output**: Updated model artifacts

#### 2. Real-time Processing
- **Frequency**: Per-request
- **Data Volume**: Single transactions
- **Processing Time**: <100ms
- **Output**: Prediction results

#### 3. Monitoring Data
- **Frequency**: Continuous
- **Data Volume**: Metrics and logs
- **Processing Time**: Real-time
- **Output**: Dashboard updates

## 🔒 Security Architecture

### 1. API Security
- **Authentication**: JWT tokens (future)
- **Authorization**: Role-based access control
- **Rate Limiting**: Request throttling
- **Input Validation**: Pydantic schemas
- **HTTPS**: SSL/TLS encryption

### 2. Data Security
- **Encryption**: Data at rest and in transit
- **Access Control**: File system permissions
- **Audit Logging**: All access attempts
- **Data Masking**: Sensitive data protection

### 3. Model Security
- **Model Signing**: Digital signatures
- **Version Control**: Model versioning
- **Access Logging**: Model usage tracking
- **Tamper Detection**: Integrity checks

## 📊 Performance Architecture

### 1. Scalability Patterns

#### Horizontal Scaling
- **API Servers**: Multiple FastAPI instances
- **Load Balancing**: Round-robin distribution
- **Database**: Read replicas (future)
- **Caching**: Redis for model caching

#### Vertical Scaling
- **Memory**: Model caching in RAM
- **CPU**: Multi-threaded processing
- **Storage**: SSD for fast I/O
- **Network**: High-bandwidth connections

### 2. Performance Metrics

#### Response Time
- **API Latency**: <100ms for predictions
- **Model Loading**: <5s on startup
- **Batch Processing**: <1s per 1000 records
- **Dashboard Updates**: <2s refresh rate

#### Throughput
- **Single Instance**: 1000 req/sec
- **Scaled Instance**: 10000 req/sec
- **Batch Processing**: 10000 records/sec
- **Concurrent Users**: 1000+ dashboard users

### 3. Optimization Strategies

#### Model Optimization
- **Model Compression**: Quantization
- **Feature Selection**: Dimensionality reduction
- **Caching**: Pre-computed features
- **Batching**: Batch predictions

#### System Optimization
- **Connection Pooling**: Database connections
- **Memory Management**: Garbage collection
- **Async Processing**: Non-blocking I/O
- **CDN**: Static asset delivery

## 🔄 Deployment Architecture

### 1. Container Strategy

#### Docker Containers
```yaml
services:
  fraud-detection-api:
    image: fraud-detection-api:latest
    ports:
      - "9000:8000"  # External 9000 → Internal 8000
    environment:
      - MODEL_DIR=/app/data/raw
    volumes:
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  fraud-detection-dashboard:
    image: fraud-detection-dashboard:latest
    ports:
      - "8601:8501"  # External 8601 → Internal 8501
    environment:
      - API_URL=http://fraud-detection-api:8000
    depends_on:
      - fraud-detection-api
```

#### Port Policy
- **API Service**: Exposed externally on port **9000** (internally 8000 in container)
- **Dashboard**: Exposed externally on port **8601** (internally 8501 in container)
- **No use of 8000/8080** for external access to avoid common port conflicts
- **All internal service-to-service communication** uses container-internal ports (8000 for API, 8501 for dashboard)

### 2. Orchestration Strategy

#### Docker Compose (Current)
- **Development**: Local development environment
- **Testing**: Integration testing
- **Staging**: Pre-production validation
- **Simple Production**: Single-server deployments

#### Kubernetes (Future)
- **Production**: Multi-node deployments
- **Auto-scaling**: Horizontal Pod Autoscaling
- **Load Balancing**: Ingress controllers
- **Service Mesh**: Istio for microservices

### 3. CI/CD Pipeline

#### GitHub Actions Workflow
```yaml
name: Build and Deploy
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: python -m pytest tests/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker image
        run: docker build -t fraud-detection-system .
      - name: Push to registry
        run: docker push ghcr.io/user/fraud-detection-system:latest
```

## 🔍 Monitoring Architecture

### 1. Observability Stack

#### Metrics Collection
- **Application Metrics**: Custom business metrics
- **System Metrics**: CPU, memory, disk usage
- **Infrastructure Metrics**: Container health, network
- **Business Metrics**: Predictions, accuracy, throughput

#### Logging Strategy
- **Structured Logging**: JSON format logs
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Log Aggregation**: Centralized log storage
- **Log Retention**: 30 days for production

#### Tracing Strategy
- **Distributed Tracing**: Request flow tracking
- **Performance Profiling**: Bottleneck identification
- **Error Tracking**: Exception monitoring
- **User Journey**: End-to-end request tracking

#### Dashboard Access
- **Dashboard**: Exposed externally on port **8601** (internally 8501)

### 2. Alerting Strategy

#### Alert Rules
- **High Error Rate**: >5% error rate
- **High Latency**: >500ms response time
- **Low Accuracy**: <95% model accuracy
- **System Down**: Health check failures

#### Notification Channels
- **Email**: Critical alerts
- **Slack**: Team notifications
- **PagerDuty**: On-call escalation
- **SMS**: Emergency alerts

### 3. Dashboard Strategy

#### Real-time Dashboards
- **System Health**: Overall system status
- **Performance Metrics**: Response times, throughput
- **Business Metrics**: Predictions, accuracy
- **Error Tracking**: Error rates, types

#### Historical Dashboards
- **Trend Analysis**: Performance over time
- **Capacity Planning**: Resource usage trends
- **Model Performance**: Accuracy degradation
- **User Behavior**: Usage patterns

## 🔮 Future Architecture Enhancements

### 1. Microservices Evolution

#### Service Decomposition
- **User Service**: Authentication and authorization
- **Model Service**: Model management and versioning
- **Prediction Service**: Real-time inference
- **Data Service**: Data processing and storage
- **Notification Service**: Alerting and notifications

#### Service Communication
- **Synchronous**: REST APIs for real-time requests
- **Asynchronous**: Message queues for batch processing
- **Event-driven**: Event streaming for real-time updates
- **gRPC**: High-performance inter-service communication

### 2. Data Architecture Evolution

#### Feature Store
- **Real-time Features**: Online feature serving
- **Batch Features**: Offline feature computation
- **Feature Registry**: Feature metadata management
- **Feature Monitoring**: Data quality and drift detection

#### Model Registry
- **Model Versioning**: Semantic versioning
- **Model Lineage**: Training data and parameters
- **Model Deployment**: A/B testing and canary deployments
- **Model Monitoring**: Performance and drift detection

### 3. Infrastructure Evolution

#### Cloud-Native Architecture
- **Kubernetes**: Container orchestration
- **Service Mesh**: Istio for microservices
- **Serverless**: FaaS for event-driven processing
- **Edge Computing**: Local prediction capabilities

#### Data Platform
- **Data Lake**: Raw data storage
- **Data Warehouse**: Structured analytics
- **Stream Processing**: Real-time data processing
- **ML Platform**: End-to-end ML lifecycle

## 📋 Architecture Decision Records (ADRs)

### ADR-001: FastAPI Framework Selection
- **Decision**: Use FastAPI for the REST API
- **Rationale**: High performance, automatic documentation, type safety
- **Alternatives**: Flask, Django REST Framework
- **Consequences**: Fast development, excellent developer experience

### ADR-002: XGBoost as Primary Model
- **Decision**: Use XGBoost as the primary ML algorithm
- **Rationale**: High accuracy, good interpretability, production-ready
- **Alternatives**: Random Forest, Neural Networks
- **Consequences**: Excellent performance, good balance of speed and accuracy

### ADR-003: Docker Containerization
- **Decision**: Containerize all services using Docker
- **Rationale**: Consistency, portability, easy deployment
- **Alternatives**: Virtual machines, bare metal
- **Consequences**: Easy scaling, consistent environments

### ADR-004: Streamlit for Monitoring
- **Decision**: Use Streamlit for the monitoring dashboard
- **Rationale**: Rapid development, Python-native, interactive
- **Alternatives**: Dash, Plotly Dash, custom web app
- **Consequences**: Fast development, good user experience

---

**Last Updated**: July 20, 2024  
**Version**: 1.1.0  
**Status**: Production Ready
