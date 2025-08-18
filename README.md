# Fraud Detection System - Financial Risk Management Capstone

A comprehensive, enterprise-grade fraud detection and risk management system designed for financial institutions. This project demonstrates advanced software engineering practices, robust testing, and business-focused solutions.

## 🎯 Project Overview

This system provides real-time fraud detection capabilities with:
- **Modular Architecture**: Clean, maintainable codebase following SOLID principles
- **Enterprise Reliability**: Comprehensive testing, error handling, and monitoring
- **Business Intelligence**: Interactive dashboard with real-time metrics and ROI tracking
- **Model Explainability**: SHAP and LIME integration for transparent decision-making
- **Production Ready**: FastAPI REST API with Docker deployment support

## 🏗️ Architecture

```
src/
├── core/           # Configuration, logging, exceptions
├── data/           # Data loading, preprocessing, validation
├── models/         # ML models, training, evaluation
├── api/            # FastAPI REST endpoints
├── dashboard/      # Streamlit interactive dashboard
└── utils/          # Helper functions, metrics, visualizations

tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── performance/    # Performance tests
└── fixtures/       # Test data

config/             # Configuration files
docs/               # Documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Adey_fraud_detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the System

1. **Start the API server**
   ```bash
   python -m src.api.fastapi_app
   ```

2. **Launch the dashboard**
   ```bash
   streamlit run src/dashboard/streamlit_app.py
   ```

3. **Run tests**
   ```bash
   pytest tests/
   ```

## 📊 Features

### Core Capabilities
- **Real-time Fraud Detection**: Instant transaction scoring
- **Risk Assessment**: Probability-based risk scoring
- **Model Explainability**: Transparent decision explanations
- **Alert System**: Configurable fraud alerts
- **Performance Monitoring**: Real-time metrics tracking

### Business Intelligence
- **Interactive Dashboard**: Real-time monitoring and analytics
- **ROI Tracking**: Cost savings and fraud prevention metrics
- **Risk Analytics**: Portfolio risk assessment
- **Audit Trail**: Complete transaction history
- **Custom Reports**: Automated report generation

### Technical Excellence
- **Modular Design**: Clean, maintainable architecture
- **Comprehensive Testing**: 90%+ test coverage
- **CI/CD Pipeline**: Automated quality checks
- **Error Handling**: Robust exception management
- **Performance Optimization**: Fast response times

## 🔧 Configuration

The system uses a hierarchical configuration system:

```yaml
# config/config.yaml
model:
  algorithm: "random_forest"
  threshold: 0.5

api:
  host: "0.0.0.0"
  port: 8000

dashboard:
  port: 8501
```

Environment variables can override any configuration setting.

## 📈 API Usage

### Single Prediction
```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "transaction_id": "txn_123",
    "amount": 150.50,
    "merchant_id": "merchant_001",
    "customer_id": "customer_123"
})

print(response.json())
# {
#   "transaction_id": "txn_123",
#   "fraud_probability": 0.0234,
#   "fraud_prediction": false,
#   "risk_level": "LOW",
#   "confidence_score": 0.95
# }
```

### Batch Prediction
```python
response = requests.post("http://localhost:8000/predict/batch", json={
    "transactions": [
        {"transaction_id": "txn_1", "amount": 100.0, ...},
        {"transaction_id": "txn_2", "amount": 250.0, ...}
    ]
})
```

## 🧪 Testing

### Running Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=src

# Specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

### Test Structure
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Load and stress testing
- **Fixtures**: Reusable test data

## 📊 Dashboard

The interactive dashboard provides:

- **Real-time Metrics**: Live fraud detection statistics
- **Risk Analytics**: Portfolio risk assessment
- **Model Performance**: Accuracy, precision, recall tracking
- **Business Impact**: Cost savings and ROI calculations
- **Alert Management**: Fraud alert configuration

Access at: `http://localhost:8501`

## 🔍 Model Explainability

The system provides multiple explainability methods:

- **SHAP Values**: Feature importance and contribution analysis
- **LIME Explanations**: Local interpretable model explanations
- **Feature Importance**: Global feature ranking
- **Decision Paths**: Model decision visualization

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t fraud-detection .

# Run container
docker run -p 8000:8000 -p 8501:8501 fraud-detection
```

### Production Considerations
- **Load Balancing**: Multiple API instances
- **Database**: PostgreSQL for transaction storage
- **Caching**: Redis for performance optimization
- **Monitoring**: Prometheus + Grafana
- **Security**: HTTPS, authentication, rate limiting

## 📈 Business Impact

### Key Metrics
- **Fraud Detection Rate**: % of fraud caught
- **False Positive Rate**: % of legitimate transactions flagged
- **Cost Savings**: Estimated fraud prevention savings
- **ROI**: Return on investment from fraud prevention
- **Risk Reduction**: Portfolio risk assessment

### Financial Benefits
- **Immediate**: Real-time fraud prevention
- **Long-term**: Reduced fraud losses
- **Operational**: Automated monitoring and alerts
- **Compliance**: Audit trail and transparency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact: your.email@example.com
- Documentation: `/docs` directory

## 🏆 Capstone Features

This project demonstrates:

### Technical Excellence
- **Advanced Python**: Type hints, dataclasses, async/await
- **Design Patterns**: Factory, Strategy, Observer patterns
- **Clean Architecture**: Separation of concerns, dependency injection
- **Performance**: Optimized algorithms, caching, async processing

### Business Focus
- **ROI Tracking**: Quantified business impact
- **Risk Management**: Comprehensive risk assessment
- **Compliance**: Audit trails and transparency
- **Scalability**: Enterprise-ready architecture

### Professional Standards
- **Testing**: Comprehensive test coverage
- **Documentation**: Clear, professional documentation
- **CI/CD**: Automated quality assurance
- **Monitoring**: Production-ready observability

---

**Built with ❤️ for the finance sector**