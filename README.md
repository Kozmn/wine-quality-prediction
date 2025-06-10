# Wine Quality Prediction

Machine learning analysis of wine quality prediction using physicochemical properties. Complete end-to-end pipeline from data exploration to model optimization.

## Overview

This project analyzes the UCI Wine Quality dataset to predict wine quality scores (3-8) based on 11 physicochemical features. The analysis compares Linear Regression and XGBoost models, demonstrating significant performance improvements through feature engineering and hyperparameter optimization.

## Key Results

- **Best Model**: XGBoost Optimized achieves **R² = 0.6234** (73% improvement over baseline)
- **Important Features**: Alcohol content, volatile acidity, sulphates, citric acid
- **Methodology**: Stratified sampling, feature standardization, RandomizedSearchCV optimization

## Dataset

- **Source**: UCI Wine Quality Dataset (Red Wine)
- **Size**: 1,599 samples, 11 features
- **Target**: Wine quality scores (3-8 scale)
- **Challenge**: Imbalanced dataset (82% samples in quality 5-6)

## Analysis Pipeline

1. **Exploratory Data Analysis** - Feature correlations and distributions
2. **Data Preprocessing** - Stratified train/test split, feature standardization
3. **Model Development** - Linear Regression baseline, XGBoost implementation
4. **Hyperparameter Optimization** - RandomizedSearchCV with cross-validation
5. **Performance Evaluation** - MSE, MAE, R² metrics comparison

## Technical Stack

- **Python**: Data analysis and machine learning
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: ML models, preprocessing, model selection
- **XGBoost**: Gradient boosting implementation
- **Matplotlib/Seaborn**: Data visualization
- **Flask/Flask-RESTX**: REST API development with Swagger docs
- **Joblib**: Model serialization and deployment

## Model Performance

| Model | Test R² | Improvement |
|-------|---------|-------------|
| Linear Regression | 0.3594 | Baseline |
| XGBoost Basic | 0.5891 | +63.9% |
| XGBoost Optimized | 0.6234 | +73.4% |

## Repository Structure

```
├── api/
│   └── app.py                       # Flask REST API with Swagger docs
├── data/
│   └── winequality-red.csv          # Wine quality dataset
├── models/
│   ├── xgb_best_all_features.pkl    # Trained XGBoost model
│   └── scaler.pkl                   # Feature scaler
├── notebooks/
│   └── 01_wine_quality_analysis.ipynb # Complete analysis pipeline
├── requirements.txt                  # Python dependencies
└── README.md                        # Project documentation
```

## API Service

The project includes a complete REST API built with Flask and Flask-RESTX for wine quality prediction.

### Features

- **RESTful Endpoints**: Health check, model info, and prediction
- **Swagger Documentation**: Auto-generated API docs at `/docs/`
- **Input Validation**: Comprehensive data validation and error handling
- **Production Ready**: Proper logging, error handling, and documentation

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start API Server**
   ```bash
   cd api
   python app.py
   ```

3. **Access Documentation**
   - API Docs: http://localhost:8000/docs/
   - Health Check: http://localhost:8000/health

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API overview and navigation |
| `/health` | GET | Health check and system status |
| `/api/data` | GET | API metadata and version info |
| `/api/info` | GET | Model specifications and features |
| `/api/predict` | POST | Wine quality prediction |

### Example Usage

```bash
# Predict wine quality
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'

# Response
{
  "predicted_quality": 5.03,
  "input_features": {...},
  "message": "prediction successful"
}
```

## Key Insights

- **Alcohol content** is the strongest predictor of wine quality
- **XGBoost significantly outperforms** linear models on this dataset
- **Feature standardization** is crucial for optimal performance
- **Hyperparameter tuning** provides meaningful improvements (5.8% boost)
- **Model shows good generalization** with minimal overfitting

## Skills Demonstrated

- End-to-end machine learning pipeline development
- Feature engineering and data preprocessing
- Model comparison and evaluation
- Hyperparameter optimization techniques
- Data visualization and statistical analysis
- REST API development and documentation
- Model deployment and productionization
- Code documentation and reproducible research

---

