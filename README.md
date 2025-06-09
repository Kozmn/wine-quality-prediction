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

## Model Performance

| Model | Test R² | Improvement |
|-------|---------|-------------|
| Linear Regression | 0.3594 | Baseline |
| XGBoost Basic | 0.5891 | +63.9% |
| XGBoost Optimized | 0.6234 | +73.4% |

## Repository Structure

```
├── data/
│   └── winequality-red.csv          # Wine quality dataset
├── notebooks/
│   └── 01_wine_quality_analysis.ipynb # Complete analysis pipeline
├── requirements.txt                  # Python dependencies
└── README.md                        # Project documentation
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
- Code documentation and reproducible research

---

