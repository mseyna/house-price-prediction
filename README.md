# house-price-prediction
House Prices Prediction using Machine Learning (Ames Housing dataset)
# House Price Prediction - Advanced Regression Techniques

## Project Overview
The goal of this project is to build a machine learning model that predicts house prices based on various property characteristics.

This project is based on the Ames Housing dataset and follows a complete data science workflow, from data preprocessing to model training and evaluation.

## Objectives
- Perform data cleaning and preprocessing
- Handle missing values and categorical variables
- Apply feature scaling when necessary
- Train and compare several regression models
- Evaluate model performance using RMSE, MSE and R²
- Select the best-performing model

---

**Dataset:** [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Training set: 1,460 houses
- Test set: 1,459 houses
- Features: 82 (including target variable)

**Best Model Performance:**
- **XGBoost Regressor**
- **RMSE:** 0.13243
- **R² Score:** 0.9060 (90.6% variance explained)
- **Cross-Validation RMSE:** 0.12511 ± 0.01087

---

## Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost
```

### Running the Project
```bash
# Clone the repository
git clone https://github.com/yourusername/house-prices-prediction.git
cd house-prices-prediction

# Run the main script
python house_prices_pipeline.py
```

### Expected Outputs
- `submission.csv` - Predictions for Kaggle submission
- `saleprice_distribution.png`: Target variable analysis
- `model_comparison.png`: Model performance comparison
- `feature_importance.png`:  Top 20 important features

---

## Project Structure

```
house-prices-prediction/
│
├── data/
│   ├── train.csv                    # Training data
│   └── test.csv                     # Test data
│
├── notebooks/
│   └── house_prices_pipeline.ipynb  # Main Jupyter notebook
│
├── outputs/
│   ├── submission.csv               # Kaggle submission file
│   ├── saleprice_distribution.png   # Visualizations
│   ├── model_comparison.png
│   └── feature_importance.png
│
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```
---

## Methodology

### 1. **Data Exploration & Cleaning**
- Analyzed 1,460 houses with 80 features
- Identified and handled missing values (19 features with NaN)
- Applied appropriate imputation strategies:
  - Categorical NaN → "None" (indicates absence)
  - Numerical NaN → Median/Mean based on distribution
  - Logical imputation (e.g., GarageYrBlt = YearBuilt)

### 2. **Feature Engineering**
- **Ordinal Encoding:** 23 quality/condition features (Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0)
- **One-Hot Encoding:** 23 nominal categorical features (no natural order)
- **Log Transformation:** Applied to target variable (SalePrice) to normalize right-skewed distribution

### 3. **Data Preprocessing**
- Train/Validation split: 80/20
- StandardScaler for linear models (Ridge, Lasso, ElasticNet)
- No scaling for tree-based models (Random Forest, XGBoost, etc.)

### 4. **Model Training & Evaluation**

Tested **12 different models** with 5-fold cross-validation:

| Rank | Model | RMSE | R² | Type |
|------|-------|------|-----|------|
| 1 | XGBoost | 0.13243 | 0.9060 | Advanced Boosting |
| 2 | Gradient Boosting | 0.13526 | 0.9020 | Ensemble |
| 3 | LightGBM | 0.13541 | 0.9017 | Advanced Boosting |
| 4 | GBR (GridSearch Tuned) | 0.13655 | 0.9001 | Ensemble |
| 5 | CatBoost | 0.14122 | 0.8931 | Advanced Boosting |
| 6 | LassoCV | 0.14227 | 0.8915 | Linear (Tuned) |
| 7 | RidgeCV | 0.14266 | 0.8909 | Linear (Tuned) |
| 8 | ElasticNetCV | 0.14306 | 0.8903 | Linear (Tuned) |
| 9 | Random Forest | 0.14700 | 0.8842 | Ensemble |
| 10 | Decision Tree | 0.18896 | 0.8087 | Single Tree |
| 11 | Ridge | 0.19374 | 0.7989 | Linear |
| 12 | Linear Regression | 0.20116 | 0.7832 | Linear |

### 5. **Hyperparameter Tuning**
- **GridSearchCV** on Gradient Boosting (54 combinations tested)
- **Cross-Validation** for robust performance estimation
- Best parameters found:
  - `n_estimators`: 300
  - `max_depth`: 3
  - `learning_rate`: 0.1
  - `subsample`: 0.9

---

##  Key Insights

### Top 5 Most Important Features (by XGBoost):
1. **OverallQual** - Overall material and finish quality
2. **GrLivArea** - Above grade living area (sq ft)
3. **TotalBsmtSF** - Total basement area (sq ft)
4. **1stFlrSF** - First floor area (sq ft)
5. **GarageCars** - Garage capacity

### Key Learnings:
- **Boosting algorithms** (XGBoost, Gradient Boosting, LightGBM) outperformed all other models
- **Regularization matters:** LassoCV (tuned) achieved 0.142 RMSE vs Lasso (default) 0.433 RMSE
- **Cross-validation is crucial:** Single train/test split can be misleading
- **Feature encoding strategy:** Proper ordinal encoding significantly improved model performance

---

## Results & Performance

### Cross-Validation Results:
- **Gradient Boosting CV:** 0.12546 ± 0.00824
- **XGBoost CV:** 0.12511 ± 0.01087
- **Random Forest CV:** 0.14431 ± 0.00814

### Final Test Predictions:
- Minimum price: $34,900
- Maximum price: $611,657
- Mean price: $180,796
- Median price: $163,000

---

## Technologies Used

**Programming Language:**
- Python Jupyter Notebook

**Libraries:**
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Advanced Models:** XGBoost, LightGBM, CatBoost

**Techniques:**
- Train/Test Split & Cross-Validation
- Feature Engineering & Encoding
- Hyperparameter Tuning (GridSearchCV)
- Ensemble Methods (Random Forest, Gradient Boosting)
- Regularization (Ridge, Lasso, ElasticNet)

---

## Future Improvements

1.  **Advanced Techniques:**
   - Stacking/Blending multiple models
   - Automated feature selection (RFE, SelectKBest)
   - Outlier detection and removal

2. **Hyperparameter Optimization:**
   - Bayesian optimization with Optuna
   - Extended GridSearchCV for XGBoost
   - Early stopping with validation monitoring

---

## What I Learned

This project helped me develop skills in:
- End-to-end ML pipeline development
- Data cleaning and missing value strategies
- Feature engineering and encoding techniques
- Model selection and comparison
- Cross-validation and hyperparameter tuning
- Working with advanced boosting algorithms
- Interpreting model results and feature importance

---
