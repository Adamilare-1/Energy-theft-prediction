# Electricity Theft Detection ⚡🕵️‍♂️

A machine learning pipeline to predict electricity theft using consumption patterns, voltage data, and customer attributes.

## Key Features
- **Data Preprocessing**: Handles missing values, feature engineering, and scaling.
- **Model Training**: Includes `RandomForest`, `XGBoost`, and `LightGBM` with hyperparameter tuning.
- **Interpretability**: SHAP values and feature importance analysis.
- **Deployment-ready**: Serialized pipeline for production use.

## Results
| Model          | ROC AUC | Precision | Recall |
|----------------|---------|-----------|--------|
| RandomForest   | 0.92    | 0.85      | 0.81   |
| XGBoost        | 0.91    | 0.83      | 0.83   |
| LightGBM       | 0.93    | 0.86      | 0.82   |

**Top 3 Predictive Features**:
1. `consumption_diff` (Actual - Billed)
2. `average_voltage`
3. `power_factor`

## Installation
```bash
git clone https://github.com/yourusername/electricity-theft-detection.git
pip install -r requirements.txt
