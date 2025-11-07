# Pharmacy Inventory Prediction System


![app.gif](image/app.gif)


This project demonstrates an AI-powered inventory management system for pharmacies. The application predicts stock usage, identifies expiry risks, and provides actionable insights to support pharmacy administrators in making timely reordering decisions. Built with **Streamlit** for an interactive web interface and machine learning models for demand forecasting.

## Live Demo

Check out the live application: [Diabetes Prediction App](https://diabetes-prediction-app-m6qd.onrender.com/)

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model](#model)
4. [Features](#features)
5. [Installation](#installation)
6. [How It Works](#how-it-works)
7. [Project Structure](#project-structure)
8. [Explanation Methods](#explanation-methods)
9. [Model Performance](#model-performance)
10. [Project Motivation](#project-motivation)
11. [Contributing](#contributing)
12. [License](#license)
13. [Contacts](#contacts)

---

## Overview

The **Pharmacy Inventory Prediction System** leverages machine learning to forecast medication demand, identify expiry risks, and prevent stockouts. Built with **Streamlit**, the system provides actionable insights for pharmacy inventory management. The system uses demand forecasting models and risk assessment algorithms to optimize inventory levels and reduce waste.

### Why This Project?

Effective pharmacy inventory management is crucial for patient care and business efficiency. This project addresses common challenges:
- **Stockout Prevention**: Ensures essential medications are always available
- **Waste Reduction**: Minimizes losses from expired medications
- **Cost Optimization**: Maintains optimal inventory levels
- **Decision Support**: Provides data-driven insights for reordering decisions

This project demonstrates:
- Practical application of machine learning in healthcare operations
- Time series forecasting for demand prediction
- Risk assessment algorithms for inventory management
- Real-world deployment of AI solutions for operational efficiency

---

## Dataset

The system is designed to work with pharmacy inventory data including:

### Required Data Structure
- **Product Information**: Product ID, name, current stock levels
- **Usage Data**: Daily/monthly dispensing volumes, historical trends
- **Supplier Data**: Lead times, reliability metrics
- **Batch Information**: Expiry dates, batch quantities, multiple batches per product
- **Time Series Data**: Historical dispensing records for forecasting

### Sample Inventory Data Structure
| product_id | product_name | current_stock | daily_dispensing_avg | monthly_dispensing_avg | supplier_lead_time_days | batch_expiry_dates | batch_quantities |
|------------|--------------|---------------|---------------------|----------------------|-------------------------|-------------------|------------------|
| P001       | Paracetamol | 5000         | 200                | 6000                | 7                      | ['2025-12-01']    | [5000]          |
| P002       | Ibuprofen    | 3000         | 150                | 4500                | 14                     | ['2025-11-30']    | [3000]          |
| P003       | Aspirin      | 2000         | 100                | 3000                | 10                     | ['2025-10-31']    | [2000]          |

### Key Features of the Dataset
- **Multiple Batches**: Support for products with different expiry dates
- **Usage Patterns**: Historical dispensing data for accurate forecasting
- **Supplier Intelligence**: Lead time information for reorder planning
- **Real-time Updates**: Current stock levels that can be updated regularly
---

## Model
You can learn more about the model in detail from [here](notebooks/Model.ipynb). The `GradientBoostingClassifier` model was chosen through experimentation and showed the best performance.
1. Stability & Generalization
Overfitting Control: Unlike Random Forest, which may sometimes be prone to overfitting, Gradient Boosting builds trees sequentially and optimizes for errors made by previous models. This helps in better generalization, which is crucial when dealing with real-world, unseen data.
Robust Performance on Noisy Data: Since Gradient Boosting focuses on correcting errors iteratively, it is often more stable than XGBoost when dealing with noise in data.
2. Interpretability & Feature Importance
Better Feature Attribution: Gradient Boosting is known for generating feature importance that can be easily interpreted using SHAP (Shapley Additive Explanations), as seen in your explainer.py file​explainer. This allows domain experts and healthcare professionals to understand what factors contribute most to the predictions.
3. Performance Beyond Accuracy (ROC AUC)
Strong ROC AUC Score (95.37%): Even though its accuracy is slightly lower than XGBoost and Random Forest, Gradient Boosting has the highest ROC AUC score (95.37%), meaning it is better at distinguishing between positive and negative cases. This is especially crucial in medical applications like diabetes prediction, where precision in identifying high-risk patients is more important than just accuracy.
4. Computational Efficiency
Less Memory Intensive than Random Forest: Gradient Boosting typically requires fewer trees than Random Forest to achieve comparable performance, making it a better choice for deployment in resource-constrained environments.
Faster Training than XGBoost: While XGBoost is an optimized implementation, its hyperparameter tuning and tree-pruning mechanisms can be computationally expensive.
5. Better Handling of Class Imbalances
In real-world applications like diabetes prediction, datasets often contain imbalanced classes (more non-diabetic than diabetic cases). Gradient Boosting handles such imbalances better due to its iterative re-weighting mechanism.
The required hyperparameters were identified using the `optuna` optimizer. For the model to function, it needs `FeatureEngineering`, `WoEEncoding`, and `ColumnSelector` transformers, which are combined through a pipeline.
`Cross-validation` and `ROC AUC` were used for model selection because the number of observations was small, and splitting into test/train sets would have been inaccurate.
The final prediction pipeline is built using Gradient Boosting and incorporates several custom transformers to enhance feature quality:

FeatureEngineering: Creates new features (e.g., PregnancyRatio, RiskScore, InsulinEfficiency) that capture underlying relationships in the data.
WoEEncoding: Transforms selected features into their Weight of Evidence (WoE) representation, improving interpretability.
ColumnSelector: Selects the most relevant engineered features for the final model.
The pipeline was constructed after experimenting with multiple models (including SVM, Decision Tree, and Random Forest) and was ultimately evaluated using cross-validation with the ROC AUC metric. The final model is saved as diabetes_prediction_pipeline.joblib for deployment.

### About tarnsformers
#### **1. FeatureEngineering**
Transforms raw data into a format suitable for machine learning. This includes scaling, encoding, creating new features, or handling missing data.


#### **2. WoEEncoding (Weight of Evidence Encoding)**
Features must help to better explain the `Outcome` after WoE.
The Weight of Evidence (WoE) for a category in a feature is calculated as:

Where:
- `P(Feature = X | Target = 1)`: Proportion of positive cases (`Target = 1`) for the category `X`.
- `P(Feature = X | Target = 0)`: Proportion of negative cases (`Target = 0`) for the category `X`.

##### Example:
If a feature `X` has the following counts:
- For `Target = 1` (Positive): `N1`
- For `Target = 0` (Negative): `N0`

#### **3. ColumnSelector**
Selects specific columns *Pregnancies*, *Glucose*, *BMI*, *PregnancyRatio*,
    *RiskScore*, *InsulinEfficiency*, *Glucose_BMI*, *BMI_Age*,
    *Glucose_woe*, *RiskScore_woe* after `FeatureEngineering`, it helps remove noice columns.

---
## Features

1. **Interactive Inventory Input**: Enter current stock levels, dispensing volumes, and supplier information.
2. **Demand Forecasting**: Predict stock usage for 1 week, 1 month, and 3 months.
3. **Risk Assessment**: Identify expiry risks and stockout probabilities.
4. **Batch Management**: Track multiple batches with different expiry dates.
5. **Actionable Insights**: Get recommendations for reorder timing and inventory optimization.

6. **Performance Dashboard**: View inventory health metrics and risk indicators.
7. **Feature Importance Analysis**: Understand which factors most influence demand forecasts.
8. **Real-Time Analysis**: Update inventory data and get immediate predictions.
9. **Educational Content**: Learn about pharmacy inventory management best practices.

---

## Installation

### Prerequisites
- Python 3.10 or above
- Pip package manager

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/abdullahi682/diabetes-prediction-app/tree/main
   cd Diabetes-Prediction
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application locally:
   ```bash
   streamlit run main.py
   ```

---

## How It Works

### Application Workflow
1. **Inventory Input**:
    - Select product and enter current inventory parameters.
    - Update stock levels, dispensing volumes, and supplier information.
2. **Demand Forecasting**:
    - AI model predicts future demand based on historical patterns.
    - View forecasts for different time horizons.
3. **Risk Analysis**:
    - System checks for expiry risks and stockout probabilities.
    - Provides alerts and recommendations.
4. **Decision Support**:
    - Get actionable insights for inventory management.
    - View performance metrics and optimization suggestions.


# Project Structure
```
Diabetes-Prediction/
├── README.md                 # Project documentation
├── main.py                   # Entry point for the Streamlit app
├── loader.py                 # Data loading and preprocessing
├── training.py               # Script for training the model
├── requirements.txt          # Project dependencies
├── LICENSE                   # License file
├── datasets/
│   ├── diabetes.csv          # Dataset used for training and predictions
├── models/
│   ├── diabetes_prediction_pipeline.joblib             # Trained machine learning model
├── images/
│   ├── page_icon.jpeg        # Application page icon
├── data/
│   ├── config.py             # Configuration variables
│   ├── base.py               # Static HTML/CSS content
├── function/
│   ├── model.py              # Custom model implementation
│   ├── function.py           # Utility functions
└── app/                      # Application logic and components
    ├── predict.py            # Prediction logic
    ├── explainer.py          # SHAP-based explanations
    ├── perm_importance.py    # Permutation importance analysis
    ├── performance.py        # Visualization of model performance metrics
    ├── input.py              # User input handling for predictions
    ├── about.py              # Informational section on diabetes
```


---

## Explanation Methods

1. **SHAP Waterfall Plot**:
   - Shows how each feature contributes positively or negatively to the prediction.
2. **SHAP Force Plot**:
   - Interactive visualization of feature contributions to individual predictions.
3. **Permutation Importance**:
   - Ranks features by their impact on the model's predictions.

---

## Model Performance

Performance metrics calculated:
- **Accuracy**: Percentage of correct predictions. (0.8571)
- **Precision**: Ratio of true positives to total positive predictions. (0.7692)
- **Recall**: Ratio of true positives to total actual positives. (0.8333)
- **F1 Score**: Harmonic mean of Precision and Recall. (0.8000)
- **ROC AUC**: Area under the ROC curve. (0.8904)

Metrics are displayed as donut charts in the application.

---

## Project Motivation

This project was developed to:
- Address real-world pharmacy inventory management challenges
- Demonstrate AI applications in healthcare operations
- Provide decision support tools for pharmacy administrators
- Reduce medication stockouts and expiry waste
- Optimize inventory costs while ensuring medication availability

---

## Contributing

Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a new feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push:
   ```bash
   git commit -m "Feature description"
   git push origin feature-name
   ```
4. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Contacts

If you have any questions or suggestions, please contact:
- Email: updulze29@gmail.com
- GitHub Issues: [Issues section](https://github.com/abdullahi682/Diabetes-Prediction/issues)
- GitHub Profile: [abdullahi682](https://github.com/abdullahi682/)
- Linkedin: [Abdullahi Ahmed](https://www.linkedin.com/in/AbdullahiAhm/)


### <i>Thank you for your interest in the project!</i>
