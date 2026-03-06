# Customer-Churn-Prediction-Model
Predictive modeling and churn‑driver analysis using the Telco Customer Churn dataset from Kaggle (originally shared by the IBM community), covering exploratory analysis, feature engineering, model comparison, hyperparameter tuning, threshold optimization, and SHAP‑based interpretability.

Project Overview

This project builds and evaluates machine‑learning models to predict customer churn in the telecommunications industry. The goal is to identify high‑risk customers, understand the drivers behind churn, and provide actionable insights for retention strategies. The workflow includes exploratory analysis, preprocessing pipelines, model development, hyperparameter tuning, and SHAP‑based interpretability.

Dataset Overview

The dataset contains 7,043 customer records with demographic, service usage, contract, and billing information.
Key characteristics:
- Target variable: Churn (Yes/No)
- Churn rate: 26.5%
- Mix of categorical and numeric features
- Includes customer tenure, monthly charges, total charges, contract type, payment method, and service add‑ons
- Contains 11 missing values in TotalCharges
- CustomerID is non‑predictive and removed during preprocessing
This dataset is publicly available on Kaggle as Telco Customer Churn.

Exploratory Data Analysis
Churn Distribution
- Churn is moderately imbalanced (≈26% churners).
- Early‑lifecycle customers churn significantly more.
Demographic Patterns
- Gender shows no meaningful churn difference.
- Customers without partners or dependents churn more.
- Senior citizens churn slightly more than younger customers.
Service‑Related Patterns
- Customers without OnlineSecurity or TechSupport churn at much higher rates.
- Fiber optic users churn more than DSL users, likely due to higher monthly charges.
- Streaming services correlate with higher ARPU and slightly higher churn.
Churn Rate by Category
- Month‑to‑month contracts have the highest churn.
- Two‑year contracts have the lowest churn.
- Electronic check users churn more than any other payment method.
- Automatic payments (credit card, bank transfer) correlate with lower churn.
Numeric Feature Insights
- Tenure: Highest churn in first few months; drops sharply after 1 year.
- MonthlyCharges: Higher charges → higher churn (price sensitivity).
- TotalCharges: Low total charges → new customers → high churn; high total charges → loyal customers.

Data Preprocessing
- Removed CustomerID and corrected data types (e.g., TotalCharges → numeric).
- Imputed 11 missing TotalCharges values using mean imputation.
- Standardized numeric features using StandardScaler.
- One‑hot encoded categorical variables using OneHotEncoder.
- Built a unified Scikit‑Learn pipeline to ensure leak‑free preprocessing.
- Train/test split applied before any transformations.

Modeling and Evaluation
Multiple models were trained and compared:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Support Vector Machine (SVM)
  
Hyperparameter Tuning

Cross‑validated grid search identified the strongest models:
Best Logistic Regression
- C = 10, penalty = L2, solver = lbfgs
- CV ROC AUC: 0.846
  
Best XGBoost (Top Performer)
- learning_rate = 0.01
- max_depth = 3
- n_estimators = 400
- subsample = 0.7
- colsample_bytree = 0.7
- CV ROC AUC: 0.850
  
Final Model Performance (Test Set)
- ROC AUC: 0.847
- Recall at default threshold (0.5): 50%
- Recall at optimized threshold (0.25–0.30): 78–82%
  
Interpretation:
Lowering the threshold significantly increases churn capture, which is often preferred in retention scenarios where missing a churner is more costly than contacting a non‑churner.

Model Interpretability (SHAP)
SHAP analysis reveals the strongest churn drivers:
- Contract type (month‑to‑month is highest risk)
- Tenure (short tenure → high churn)
- MonthlyCharges (higher charges → higher churn)
- Payment method (electronic check → high churn)
- OnlineSecurity / TechSupport (absence increases churn risk)
- TotalCharges (proxy for customer lifetime value)
These insights align with business intuition and support targeted retention strategies.

Key Takeaways
- Churn is driven by early lifecycle, high monthly charges, and lack of value‑added services.
- Contract type and payment method are strong categorical predictors.
- XGBoost provides the best predictive performance with stable generalization.
- Threshold tuning is essential to maximize churn recall in real‑world applications.
- SHAP enables transparent, actionable insights for business decision‑making.

Technologies Used
- Python
- Pandas, NumPy
- Scikit‑Learn
- XGBoost, LightGBM
- SHAP
- Matplotlib, Seaborn

