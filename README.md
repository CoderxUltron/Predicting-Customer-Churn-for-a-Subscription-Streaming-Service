# Predicting-Customer-Churn-for-a-Subscription-Streaming-Service

# Customer Churn Prediction for Subscription Services

[![View Notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/Customer_Churn_Prediction.ipynb)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)](https://scikit-learn.org/)

An end-to-end machine learning pipeline built to predict customer churn for a subscription streaming service. The project evaluates multiple classification algorithms utilizing robust feature selection voting techniques, data leakage-proof pipelines, and randomized hyperparameter tuning.

---

## 👤 Author Information
* **Name:** Ayush Jha
* **Registration No:** 23BCE10592
* **Institution:** VIT Bhopal University
* **Email:** ayush.23bce10592@vitbhopal.ac.in

---

## 📌 Project Overview & Domain
* **Domain:** Media, Subscription Analytics & Business Intelligence
* **Primary Objective:** Identify high-risk subscribers before cancellation to optimize proactive marketing retention campaigns.
* **Core Methodologies:** Correlation Filtering, Recursive Feature Elimination (RFECV), Stratified $k$-Fold Cross-Validation, Hyperparameter Optimization, and Model Explainability via Gini Feature Importance.

---

## 📊 Dataset & Scenario Mapping
This project utilizes the benchmark **Telco Customer Churn (IBM Sample Data Set)**. The attributes map seamlessly to a modern streaming infrastructure scenario:

| Original Feature | Streaming Service Equivalent |
| :--- | :--- |
| `tenure` | Account / Subscription Age |
| `MonthlyCharges` / `TotalCharges` | Billing history metrics |
| `Contract` / `PaymentMethod` | Billing frequency and transaction type |
| `OnlineSecurity` / `TechSupport` / `StreamingTV` | Engagement features & Add-on service usage signals |
| `Gender` / `SeniorCitizen` / `Dependents` | Subscriber Demographic Data |
| `Churn` | **Target Label** (Binary: Cancelled / Active) |

* **Data URL:** Direct stream from [IBM Data Repository](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) (Fully reproducible without API keys).

---

## 🛠️ Tech Stack & Dependencies
The notebook runs entirely within a standard Python data science environment:
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning Stack:** `scikit-learn` (Pipelines, ColumnTransformer, Feature Selection, and Classifiers)

---

## 🚀 Key Pipeline Implementations

### 1. Exploratory Data Analysis (EDA) Insights
* Detected a class imbalance (~26.5% overall churn rate), justifying the transition from standard *Accuracy* to *F1-Score* and *ROC-AUC* as key optimization targets.
* Handled structural missing data inside `TotalCharges` dynamically via localized median imputation.
* Identified heavy correlation signals linking short-tenure profiles and Month-to-Month contracts directly to high churn rates.

### 2. Multi-Criteria Feature Selection (Voting Strategy)
To prevent algorithmic bias, three separate feature selection techniques voted on the optimal subset:
1. **Correlation Filtering:** Dropped redundant dummy variables exhibiting absolute multi-collinearity ($> 0.90$).
2. **RFECV:** Wrapped a shallow Decision Tree to select the optimal feature volume maximizing cross-validated F1 score.
3. **Random Forest Gini Importance:** Screened out low-signal variables by retaining the top 90% cumulative feature importance.
* *Result:* The initial 40 encoded features were robustly reduced to a consolidated **24 consensus features**, eliminating compute cost without losing predictive performance.

### 3. Rigorous Cross-Validation Framework
All estimators were assessed inside a custom `Pipeline` structure utilizing `StratifiedKFold(n_splits=5)`. Feature scaling and categorical encodings were dynamically computed inside each individual fold, enforcing strict isolation and preventing **data leakage** from validation sets.

---

## 📈 Model Performance Matrix

Evaluation of untuned baselines versus the optimized pipeline on the independent, held-out validation test split ($20\%$):

| Model Topology | Feature Set | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Tuned)** | **Reduced (24)** | **80.7%** | **0.682** | **0.516** | **0.588** | **0.843** |
| SVM (RBF, Untuned) | Reduced (24) | 79.7% | 0.652 | 0.510 | 0.572 | 0.793 |
| SVM (RBF, Untuned) | Full (40) | 79.6% | 0.651 | 0.500 | 0.566 | 0.794 |
| Random Forest (Untuned)| Full (40) | 78.4% | 0.620 | 0.492 | 0.548 | 0.821 |
| Decision Tree (Untuned) | Full (40) | 77.4% | 0.587 | 0.505 | 0.543 | 0.795 |

### 🏆 Final Model Selection Justification
The **Tuned Random Forest Model** on the reduced consensus feature set was chosen for production deployment because it:
1. Achieved the highest global performance across both threshold-dependent and threshold-free criteria ($F1 = 0.588$, $AUC = 0.843$).
2. Executes inference and re-training significantly faster than RBF-Kernel SVM architectures.
3. Provides explicit, transparent feature importance evaluations directly interpretable by strategic marketing teams.

---

## 💼 Core Business Recommendations
1. **First-Quarter Proactive Save Campaigns:** Prioritize automated, high-incentive upgrades targeting subscribers on Month-to-Month payment schedules during their first 90 days.
2. **Value-Added Service Bundling:** Push strategic promotions for technical add-on features (e.g., Online Security, Tech Support) to unengaged profiles, as data implies these significantly extend subscriber lifetime value.
