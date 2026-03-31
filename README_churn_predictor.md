# 📉 Customer Churn Predictor

> Predicts which customers are likely to leave using 4 ML models — with risk scoring and retention recommendations.

!\[Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square\&logo=python)
!\[XGBoost](https://img.shields.io/badge/XGBoost-ML-orange?style=flat-square)
!\[scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green?style=flat-square)
!\[Accuracy](https://img.shields.io/badge/ROC--AUC-0.85%2B-brightgreen?style=flat-square)
!\[Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square)

\---

## 📌 Overview

A machine learning system that predicts customer churn — identifying which customers are likely to cancel their subscription before they actually leave. Trains and compares 4 models, handles class imbalance using SMOTE, assigns every customer a risk score (High/Medium/Low), and recommends specific retention actions for each risk level.

\---

\---

## ✨ Features

* **Exploratory Data Analysis** — 5 charts revealing why customers churn
* **4 ML Models** — Logistic Regression, Random Forest, XGBoost, Gradient Boosting
* **SMOTE Balancing** — Synthetic oversampling to fix class imbalance
* **ROC Curves** — Visual comparison of all 4 models
* **Confusion Matrix** — Detailed breakdown of predictions vs actuals
* **Feature Importance** — Top 15 factors driving churn
* **Risk Scoring** — Every customer labeled High / Medium / Low risk
* **Retention Actions** — Specific recommendations per risk level
* **Single Customer Predictor** — Adjust sliders to predict any customer's churn risk

\---

## 🧠 Key Concepts

|Concept|Explanation|
|-|-|
|**Churn**|When a customer cancels/stops using a service|
|**Class Imbalance**|Churned customers are rare (\~26%) — models get biased without fixing this|
|**SMOTE**|Creates synthetic churned customer examples to balance training data|
|**ROC-AUC**|Area under ROC curve — measures model's ability to distinguish churners from non-churners|
|**Feature Importance**|Which variables (contract type, tenure, charges) most strongly predict churn|
|**Threshold**|Probability cutoff above which a customer is flagged as likely to churn|

\---

## 🛠️ Tech Stack

|Tool|Purpose|
|-|-|
|`Python`|Core programming language|
|`XGBoost`|Gradient boosting model|
|`scikit-learn`|ML models and evaluation|
|`imbalanced-learn`|SMOTE for class balancing|
|`Streamlit`|Interactive dashboard|
|`Plotly`|ROC curves and charts|
|`Pandas`|Data preprocessing|

\---

## 🚀 Getting Started

### 1\. Clone the Repository

```bash
git clone https://github.com/YOUR\_USERNAME/customer-churn-predictor.git
cd customer-churn-predictor
```

### 2\. Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly matplotlib imbalanced-learn
```

### 3\. Run the App

```bash
streamlit run churn\_predictor.py
```

> ✅ Works immediately with built-in sample data — no dataset needed!

\---

## 📂 Dataset

To use real data, download the IBM Telco Customer Churn dataset:
👉 [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Upload via the sidebar in the running app.

\---

## 📊 Model Results

|Model|ROC-AUC|Accuracy|
|-|-|-|
|Logistic Regression|\~0.82|\~80%|
|Random Forest|\~0.84|\~81%|
|XGBoost|\~0.86|\~82%|
|Gradient Boosting|\~0.85|\~81%|

\---

## 📁 Project Structure

```
customer-churn-predictor/
├── churn\_predictor.py    # Main Streamlit app
├── preview.png           # Dashboard screenshot
└── README.md             # This file
```

\---

## 🔮 Future Improvements

* \[ ] Add SHAP explainability for individual predictions
* \[ ] Add survival analysis (time-to-churn modeling)
* \[ ] Connect to live CRM database
* \[ ] Build automated retention email trigger
* \[ ] Add A/B test simulation for retention offers

\---

## 👤 Author

**Syed Abideen** · [GitHub](https://github.com/your-username)

