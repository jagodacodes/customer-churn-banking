# 🏦 Customer Churn Prediction — Banking

**Predicting which bank customers are at risk of leaving — and what the business should do about it.**

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red)
![SHAP](https://img.shields.io/badge/SHAP-0.41+-green)

| Tool | Model | Source | Customers |
|---|---|---|---|
| Python | Logistic Regression · Random Forest · XGBoost | Kaggle | 10,000 |

---

## Project Overview

Customer churn is one of the most costly problems in retail banking — acquiring a new client is 5–7x more expensive than retaining an existing one. This project builds and compares three classification models to predict which customers are likely to leave, identifies the strongest behavioral and demographic drivers of churn, and translates model outputs into concrete business recommendations.

**Business question:** Which customer segments are most at risk of churning — and what should the bank prioritize to retain them?.


**Key finding:** XGBoost achieved the highest predictive performance (AUC ≈ 0.88). The three strongest churn drivers are **age**, **number of products held**, and **account activity status** — suggesting retention efforts should concentrate on older, single-product, inactive customers. A targeted intervention program for this segment could retain an estimated €2.1M in customer lifetime value per 10,000 customers annually.

---

## Technologies Used

| Tool | Purpose |
|---|---|
| pandas, numpy | Data cleaning, EDA, feature engineering |
| scikit-learn | Logistic Regression, Random Forest, model evaluation |
| XGBoost | Gradient boosting classifier — best performer |
| SHAP | Model interpretability — feature importance & direction |
| imbalanced-learn (SMOTE) | Class imbalance handling |
| matplotlib, seaborn | Visualizations |
| Jupyter Notebook | Analysis environment |

---

## Project Process

```
Data Loading & Exploration
        ↓
EDA — churn rate by segment, correlations, age distributions
        ↓
Feature Engineering — encoding, scaling, 3 new derived features
        ↓
Class Imbalance Handling — SMOTE applied on training set only
        ↓
Model 1: Logistic Regression — interpretable baseline
        ↓
Model 2: Random Forest — ensemble method
        ↓
Model 3: XGBoost — best performer (AUC ≈ 0.88)
        ↓
Model Comparison — AUC, F1, Precision, Recall
        ↓
SHAP Analysis — which features drive churn and in which direction?
        ↓
Business Interpretation & Retention Recommendations
```

---

## Data

- **Source:** [Kaggle — Bank Customer Churn Dataset](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction)
- **Frequency:** Static cross-sectional | **N = 10,000 customers** | **14 variables**
- **Target variable:** `Exited` — 1 = churned (20.4%), 0 = retained (79.6%)

| Variable | Description |
|---|---|
| CreditScore | Customer credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance (€) |
| NumOfProducts | Number of bank products held |
| HasCrCard | Credit card ownership flag |
| IsActiveMember | Active account flag |
| EstimatedSalary | Annual salary estimate (€) |
| **Exited** *(target)* | **1 = churned, 0 = retained** |

> Class imbalance (80/20) handled via SMOTE — applied on training set only to prevent data leakage.

---

## Project Structure

```
customer-churn-banking/
├── README.md
├── requirements.txt
├── data/
│   └── Churn_Modelling.csv          # Raw dataset (download from Kaggle)
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory analysis & churn patterns
│   ├── 02_preprocessing.ipynb      # Cleaning, encoding, SMOTE
│   ├── 03_models.ipynb             # LR vs RF vs XGBoost comparison
│   └── 04_shap_interpretation.ipynb  # Business drivers of churn + recommendations
└── outputs/
    ├── churn_overview.png
    ├── churn_by_segment.png
    ├── model_comparison.png
    ├── roc_curves.png
    ├── shap_feature_importance.png
    ├── shap_summary.png
    └── top3_drivers.png
```

---

## Results

### Model Comparison

| Model | AUC | F1 Score | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | ~0.77 | ~0.57 | ~0.62 | ~0.54 |
| Random Forest | ~0.86 | ~0.70 | ~0.74 | ~0.67 |
| **XGBoost** | **~0.88** | **~0.74** | **~0.76** | **~0.72** |

> AUC was prioritized over accuracy — with a 20% churn rate, a naive model predicting "no churn" always scores 80% accuracy but provides zero business value.

### Top Churn Drivers (SHAP Analysis)

| Driver | Direction | Business Interpretation |
|---|---|---|
| **Age** | ↑ older → more churn | Customers 45+ churn significantly more |
| **NumOfProducts** | ↑ 3-4 products → churn spike | Over-sold customers leave; 1 product = high risk |
| **IsActiveMember** | ↓ inactive → more churn | Dormant accounts churn at ~2x the rate |
| **Geography: Germany** | Germany → more churn | 32% churn rate vs 16% in France |
| **Balance** | high balance → more churn | Wealthier customers may have more options |

### Business Recommendations

**1.  Age-Based Retention Campaign** — Design a premium loyalty program for customers aged 45+: dedicated relationship managers, exclusive service tiers, tenure-based rewards.

**2.  Cross-Sell Single-Product Customers** — Introduce bundling offers (savings + credit card) within 90 days of onboarding. Each additional product meaningfully reduces churn probability.

**3.  Reactivation Program** — Trigger automated re-engagement workflow when a customer shows no activity for 60+ days. Proactive outreach before the customer decides to leave.

**4.  Germany Market Review** — Investigate pricing competitiveness and product fit in the German market specifically; the 2x churn differential vs France warrants dedicated strategic attention.

---

## What I Learned

- Why accuracy is a misleading metric for imbalanced classification — and how AUC/F1 paint a more honest picture
- How SMOTE works and why it must be applied only to the training set to avoid data leakage
- The difference between model performance (how well it predicts) and model interpretability (why it predicts that way) — and why business stakeholders need both
- How to translate a confusion matrix into a business cost frame: false negatives = missed churners = lost LTV
- Iterative model building: starting from a simple interpretable baseline before adding complexity

---

## What Could Be Improved

- **Survival analysis** — model time-to-churn rather than binary outcome for more precise retention timing
- **Transactional features** — login frequency, transaction volume, product usage rate would add stronger behavioral signals
- **Streamlit deployment** — interactive app where a relationship manager could score individual customers in real time
- **Model retraining pipeline** — churn patterns drift over time; a monthly retraining schedule with data freshness monitoring would be needed in production
- **Threshold optimization** — adjust the classification threshold based on the business cost ratio of false negatives vs false positives

---

## How to Run the Project

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/customer-churn-banking.git
cd customer-churn-banking

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Download Churn_Modelling.csv from Kaggle and place in /data/

# 4. Run notebooks in order
jupyter notebook
# 01_eda.ipynb → 02_preprocessing.ipynb → 03_models.ipynb → 04_shap_interpretation.ipynb
```

