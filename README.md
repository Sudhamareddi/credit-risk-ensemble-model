 Quantitative Explainable Ensemble Model for Credit Risk Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-optimized-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP%20%7C%20LIME-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-Home%20Credit%20Kaggle-blue)

 Project Overview

A full end-to-end machine learning pipeline for credit risk prediction
using ensemble models, model explainability, and bias & stability
analysis — aligned with quantitative validation standards used in
real-world financial institutions.

 Objectives

- Build ensemble classification models for financial default prediction
- Evaluate models using industry-standard metrics
- Apply SHAP & LIME for model interpretability
- Perform bias and stability analysis across demographic groups
- Build an automated ML pipeline for reproducibility

 Dataset

- **Source:** [Kaggle — Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
- **Size:** 307,511 applicants × 122 features
- **Target:** Binary (1 = Default, 0 = Repaid)
- **Class Imbalance:** ~92% repaid, ~8% defaulted

Project StructureExploratory Data Analysis — Home Credit Default Risk
credit-risk-ensemble-model/
│
├── notebooks/
│   ├── 1_EDA.ipynb
│   ├── 2_Preprocessing.ipynb
│   ├── 3_Model_Training.ipynb
│   ├── 4_Explainability.ipynb
│   ├── 5_Bias_Stability.ipynb
│   └── 6_Automated_Pipeline.ipynb
│ README.nd

---

## Pipeline Stages

1. Exploratory Data Analysis
- Distribution analysis of 122 features
- Missing value analysis
- Class imbalance visualization
- Correlation heatmap
- Borrower behavior pattern analysis

2. Preprocessing
- Median imputation for numerical features
- Mode imputation for categorical features
- Label encoding for categorical variables
- Feature engineering:
  - Age in years
  - Employment years
  - Credit to income ratio
  - Annuity to income ratio
- Standard scaling

3. Model Training
- Logistic Regression (baseline)
- Random Forest (balanced class weights)
- XGBoost (scale_pos_weight for imbalance)
- Ensemble (average of all three)

4. Explainability
- SHAP TreeExplainer for global feature importance
- SHAP Waterfall plot for individual predictions
- LIME for local model interpretability
- Cross-method validation between SHAP & LIME

5. Bias & Stability Analysis
- Gender bias analysis
- Age group bias analysis
- KS Stability test
- 5-Fold Cross Validation stability check

6. Automated Pipeline
- End-to-end pipeline with config-driven settings
- Automated reporting

Results

Model Performance

| Model | ROC-AUC | F1 Score |
|---|---|---|
| Logistic Regression | 0.7469 | 0.2596 |
| Random Forest | 0.7371 | 0.2602 |
| **XGBoost** | **0.7577** | **0.2726** |
| Ensemble | 0.7563 | — |

Stability
| Metric | Value | Status |
|---|---|---|
| KS Statistic | 0.3832 | GOOD |
| CV Mean AUC | 0.7426 | Stable |
| CV Std Dev | 0.0044 | Very Stable  |

### Top Risk Predictors (SHAP)

| Feature | SHAP Importance | Meaning |
|---|---|---|
| EXT_SOURCE_3 | 0.3865 | External risk score |
| EXT_SOURCE_2 | 0.3479 | External risk score |
| AMT_GOODS_PRICE | 0.1676 | Goods price amount |

> SHAP & LIME both identify EXT_SOURCE_3 as the #1 predictor
> — cross method validation confirmed 

Known Limitations

- Model overpredicts defaults due to class imbalance
- Gender & age bias detected — requires monitoring
- External scores dominate predictions — limited
  interpretability beyond top features

Libraries Used

| Library | Purpose |

| Pandas, NumPy | Data manipulation |
| Scikit-learn | Preprocessing & modeling |
| XGBoost | Gradient boosting |
| SHAP | Global explainability |
| LIME | Local explainability |
| Matplotlib, Seaborn | Visualization |
| SciPy | KS statistical test |

How to Run

1. Open [Kaggle](https://kaggle.com) and join the
   Home Credit Default Risk competition
2. Run notebooks **in order** as mentioned in project structure
3. Each notebook saves outputs used by the next



**G.Sudhama**
[GitHub](https://github.com/Sudhamareddi) |
[LinkedIn](www.linkedin.com/in/sudhama-g)
