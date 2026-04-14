Exploratory Data Analysis — Home Credit Default Risk

This project focuses on Explatory Data Analysis (EDA) of the Home Credit Default Risk dataset from Kaggle. The objective is to understand the underlying data patterns, identify key factors influencing loan default, and prepare insights that can guide feature engineering and model building.

Financial institutions often struggle with assessing credit risk due to limited or noisy data. Through this EDA, we aim to uncover meaningful relationships between customer attributes and repayment behavior.

Objectives
Understand the structure and distribution of the dataset
Identify missing values and data quality issues
Analyze relationships between features and target variable (TARGET)
Detect patterns, trends, and anomalies in borrower behavior
Generate insights useful for downstream machine learning models
Dataset Description

The dataset is sourced from Kaggle’s Home Credit Default Risk competition.

Key Features:
TARGET: Binary variable (1 = default, 0 = repaid)
Demographics: Age, gender, family status
Financial Info: Income, credit amount, annuity
Employment: Employment duration, occupation
External Scores: Risk scores from external sources

Tools & Libraries Used
Python 
Pandas
NumPy
Matplotlib
Seaborn


EDA Workflow
1. Data Loading & Inspection
Loaded dataset using Pandas
Checked shape, column types, and basic statistics
Identified numerical and categorical features
2. Missing Value Analysis
Calculated percentage of missing values per column
Visualized missing data distribution
Observed that several features have high missingness (common in financial datasets)
3. Target Variable Analysis
Analyzed class imbalance in TARGET
Observed that:
Majority of applicants repay loans
Minority class represents defaults → imbalanced dataset
4. Univariate Analysis
Distribution plots for:
Income
Credit amount
Age
Identified skewness and outliers in financial variables
5. Bivariate Analysis
Compared features against TARGET
Key observations:
Lower income groups show higher default probability
Higher credit amounts correlate with risk in certain segments
External risk scores strongly influence repayment behavior
6. Categorical Feature Analysis
Analyzed:
Gender vs default
Occupation vs default
Family status vs default
Used count plots and bar charts for comparison
7. Correlation Analysis
Generated correlation heatmap
Identified features with strong positive/negative correlation with TARGET
Helped shortlist important variables for modeling
8. Outlier Detection
Used boxplots to detect extreme values
Observed significant outliers in income and credit-related fields

Key Insights
Dataset is highly imbalanced, requiring special handling in modeling
External risk scores are strong predictors of default
Income and employment stability significantly influence repayment
Certain demographic groups show higher default tendencies
Missing values are substantial and must be handled carefully

Challenges
High number of missing values
Imbalanced target variable
Presence of outliers
Large dataset size impacting performance

Next Steps
Handle missing values (imputation strategies)
Perform feature engineering
Apply scaling/encoding techniques
Train machine learning models:
Logistic Regression
Random Forest
XGBoost / LightGBM
Evaluate using metrics like:
ROC-AUC
Precision-Recall
F1-score
