# Project Report: Credit Scoring Model

## 1. Introduction
Credit scoring is a classification problem in which financial information is used to estimate whether a person is likely to be creditworthy.

## 2. Objective
The objective of this project is to predict an individual's creditworthiness using financial-history features and evaluate the model using standard classification metrics.

## 3. Dataset
A synthetic dataset is generated inside the project so that the complete workflow can run without requiring an external download. The dataset contains financial-style attributes such as annual income, debt, payment history, credit utilization, savings, and employment history.

## 4. Feature Engineering
A `debt_to_income` feature is calculated as:

`total_debt / annual_income`

This feature represents the relationship between a person's debt and income.

## 5. Data Preprocessing
- The target column is separated from the input features.
- The dataset is split into 80% training data and 20% testing data.
- Feature values are standardized using `StandardScaler`.

## 6. Algorithm
Logistic Regression is used because the problem has two classes:
- 1 = Creditworthy
- 0 = Not Creditworthy

## 7. Model Evaluation
The project evaluates the trained model using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

The exact results produced by the current run are available in `results/model_metrics.csv`.

## 8. Output
The trained model predicts the creditworthiness class and also produces the probability that a sample belongs to the creditworthy class.

## 9. Conclusion
The project demonstrates a complete machine-learning classification pipeline including dataset creation, feature engineering, preprocessing, model training, prediction, and evaluation.

## 10. Limitation
The dataset is synthetic and is used only for educational demonstration. A production credit-scoring system would require real, legally obtained, carefully validated data along with fairness, compliance, explainability, and risk controls.
