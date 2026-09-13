# Task 1 - Credit Scoring Model

## CodeAlpha Machine Learning Internship

### Objective
Build a machine learning model that predicts whether an individual is creditworthy using financial-history features.

### Features used
- Age
- Annual income
- Total debt
- Credit history length
- Late payments
- Credit utilization
- Existing loans
- Savings
- Employment years
- Debt-to-income ratio

### Model
This project uses **Logistic Regression** for binary classification.

### Evaluation metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

### Dataset
This project is self-contained. A synthetic financial dataset is generated for demonstration and learning purposes.  
**Important:** It is not real customer data and should not be used for real lending decisions.

### Project structure
```text
Task1_Credit_Scoring_Model/
├── credit_scoring_model.py
├── credit_data.csv
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── LINKEDIN_POST.md
├── VIDEO_SCRIPT.md
└── results/
    ├── model_metrics.csv
    ├── classification_report.txt
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── feature_coefficients.csv
    └── sample_prediction.txt
```

### How to run
```bash
pip install -r requirements.txt
python credit_scoring_model.py
```

### Workflow
1. Generate/load the financial dataset.
2. Separate features and target.
3. Split data into training and testing sets.
4. Standardize numeric features.
5. Train Logistic Regression.
6. Predict creditworthiness.
7. Evaluate the model.
8. Save metrics and visualizations in the `results` folder.

### Target
- `1` = Creditworthy
- `0` = Not Creditworthy

### Disclaimer
This is an educational internship project using synthetic data. It is not intended for real-world credit or lending decisions.
