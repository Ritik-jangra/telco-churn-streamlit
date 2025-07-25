# import Required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("Telco_Customer_Churn_Cleaned.csv")

# Feature Selection
y = df['Churn']
x = df.drop('Churn', axis=1)

# Train-Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature Scaling               -------> For SVM and Logistic Regression
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# Model Training

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train_scaled, y_train)
# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(x_train_scaled, y_train)
# Support Vector Machine
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(x_train_scaled, y_train)

# Model Testing

# Random Forest Prediction
rf_preds = rf_model.predict(x_test_scaled)
# Logistic Regression Prediction
lr_preds = lr_model.predict(x_test_scaled)
# SVM Predictions
svm_preds = svm_model.predict(x_test_scaled)

from sklearn.metrics import classification_report

# Random Forest
print("Random Forest")
print(classification_report(y_test, rf_preds))
print("-" * 40)

# 📊 Logistic Regression
print("Logistic Regression")
print(classification_report(y_test, lr_preds))
print("-" * 40)

# 📊 SVM
print("SVM")
print(classification_report(y_test, svm_preds))
print("-" * 40)


# Save Logistic Regression model
import joblib
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(x_train.columns.tolist(), 'model_columns.pkl')
