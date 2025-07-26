# Import libraries (already perfect)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("Telco_Customer_Churn_Cleaned.csv")

# Split features and target
y = df['Churn']
x = df.drop('Churn', axis=1)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature scaling for LR & SVM only
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# --- Model Training ---

# Random Forest (unscaled features)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)

# Logistic Regression (scaled)
lr_model = LogisticRegression(random_state=42)
lr_model.fit(x_train_scaled, y_train)

# SVM (scaled)
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(x_train_scaled, y_train)

# --- Evaluation ---
print("Random Forest")
print(classification_report(y_test, rf_model.predict(x_test)))
print("-" * 40)

print("Logistic Regression")
print(classification_report(y_test, lr_model.predict(x_test_scaled)))
print("-" * 40)

print("SVM")
print(classification_report(y_test, svm_model.predict(x_test_scaled)))
print("-" * 40)

# --- Save models and metadata ---
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(x.columns.tolist(), 'model_columns.pkl')
