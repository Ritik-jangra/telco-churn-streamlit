# 📊 Telco Customer Churn Prediction App

This is a machine learning web app that predicts whether a telecom customer is likely to **churn** (i.e., stop using the service) based on their contract details, usage behavior, and service preferences. Built with **Streamlit** and powered by a **Random Forest Classifier**, it also uses **SHAP (SHapley Additive Explanations)** to visually explain each prediction.

---

## 🔗 Live Demo

🌐 **Streamlit App**: [Click to Try](https://telco-churn-app-htpapcby5ro6gkvr3c52ff.streamlit.app/)  
📁 **GitHub Repo**: [View on GitHub](https://github.com/Ritik-jangra/telco-churn-streamlit)

---

## 🎯 Objective

To help telecom companies **predict customer churn** in advance so they can retain valuable customers and reduce business loss. The app provides:
- Real-time churn prediction
- Explainability using SHAP
- Simple, clean interface for any user

---

## 🔧 Tech Stack

| Category        | Tools Used                                   |
|----------------|-----------------------------------------------|
| ML Model       | `RandomForestClassifier` (scikit-learn)       |
| Data Handling  | `Pandas`, `NumPy`                             |
| Model Scaling  | `StandardScaler` (scikit-learn)               |
| Explainability | `SHAP` (SHapley Additive Explanations)        |
| Frontend UI    | `Streamlit` (Python-based UI framework)       |
| Deployment     | `Streamlit Cloud`, `GitHub`                   |
| Model Storage  | `joblib`                                      |

---

## 📋 Features

✅ Predicts customer churn (Yes/No)  
✅ Probability score of churn  
✅ SHAP-based visual explanation for trust & transparency  
✅ Dropdowns and numeric fields for easy input  
✅ Fully deployed on Streamlit Cloud

---

## 🧠 SHAP Explanation (Why This Prediction?)

The app generates a **SHAP Waterfall Plot** to explain how each input feature impacts the final prediction.

### 🔍 Example:
![SHAP Waterfall Plot](https://github.com/Ritik-jangra/telco-churn-streamlit/blob/main/assets/shap_plot.png)

- 🔴 Red bars increase the chance of churn  
- 🔵 Blue bars reduce the chance of churn  
- Features are ordered by impact (top = most important)  
- Value of `f(x)` = Final prediction score (here: 0.48)

---

## 📈 Model Performance

| Metric         | Score |
|----------------|-------|
| Accuracy       | 79%   |
| Precision (0)  | 83%   |
| Precision (1)  | 65%   |
| Recall (0)     | 91%   |
| Recall (1)     | 47%   |
| F1-Score (1)   | 54%   |

> The model performs well in identifying both churn and non-churn classes.

---

## 📊 Dataset Info

- 📁 Dataset: [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- 🧾 Records: ~7,000 customers  
- 🎯 Target column: `Churn`  
- 🧹 Cleaned version used for training/testing

---

## 📂 Folder Structure

telco-churn-streamlit/
│
├── app.py # Streamlit main app
├── rf_model.pkl # Trained ML model
├── scaler.pkl # StandardScaler object
├── model_columns.pkl # Columns used during training
├── requirements.txt # Python dependencies
├── README.md # This file
└── assets/
└── shap_plot.png # Sample SHAP plot

👨‍💻 Author
Ritik Jangra
📧 Email: ritikjangra87@gmail.com
🌐 GitHub: @Ritik-jangra