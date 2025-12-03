# 📊 Customer Churn Prediction System

A complete end-to-end machine learning project that predicts whether a telecom customer will churn using a **Random Forest Classifier** with hyperparameter tuning, preprocessing pipelines, feature engineering, evaluation metrics, and a fully functional **Streamlit web application** for real-time predictions.

---

## 🚀 Project Overview

This project analyzes telecom customer behavior and builds a predictive machine learning system to identify customers at risk of leaving the service ("churn").
The workflow includes:

* Data loading (80–20 train–test split)
* Data cleaning and preprocessing
* Label encoding & feature scaling
* Model training using **Random Forest**
* Hyperparameter tuning using **RandomizedSearchCV**
* Evaluation using ROC, Precision-Recall curve & Confusion Matrix
* Model saving using **Joblib**
* Deployment using **Streamlit**

---

## 📂 Folder Structure

```
customer-churn-prediction/
│── data/
│   ├── churn-bigml-80.csv
│   ├── churn-bigml-20.csv
│   ├── data_dictionary.md
│
│── models/
│   ├── best_churn_model.pkl
│   ├── churn_scaler.pkl
│
│── notebook/
│   ├── Customer_Churn_Prediction_System_Modified.ipynb
│
│── app/
│   ├── app.py
│
│── README.md
│── requirements.txt
```

---

## 📘 Dataset Description

The project uses the **BigML Telecom Churn Dataset**, split into:

* **80% training dataset:** `churn-bigml-80.csv`
* **20% testing dataset:** `churn-bigml-20.csv`

Total Columns: **22**
Target Column: **Churn (Yes/No)**

The dataset includes:

* Customer demographics
* Subscription plans
* Call usage analytics
* International call charges
* Day/Eve/Night usage metrics
* Customer service interactions

---

## 📖 Data Dictionary

A PDF version is included in the `data/` folder.

### **1. Customer Information**

| Column    | Description           |
| --------- | --------------------- |
| State     | US State              |
| Area code | Telephone area code   |
| Phone     | Customer phone number |

### **2. Account Behavior**

| Column                | Description          |
| --------------------- | -------------------- |
| Account length        | Days customer stayed |
| International plan    | Yes/No               |
| Voice mail plan       | Yes/No               |
| Number vmail messages | Count of messages    |

### **3. Usage Metrics**

| Category            | Details                |
| ------------------- | ---------------------- |
| Day Usage           | Minutes, calls, charge |
| Evening Usage       | Minutes, calls, charge |
| Night Usage         | Minutes, calls, charge |
| International Usage | Minutes, calls, charge |

### **4. Customer Service**

| Column                 | Description                |
| ---------------------- | -------------------------- |
| Customer service calls | Number of calls to support |

### **5. Target**

| Column | Description                    |
| ------ | ------------------------------ |
| Churn  | Whether customer left (Yes/No) |

---

## 🧠 Machine Learning Workflow

1. Load and inspect dataset
2. Handle missing values
3. Encode categorical features using **LabelEncoder**
4. Scale numerical features using **StandardScaler**
5. Build baseline Random Forest model
6. Optimize using **RandomizedSearchCV**
7. Plot performance metrics:

   * ROC Curve
   * PR Curve
   * Confusion Matrix
8. Save final model and scaler:

   * `best_churn_model.pkl`
   * `churn_scaler.pkl`

---

## 🌐 Streamlit App

The project includes a fully functional Streamlit UI for interactive predictions.

### ▶️ Run the app:

```bash
streamlit run app/app.py
```

The app allows users to:

* Enter customer details manually
* View predicted label (Churn Yes/No)
* View model churn probability
* Display confidence visually

---

## 🛠 Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Typical dependencies include:

```
streamlit
pandas
numpy
scikit-learn
matplotlib
joblib
```

(Your `requirements.txt` will list exact versions.)

---

## 🔮 Future Improvements

* Add SHAP explainability for feature importance
* Deploy web app using Streamlit Cloud / Render
* Add API using FastAPI
* Automate retraining pipeline
* Add dashboard with churn analytics

---

## 👨‍💻 Author

**Abhishek Shelke**

* GitHub: [https://github.com/Redskull2525](https://github.com/Redskull2525)
* LinkedIn: [https://www.linkedin.com/in/abhishek-s-b98895249](https://www.linkedin.com/in/abhishek-s-b98895249)

---
