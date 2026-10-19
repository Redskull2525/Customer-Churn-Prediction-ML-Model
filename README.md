# 📊 Customer Churn Prediction System

An end-to-end machine learning project that predicts whether a telecom customer will churn (leave the service). It uses a scikit-learn **Pipeline** (preprocessing + **Random Forest**), tunes the model with **RandomizedSearchCV**, and serves predictions through a **Streamlit** web app.

---

## 🎯 Business problem

Keeping an existing customer costs less than winning a new one. If the company can flag customers who are likely to leave, the retention team can contact them first, for example with a discount or a plan change.

---

## 📘 Dataset

- **Source:** BigML Telecom Churn dataset, a public dataset of US telecom customers
- **Size:** 3,333 customers. The dataset comes pre-split into `churn-bigml-80.csv` (2,666 rows, used for training) and `churn-bigml-20.csv` (667 rows, held-out test).
- **Columns:** 19 features plus the target `Churn` (True/False)
- **Class balance:** about 14% of customers churn, so the data is **imbalanced**. Accuracy alone is therefore misleading: a model that always predicts "no churn" already scores ~86%.

See [`data/data_dictionary.md`](data/data_dictionary.md) for column descriptions.

---

## 🧠 Approach

1. Split the columns into numeric (account length, minutes, calls, charges, service calls) and categorical (State, International plan, Voice mail plan).
2. Preprocess them with a `ColumnTransformer`: `StandardScaler` on the numeric columns and `OneHotEncoder` on the categorical ones.
3. Build a `Pipeline` of the preprocessing followed by a `RandomForestClassifier`. Keeping both in one pipeline means the app applies exactly the same transformations at prediction time.
4. Tune with `RandomizedSearchCV`: 8 candidates, 3-fold cross-validation, optimising **F1 for the churn class**.
5. Evaluate once on the untouched 20% test set.
6. Save the fitted pipeline with `joblib`. The Streamlit app loads it.

---

## 📈 Results (held-out test set, 667 customers)

| Metric | Value |
|---|---|
| Accuracy | 0.94 |
| Churn precision | 0.97 |
| Churn recall | 0.61 |
| Churn F1 | 0.75 |
| ROC-AUC | 0.93 |
| PR-AUC (average precision) | 0.87 |

Confusion matrix: `[[570, 2], [37, 58]]`. Of the 95 customers who actually churned, the model caught 58 and missed 37. When the model does flag a customer, it is almost always right.

**What this means:** the model is precise but conservative. To catch more churners, lower the decision threshold (for example from 0.5 to 0.3). That raises recall at the cost of more false alarms.

---

## ⚠️ Limitations & next steps

- Recall (0.61) is the weak point. Next steps are threshold tuning, `class_weight="balanced"`, and comparing against Logistic Regression or Gradient Boosting.
- The dataset is small and old, so the results may not transfer to a different telecom company.
- Feature importance and explainability (e.g. SHAP) are not included yet.

---

## 📂 Project structure

```
├── app/app.py                 # Streamlit app
├── data/
│   ├── churn-bigml-80.csv     # training data
│   ├── churn-bigml-20.csv     # test data
│   └── data_dictionary.md
├── models/
│   ├── churn_pipeline.pkl     # preprocessing + Random Forest
│   ├── feature_cols.pkl, num_cols.pkl, cat_cols.pkl
│   └── feature_names_after_ohe.pkl
├── notebook/Customer_Churn_Prediction_System_Modified_Final.ipynb
└── requirements.txt
```

---

## ▶️ Run it

```bash
pip install -r requirements.txt
jupyter notebook notebook/Customer_Churn_Prediction_System_Modified_Final.ipynb   # retrain (optional)
streamlit run app/app.py
```

---

## 👨‍💻 Author

**Abhishek Shelke** · [GitHub](https://github.com/Redskull2525) · [LinkedIn](https://www.linkedin.com/in/abhishek-s-b98895249)
