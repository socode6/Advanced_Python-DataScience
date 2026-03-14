# 🎓 Student Dropout Early Warning System

A full end-to-end machine learning pipeline that predicts student dropout risk and generates an **actionable risk report** for educators — built with explainable AI so predictions are transparent and trustworthy.

---

## 📌 Project Overview

Student dropout is one of the most costly and preventable problems in higher education. This project moves beyond simple classification — it builds a system that:

1. **Learns** patterns from students with known outcomes (Graduated or Dropped Out)
2. **Predicts** dropout probability for currently enrolled students
3. **Explains** why each student is flagged as at-risk
4. **Generates** a human-readable risk report with intervention priority levels

> The core insight: enrolled students are not training data — they are the real prediction targets.

---

## 📊 Results

### Model Comparison

| Model | Accuracy | Precision (Dropout) | Recall (Dropout) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 91.32% | 86.71% | 91.90% | 97.35% |
| Random Forest | 91.74% | 91.18% | 87.32% | 96.82% |
| XGBoost | 92.56% | 91.67% | 89.08% | 97.20% |
| **Stacking Meta-Learner** | **92.98%** | **90.00%** | **93.00%** | **97.43%** |

> **Why Recall matters most here:** Missing an at-risk student is far more costly than a false alarm. The Meta-Learner achieves the best balance — highest overall AUC and strong Recall for the Dropout class.

### Risk Report Output (Sample)

| Student ID | Dropout Probability | Risk Status |
|---|---|---|
| 553 | 98.83% | 🔴 CRITICAL — Intervene Now |
| 32 | 98.78% | 🔴 CRITICAL — Intervene Now |
| 605 | 2.49% | 🟢 LOW — On Track |
| 45 | 2.57% | 🟢 LOW — On Track |

---

## 🗂️ Dataset

- **Source:** [UCI ML Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Size:** 4,424 students × 35 features
- **Original Target:** 3 classes — `Graduate`, `Dropout`, `Enrolled`
- **Reframed Target:** Binary — `Dropout (1)` vs `Graduate (0)`

### Feature Types

| Type | Examples | Count |
|---|---|---|
| Binary flags | Gender, Scholarship, Debtor, Tuition up to date | 8 |
| Nominal categorical | Course, Nationality, Parents' occupation/qualification | 10 |
| Continuous numerical | Age, Semester grades, GDP, Unemployment rate | 16 |

---

## ⚙️ Preprocessing Pipeline

### The Critical Design Decision — Separating Enrolled Students

```
Full Dataset (4,424 rows)
        │
        ├── Enrolled students (794 rows)  →  SET ASIDE  →  Inference Set
        │
        └── Known outcomes (3,630 rows)   →  Training Pipeline
                    │
                    ├── Dropout  = 1
                    └── Graduate = 0
```

Enrolled students never touch the training or evaluation process. They are the real-world targets the model is built to serve.

### Step-by-Step

**1. One-Hot Encoding (sklearn `OneHotEncoder`)**

`pd.get_dummies` was deliberately avoided. When applied separately to two DataFrames, it causes category mismatches and inconsistent column dropping. sklearn's encoder `fit`s on training data only, then `transform`s everything else using the same learned category dictionary.

```python
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
ohe.fit_transform(X_train[nominal_cols])   # learns categories from training data
ohe.transform(X_enrolled[nominal_cols])    # applies same rules — never re-fits
```

**2. Train/Test Split before Scaling**

The scaler must only learn statistics (mean, std) from training data. Fitting on the full dataset would leak test set information into preprocessing.

**3. StandardScaler**

Applied to 16 continuous columns. Fitted on training data only, then applied to test and inference sets using the same fitted object.

**4. SMOTE — Training Set Only**

```python
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
# X_test is never touched — must reflect real-world distribution
```

---

## 🤖 Modeling

### Base Models

**Logistic Regression** — interpretable baseline; coefficients directly show which features increase dropout risk.

**Random Forest** — 100 decision trees voting on outcome; handles non-linear relationships and is robust to noise.

**XGBoost** — gradient boosting; sequentially corrects errors of previous trees; typically the strongest individual model.

### Meta-Learner (Stacking Ensemble)

```python
stacking_model = StackingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
    final_estimator=LogisticRegression(),
    cv=5  # out-of-fold predictions prevent leakage into meta-learner
)
```

The meta-learner takes the predictions of all three base models as its input and learns how to combine them optimally. Using `cv=5` ensures it trains on out-of-fold predictions — preventing the meta-learner from seeing inflated scores and overfitting.

---

## 🚨 Risk Report

The final output of the pipeline is a CSV report prioritising enrolled students by dropout risk, designed to be handed directly to a student support team.

```python
def assign_risk_label(prob):
    if prob >= 80: return 'CRITICAL  — Intervene Now'
    if prob >= 60: return 'HIGH      — Priority Support'
    if prob >= 40: return 'MEDIUM    — Monitor Closely'
    return         'LOW       — On Track'
```

**Output file:** `student_risk_report.csv`

| Column | Description |
|---|---|
| `Student_ID` | Row index from the original dataset |
| `Dropout_Probability` | Model's predicted dropout probability (%) |
| `Risk_Status` | Human-readable intervention priority label |

---

## 🔍 Explainability

### Logistic Regression Coefficients

Since features are scaled, LR coefficients are directly comparable — a positive coefficient means that feature increases dropout risk, negative means it reduces it. Top 10 risk factors and top 10 protective factors are visualised as bar charts.

### SHAP *(Planned)*

- **TreeExplainer** for Random Forest and XGBoost — fast and exact
- Global feature importance across all students
- Local explanations — why *this specific student* was flagged

### DiCE Counterfactuals *(Planned)*

Using [DiCE](https://github.com/interpretml/DiCE) to generate actionable recommendations:

> *"Student #553 is predicted to drop out (98.83% risk). If their 2nd semester approved units increased from 2 to 6 and tuition fees were brought up to date, this prediction would change to Graduate."*

---

## 🗃️ Project Structure

```
├── dataset.csv                  # Raw dataset
├── Clean_data.ipynb             # EDA + full preprocessing pipeline
├── Modeling.ipynb               # Model training, evaluation, risk report
├── train_data.csv               # SMOTE-balanced training set
├── test_data.csv                # Real holdout test set
├── enrolled_inference.csv       # Enrolled students for prediction
├── student_risk_report.csv      # Final output — risk scores per student
├── scaler.pkl                   # Fitted StandardScaler (reusable)
├── ohe_encoder.pkl              # Fitted OneHotEncoder (reusable)
└── README.md
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Preprocessing, modeling, evaluation, stacking |
| `imbalanced-learn` | SMOTE for class imbalance |
| `xgboost` | Gradient boosting classifier |
| `shap` | Global + local model explainability *(planned)* |
| `dice-ml` | Counterfactual explanations *(planned)* |
| `matplotlib`, `seaborn` | Visualisation |

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/socode6/Advanced_Python-DataScience.git
cd Advanced_Python-DataScience

# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap dice-ml matplotlib seaborn

# Step 1 — Preprocessing
jupyter notebook Clean_data.ipynb

# Step 2 — Modeling + Risk Report
jupyter notebook Modeling.ipynb
```

After running both notebooks, `student_risk_report.csv` will be generated with dropout risk scores for every currently enrolled student.

---

## 💡 Key Design Decisions

Non-obvious choices made in this project and why:

- **No outlier removal** — A 70-year-old student enrolling is real data, not measurement error. IQR removal is for sensor noise, not human behavioral data.
- **sklearn OHE over get_dummies** — `get_dummies` on two separate DataFrames produces mismatched columns. sklearn's fit/transform pattern guarantees consistency across train, test, and inference sets.
- **SMOTE after splitting** — Applying SMOTE before splitting leaks synthetic samples into the test set, making evaluation dishonest.
- **Out-of-fold CV in stacking** — Training the meta-learner on base model predictions from the same training data causes overfitting. `cv=5` generates honest held-out predictions.
- **Recall over Accuracy** — A model that predicts "Graduate" for every student gets ~61% accuracy but catches zero dropouts. Recall on the Dropout class is what actually matters for this problem.

---

## 👤 Author: SUBODH BHATTA

**Socode6**  
Advanced Python & Data Science — Submission Project
