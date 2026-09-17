"""
Loan Prediction — deployment packaging of the ORIGINAL notebook logic.
No modeling changes. This mirrors your notebook's preprocessing and its
chosen final model (XGBoost) exactly, and just saves the fitted pieces
(model + encoders + fill values) so they can be reused outside the notebook.
"""
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier

DATA_PATH = "07_loan_train.csv"
ARTIFACT_PATH = "/mnt/user-data/outputs/loan_model_original.joblib"

# ---- original notebook logic, unchanged ----
df = pd.read_csv(DATA_PATH)
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

for c in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
          'Property_Area', 'Credit_History']:
    df[c] = df[c].astype('category')

X = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mode = X_train['Gender'].dropna().mode()[0]
mode1 = X_train['Married'].dropna().mode()[0]
mode2 = X_train['Dependents'].dropna().mode()[0]
mode3 = X_train['Self_Employed'].dropna().mode()[0]
mode4 = X_train['Property_Area'].dropna().mode()[0]
mode5 = X_train['Credit_History'].dropna().mode()[0]
median = X_train['LoanAmount'].dropna().median()
median1 = X_train['Loan_Amount_Term'].dropna().median()

X_train['Gender'].fillna(mode, inplace=True)
X_train['Married'].fillna(mode1, inplace=True)
X_train['Dependents'].fillna(mode2, inplace=True)
X_train['Self_Employed'].fillna(mode3, inplace=True)
X_train['Property_Area'].fillna(mode4, inplace=True)
X_train['Credit_History'].fillna(mode5, inplace=True)
X_train['LoanAmount'].fillna(median, inplace=True)
X_train['Loan_Amount_Term'].fillna(median1, inplace=True)

X_test['Gender'].fillna(mode, inplace=True)
X_test['Married'].fillna(mode1, inplace=True)
X_test['Dependents'].fillna(mode2, inplace=True)
X_test['Self_Employed'].fillna(mode3, inplace=True)
X_test['Property_Area'].fillna(mode4, inplace=True)
X_test['Credit_History'].fillna(mode5, inplace=True)
X_test['LoanAmount'].fillna(median, inplace=True)
X_test['Loan_Amount_Term'].fillna(median1, inplace=True)

le_target = LabelEncoder()
y_train_enc = le_target.fit_transform(y_train)
y_test_enc = le_target.transform(y_test)

cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'Property_Area', 'Credit_History']

feature_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X_train[c] = le.fit_transform(X_train[c].astype('category'))
    X_test[c] = le.transform(X_test[c].astype('category'))
    feature_encoders[c] = le  # saved so the app can apply the same mapping

XGBoost_clf = XGBClassifier(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=10,
    eval_metric="logloss",
)
XGBoost_clf.fit(X_train, y_train_enc)

y_train_pred = XGBoost_clf.predict(X_train)
y_test_pred = XGBoost_clf.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train_enc, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test_enc, y_test_pred))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test_enc, y_test_pred))

# ---- persist everything needed to reproduce this exact model at inference ----
joblib.dump({
    "model": XGBoost_clf,
    "target_encoder": le_target,
    "feature_encoders": feature_encoders,
    "fill_values": {
        "Gender": mode, "Married": mode1, "Dependents": mode2,
        "Self_Employed": mode3, "Property_Area": mode4, "Credit_History": mode5,
        "LoanAmount": median, "Loan_Amount_Term": median1,
    },
    "feature_columns": list(X_train.columns),
}, ARTIFACT_PATH)

print(f"\nSaved model bundle to {ARTIFACT_PATH}")
