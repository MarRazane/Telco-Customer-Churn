import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_preprocess(X: pd.DataFrame, num_cols: list[str]) -> ColumnTransformer:
    cat_cols_model = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_model),
        ],
        remainder="drop"
    )

def train_models(df: pd.DataFrame, random_state: int = 42):
    X = df.drop(columns=["Churn"]).copy()
    y = df["Churn"].copy()

    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "addon_service_count"]
    preprocess = build_preprocess(X, num_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Logistic Regression
    logreg_pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", LogisticRegression(max_iter=2000))
    ])
    logreg_pipe.fit(X_train, y_train)
    proba_lr = logreg_pipe.predict_proba(X_test)[:, 1]
    pred_lr = (proba_lr >= 0.5).astype(int)

    lr_metrics = {
        "roc_auc": roc_auc_score(y_test, proba_lr),
        "confusion_matrix": confusion_matrix(y_test, pred_lr),
        "report": classification_report(y_test, pred_lr, digits=4),
        "proba": proba_lr,
        "pred": pred_lr,
        "y_test": y_test
    }

    # Random Forest
    rf_pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=400,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])
    rf_pipe.fit(X_train, y_train)
    proba_rf = rf_pipe.predict_proba(X_test)[:, 1]
    pred_rf = (proba_rf >= 0.5).astype(int)

    rf_metrics = {
        "roc_auc": roc_auc_score(y_test, proba_rf),
        "confusion_matrix": confusion_matrix(y_test, pred_rf),
        "report": classification_report(y_test, pred_rf, digits=4),
        "proba": proba_rf,
        "pred": pred_rf
    }

    return logreg_pipe, rf_pipe, lr_metrics, rf_metrics

def logistic_coefficients(logreg_pipe, df: pd.DataFrame):
    X = df.drop(columns=["Churn"]).copy()
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "addon_service_count"]

    ohe = logreg_pipe.named_steps["prep"].named_transformers_["cat"]
    cat_cols_model = [c for c in X.columns if c not in num_cols]
    ohe_feature_names = ohe.get_feature_names_out(cat_cols_model)

    feature_names = np.concatenate([np.array(num_cols), ohe_feature_names])
    coef = logreg_pipe.named_steps["model"].coef_.ravel()

    coef_df = pd.DataFrame({"feature": feature_names, "coef": coef}).sort_values("coef", ascending=False)
    return coef_df
