# train_ml_fast.py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier

from data_generator import generate_dataset

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["foyda"] = df["oylik_daromad"] - df["oylik_xarajat"]
    df["foyda_marjasi"] = df["foyda"] / df["oylik_daromad"].clip(lower=1)
    df["qarz_nisbati"] = df["qarz_miqdori"] / df["oylik_daromad"].clip(lower=1)
    df["xarajat_nisbati"] = df["oylik_xarajat"] / df["oylik_daromad"].clip(lower=1)
    df["kredit_yuklama"] = df["kredit_oylik_tolovi"] / df["oylik_daromad"].clip(lower=1)
    df = pd.get_dummies(df, columns=["soha", "hudud"], drop_first=True)
    return df

def find_best_threshold(y_true, y_proba, min_recall=0.75):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.0)

    best_t = 0.5
    best_f1 = -1
    for p, r, t in zip(precision, recall, thresholds):
        if r >= min_recall:
            f1 = (2 * p * r) / (p + r + 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
    return best_t, best_f1

def main():
    # ✅ tezroq: 12k satr yetadi (keyin xohlasang 50k qilamiz)
    df = generate_dataset(n=12000, seed=42)
    df = make_features(df)

    features = [c for c in df.columns if c not in ["risk_score", "bankrot"]]
    X = df[features]
    y = df["bankrot"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    # ✅ RandomizedSearch: oz sonli sinov -> tez
    param_dist = {
        "n_estimators": [200, 300, 400],
        "max_depth": [8, 10, 12, None],
        "min_samples_leaf": [3, 5, 8, 12],
        "min_samples_split": [8, 12, 20],
        "max_features": ["sqrt", 0.6, 0.8]
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=18,                 # ✅ 18 ta sinov (tez)
        scoring="roc_auc",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1                  # ✅ parallel
    )

    search.fit(X_train, y_train)
    model = search.best_estimator_

    print("\n✅ Best params:", search.best_params_)
    print("✅ Best CV AUC:", round(search.best_score_, 4))

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    best_t, best_f1 = find_best_threshold(y_test, proba, min_recall=0.75)
    pred = (proba >= best_t).astype(int)

    print("\n=== TEST METRICS ===")
    print("AUC:", round(auc, 4))
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("Chosen threshold:", round(best_t, 3), "| F1:", round(best_f1, 4))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nReport:\n", classification_report(y_test, pred))

    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n=== TOP 15 IMPORTANT FEATURES ===")
    print(fi.head(15).round(4))

    import joblib
    joblib.dump(model, "rf_bankrot_model.pkl")
    joblib.dump(features, "model_features.pkl")
    joblib.dump(best_t, "best_threshold.pkl")
    print("\n✅ Saved: rf_bankrot_model.pkl, model_features.pkl, best_threshold.pkl")

if __name__ == "__main__":
    main()