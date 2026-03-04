# train_ml.py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier

from data_generator import generate_dataset

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # Feature engineering
    df = df.copy()
    df["foyda"] = df["oylik_daromad"] - df["oylik_xarajat"]
    df["foyda_marjasi"] = df["foyda"] / df["oylik_daromad"].clip(lower=1)
    df["qarz_nisbati"] = df["qarz_miqdori"] / df["oylik_daromad"].clip(lower=1)
    df["xarajat_nisbati"] = df["oylik_xarajat"] / df["oylik_daromad"].clip(lower=1)
    df["kredit_yuklama"] = df["kredit_oylik_tolovi"] / df["oylik_daromad"].clip(lower=1)

    # Categorical encoding
    df = pd.get_dummies(df, columns=["soha", "hudud"], drop_first=True)
    return df

def find_best_threshold(y_true, y_proba, min_recall=0.75):
    """
    Bankrot (1) ni o'tkazib yubormaslik uchun recallni pastga tushirmay,
    eng yaxshi threshold topadi.
    """
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
    # 1) Dataset
    df = generate_dataset(n=20000, seed=42)  # ko'proq data -> barqarorroq
    df = make_features(df)

    # 2) Split
    features = [c for c in df.columns if c not in ["risk_score", "bankrot"]]
    X = df[features]
    y = df["bankrot"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3) Model + Tuning (GridSearch)
    base = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [8, 12, None],
        "min_samples_leaf": [5, 10],
        "min_samples_split": [10, 20],
        "max_features": ["sqrt", 0.6]
    }

    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        verbose=1
    )

    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    print("\n✅ Best params:", grid.best_params_)
    print("✅ Best CV AUC:", round(grid.best_score_, 4))

    # 4) Evaluate
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    # 5) Threshold tuning (recall priority)
    best_t, best_f1 = find_best_threshold(y_test, proba, min_recall=0.75)
    pred = (proba >= best_t).astype(int)

    print("\n=== TEST METRICS ===")
    print("AUC:", round(auc, 4))
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("Chosen threshold:", round(best_t, 3), "| F1:", round(best_f1, 4))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nReport:\n", classification_report(y_test, pred))

    # 6) Feature importance (top 15)
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n=== TOP 15 IMPORTANT FEATURES ===")
    print(fi.head(15).round(4))

    # 7) Save model artifacts
    # Minimal: modelni joblib bilan saqlab ketamiz
    import joblib
    joblib.dump(model, "rf_bankrot_model.pkl")
    joblib.dump(features, "model_features.pkl")
    print("\n✅ Saved: rf_bankrot_model.pkl, model_features.pkl")

if __name__ == "__main__":
    main()