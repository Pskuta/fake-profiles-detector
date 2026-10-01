"""
Trening trzech modeli: Logistic Regression, Random Forest, XGBoost.
Dla każdego modelu:
  - ewaluacja na zbiorze val i test
  - macierz pomyłek (PNG)
  - krzywa ROC (PNG)
  - tabela wyników zapisana do CSV
"""
import pandas as pd
import joblib
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    confusion_matrix, accuracy_score,
    RocCurveDisplay
)
from xgboost import XGBClassifier

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate_and_plot(model, name: str, X_val, y_val, X_test, y_test):
    """Ewaluacja modelu + zapis macierzy pomyłek i krzywej ROC."""
    safe_name = name.lower().replace(" ", "_")

    y_val_pred  = model.predict(X_val)
    y_val_prob  = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*55}")
    print(f"  MODEL: {name}")
    print(f"{'='*55}")
    print(classification_report(y_test, y_test_pred, target_names=["human", "bot"]))

    # ── Macierz pomyłek – TEST ────────────────────────────────
    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["human", "bot"],
                yticklabels=["human", "bot"], ax=ax)
    ax.set_title(f"Macierz pomyłek – {name} (test)")
    ax.set_ylabel("Prawdziwa")
    ax.set_xlabel("Przewidywana")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"cm_{safe_name}_test.png", dpi=150)
    plt.close(fig)

    # ── Krzywa ROC – TEST (bez zmian) ─────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_test_prob, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title(f"Krzywa ROC – {name} (test)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"roc_{safe_name}_test.png", dpi=150)
    plt.close(fig)

    metrics = {
        "Model":             name,
        "Accuracy (val)":    accuracy_score(y_val,  y_val_pred),
        "F1 (val)":          f1_score(y_val,  y_val_pred),
        "ROC-AUC (val)":     roc_auc_score(y_val,  y_val_prob),
        "Accuracy (test)":   accuracy_score(y_test, y_test_pred),
        "F1 (test)":         f1_score(y_test, y_test_pred),
        "ROC-AUC (test)":    roc_auc_score(y_test, y_test_prob),
    }
    return metrics


def main() -> None:
    # ── Wczytaj przygotowane dane ──────────────────────────────────────────────
    data = joblib.load(MODELS_DIR / "prepared_data.joblib")
    X_train_sc = data["X_train"];  y_train = data["y_train"]
    X_val_sc   = data["X_val"];    y_val   = data["y_val"]
    X_test_sc  = data["X_test"];   y_test  = data["y_test"]
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")

    print(f"Train: {len(X_train_sc)} | Val: {len(X_val_sc)} | Test: {len(X_test_sc)}")

    # ── 1. Logistic Regression ────────────────────────────────────────────────
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=42, C=1.0)
    lr.fit(X_train_sc, y_train)

    lr_metrics = evaluate_and_plot(lr, "Logistic Regression",
                                      X_val_sc, y_val, X_test_sc, y_test)
    joblib.dump(lr, MODELS_DIR / "lr_balanced.pkl")

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train)

    rf_metrics = evaluate_and_plot(rf, "Random Forest",
                                      X_val_sc, y_val, X_test_sc, y_test)
    joblib.dump(rf, MODELS_DIR / "rf_balanced.pkl")

    # ── 3. XGBoost ────────────────────────────────────────────────────────────
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train_sc, y_train,
            eval_set=[(X_val_sc, y_val)],
            verbose=50)
    xgb_metrics = evaluate_and_plot(xgb, "XGBoost",
                                        X_val_sc, y_val, X_test_sc, y_test)

    joblib.dump(xgb, MODELS_DIR / "xgb_balanced.pkl")

    print(f"\nNajlepszy numer drzewa XGBoost: {xgb.best_iteration}")

    # ── 4. XGBoost bez cechy "verified"───────────────────────────────────────
    feature_names_no_ver = [f for f in feature_names if f != "verified"]
    idx = [feature_names.index(f) for f in feature_names_no_ver]

    xgb_no_ver = XGBClassifier(**xgb.get_params())
    xgb_no_ver.fit(X_train_sc[:, idx], y_train,
               eval_set=[(X_val_sc[:, idx], y_val)],
               verbose=False)

    xgb_no_ver_metrics = evaluate_and_plot(
    xgb_no_ver, "XGBoost (bez verified)",
    X_val_sc[:, idx], y_val, X_test_sc[:, idx], y_test)
    joblib.dump(xgb_no_ver, MODELS_DIR / "xgb_no_ver_balanced.pkl")

    # ── Feature importance – Logistic Regression ─────────────────────────────────
    feat_imp_lr = (
        pd.DataFrame({"feature": feature_names,
                    "importance": np.abs(lr.coef_[0])})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(feat_imp_lr)), feat_imp_lr["importance"], color="#E3F02E")
    ax.set_yticks(range(len(feat_imp_lr)))
    ax.set_yticklabels(feat_imp_lr["feature"])
    ax.set_title("Top 15 cech – Logistic Regression (|współczynnik|)")
    ax.set_xlabel("|Wartość współczynnika|")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance_lr.png", dpi=150)
    plt.close(fig)
    print("\nTop 15 cech LR:")
    print(feat_imp_lr.to_string(index=False))

    # ── Feature importance – Random Forest ───────────────────────────────────────
    feat_imp_rf = (
        pd.DataFrame({"feature": feature_names,
                    "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(feat_imp_rf)), feat_imp_rf["importance"], color="#3A86FF")
    ax.set_yticks(range(len(feat_imp_rf)))
    ax.set_yticklabels(feat_imp_rf["feature"])
    ax.set_title("Top 15 cech – Random Forest (feature importance)")
    ax.set_xlabel("Importance (Gini)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance_rf.png", dpi=150)
    plt.close(fig)
    print("\nTop 15 cech RF:")
    print(feat_imp_rf.to_string(index=False))

    # ── Feature importance – XGBoost ─────────────────────────────────────────────
    feat_imp = (
        pd.DataFrame({"feature": feature_names,
                      "importance": xgb.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(feat_imp)), feat_imp["importance"], color="#1EDD9D")
    ax.set_yticks(range(len(feat_imp)))
    ax.set_yticklabels(feat_imp["feature"])
    ax.set_title("Top 15 cech – XGBoost (feature importance)")
    ax.set_xlabel("Importance (Gain)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance_xgb.png", dpi=150)
    plt.close(fig)
    print("\nTop 15 cech XGB:")
    print(feat_imp.to_string(index=False))

    # ── Zapis tabeli wyników ──────────────────────────────────────────────────
    results_df = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics, xgb_no_ver_metrics]).round(4) 
    results_df.to_csv(RESULTS_DIR / "model_results.csv", index=False)

    print("\n── Tabela wyników (test) ──")
    print(results_df[["Model", "Accuracy (test)", "F1 (test)",
                       "ROC-AUC (test)"]].to_string(index=False))
    print("\n Modele zapisane! Sprawdź results/ dla wykresów i tabeli.")


if __name__ == "__main__":
    main()

