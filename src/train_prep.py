import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def main():
    df_train = pd.read_parquet(PROCESSED_DIR / "features_train.parquet")
    df_val   = pd.read_parquet(PROCESSED_DIR / "features_dev.parquet")
    df_test  = pd.read_parquet(PROCESSED_DIR / "features_test.parquet")

    X_train = df_train.drop(["user_id", "label"], axis=1)
    y_train = df_train["label"]

    X_val   = df_val.drop(["user_id", "label"], axis=1)
    y_val   = df_val["label"]

    X_test  = df_test.drop(["user_id", "label"], axis=1)
    y_test  = df_test["label"]

    print(f"Train : {len(X_train):>6}  |  Val : {len(X_val):>6}  |  Test : {len(X_test):>6}")
    print(f"Proporcja train/val/test: "
          f"{len(X_train)/(len(X_train)+len(X_val)+len(X_test)):.0%} / "
          f"{len(X_val)/(len(X_train)+len(X_val)+len(X_test)):.0%} / "
          f"{len(X_test)/(len(X_train)+len(X_val)+len(X_test)):.0%}")
    
    print("\nRozkład klas:")
    print(f"  train : {y_train.value_counts().to_dict()}")
    print(f"  val   : {y_val.value_counts().to_dict()}")
    print(f"  test  : {y_test.value_counts().to_dict()}")


#   Skalowanie – fit TYLKO na train 
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

#   Zapis artefaktów 
    joblib.dump(
        {"X_train": X_train_sc, "y_train": y_train,
         "X_val":   X_val_sc,   "y_val":   y_val,
         "X_test":  X_test_sc,  "y_test":  y_test},
        MODELS_DIR / "prepared_data.joblib"
    )
    joblib.dump(scaler,                   MODELS_DIR / "scaler.pkl")
    joblib.dump(X_train.columns.tolist(), MODELS_DIR / "feature_names.pkl")

    print("\n Gotowe! Dane zapisane do models/prepared_data.joblib")


if __name__ == "__main__":
    main()

