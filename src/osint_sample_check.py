import pandas as pd
import joblib
import json
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
OSINT_DIR     = Path("osint")
OSINT_DIR.mkdir(exist_ok=True)

SAMPLE_SIZE  = 20
RANDOM_STATE = 42


def main() -> None:
    data      = joblib.load(MODELS_DIR / "prepared_data.joblib")
    xgb_model = joblib.load(MODELS_DIR / "xgb_balanced.pkl")
    df_test   = pd.read_parquet(PROCESSED_DIR / "features_test.parquet")

    with open("data/raw/TwiBot-20/test.json", "r", encoding="utf-8") as f:
        raw_test = json.load(f)

    screen_names = {
        str(i.get("profile", {}).get("id",          "")).strip():
        str(i.get("profile", {}).get("screen_name", "")).strip()
        for i in raw_test
    }

    display_names = {
        str(i.get("profile", {}).get("id",   "")).strip():
        str(i.get("profile", {}).get("name", "")).strip()
        for i in raw_test
    }

    display_profile_image_url = {
        str(i.get("profile", {}).get("id",   "")).strip():
        str(i.get("profile", {}).get("profile_image_url", "")).strip()
        for i in raw_test
    }

    df_test["screen_name"] = df_test["user_id"].astype(str).str.strip().map(screen_names).fillna("")
    df_test["name"]        = df_test["user_id"].astype(str).str.strip().map(display_names).fillna("")

    print(f"Dopasowano screen_name: {(df_test['screen_name'] != '').sum()}/{len(df_test)}")

    df_test["ai_prob"] = xgb_model.predict_proba(data["X_test"])[:, 1]
    df_test["ai_pred"] = (df_test["ai_prob"] >= 0.5).astype(int)
    df_test["correct"] = df_test["ai_pred"] == df_test["label"]

    pewne_boty   = df_test[(df_test["label"]==1) & (df_test["ai_prob"]>0.90)].sample(
        min(SAMPLE_SIZE, int(((df_test["label"]==1) & (df_test["ai_prob"]>0.90)).sum())), random_state=RANDOM_STATE).copy()
    
    pewni_ludzie = df_test[(df_test["label"]==0) & (df_test["ai_prob"]<0.10)].sample(
        min(SAMPLE_SIZE, int(((df_test["label"]==0) & (df_test["ai_prob"]<0.10)).sum())), random_state=RANDOM_STATE).copy()
    
    graniczne    = df_test[(df_test["ai_prob"]>=0.40) & (df_test["ai_prob"]<=0.60)].sample(
        min(SAMPLE_SIZE, int(((df_test["ai_prob"]>=0.40) & (df_test["ai_prob"]<=0.60)).sum())), random_state=RANDOM_STATE).copy()
    
    sprzecznosci = df_test[~df_test["correct"]].sample(
        min(SAMPLE_SIZE, int((~df_test["correct"]).sum())), random_state=RANDOM_STATE).copy()

    pewne_boty["group"]   = "pewny_bot"
    pewni_ludzie["group"] = "pewny_human"
    graniczne["group"]    = "graniczny"
    sprzecznosci["group"] = "sprzecznosc"

    probka = pd.concat([pewne_boty, pewni_ludzie, graniczne, sprzecznosci]) \
               .drop_duplicates(subset="user_id", keep="first")

    probka_out = probka[["user_id", "screen_name", "name",
                          "ai_prob", "ai_pred", "label", "group"]].copy()

    probka_out["profile_image_url_original"] = probka_out["user_id"].astype(str).str.strip().map(display_profile_image_url).fillna("")
    probka_out["profile_image_url"] = probka_out["profile_image_url_original"].apply(
        lambda url: f"https://web.archive.org/web/*/{url}" if url else "")
    probka_out["wayback_machine_url"] = probka_out["screen_name"].apply(
        lambda sn: f"https://web.archive.org/web/*/https://twitter.com/{sn}" if sn else "")

    probka_out["ai_prob"]       = probka_out["ai_prob"].round(4)
    probka_out["ai_pred_label"] = probka_out["ai_pred"].map({1: "bot",  0: "human"})
    probka_out["dataset_label"] = probka_out["label"].map({1: "bot",    0: "human"})

    for col in ["botometer_score", "sherlock_profile_notes", "reverse_image",
                "wayback_notes", "sum_points", "osint_label", "comments"]:
        probka_out[col] = ""

    out_path = OSINT_DIR / "sample_to_check.csv"
    probka_out.drop(columns=["ai_pred", "label"]).to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\nRozmiar próbki: {len(probka_out)} profili")
    print(f"  - pewne boty:    {len(pewne_boty)}")
    print(f"  - pewni ludzie:  {len(pewni_ludzie)}")
    print(f"  - graniczne:     {len(graniczne)}")
    print(f"  - sprzeczności:  {len(sprzecznosci)}")
    print(f"\n Zapisano {len(probka_out)} profili → {out_path}")
    print(probka_out[["screen_name", "name", "profile_image_url",
                       "wayback_machine_url", "ai_prob",
                       "ai_pred_label", "dataset_label", "group"]].head())


if __name__ == "__main__":
    main()


