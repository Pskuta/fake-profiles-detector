from pathlib import Path
import pandas as pd
from tqdm import tqdm 

from data_loading import load_twibot_split, normalize_twibot_record
from feature_extraction import extract_all_features

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_features_for_split(split: str, limit=None) -> pd.DataFrame:
    raw_data = load_twibot_split(split)
    if limit is not None:
        raw_data = raw_data[:limit]

    records = []
    for item in tqdm(raw_data, desc=f"Extracting features ({split})"):
        norm  = normalize_twibot_record(item)
        feats = extract_all_features(norm["user"], norm["tweets"])
        feats["label"]   = norm["label"]
        feats["user_id"] = norm["user"]["id"]
        records.append(feats)

    df = pd.DataFrame(records).fillna(0)
    out_path = PROCESSED_DIR / f"features_{split}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[{split}] Zapisano {len(df)} rekordów → {out_path}")
    return df


if __name__ == "__main__":
    build_features_for_split("train")
    build_features_for_split("dev")    
    build_features_for_split("test")
    print("\nBudowanie cech zakończone dla train / dev / test.")


