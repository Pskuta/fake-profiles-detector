from pathlib import Path
import json
from datetime import datetime, timezone
from feature_extraction import build_tweet_objects

DATA_DIR = Path("data/raw/TwiBot-20")

REF_DATE = datetime(2020, 9, 1, tzinfo=timezone.utc)


def load_twibot_split(split: str):
    path = DATA_DIR / f"{split}.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def parse_int_safe(value, default=0):
    try:
        return int(str(value).strip())
    except Exception:
        return default


def parse_bool_safe(value, true_values=("True", "true", "1")):
    if isinstance(value, bool):
        return value
    return str(value).strip() in true_values


def parse_created_at(created_at_str: str):
    s = created_at_str.strip()
    return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")


def compute_account_age_days(created_at, ref_date=REF_DATE):
    """
    Oblicza wiek konta w dniach względem stałej daty referencyjnej (2020-09-01).
    """
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return max((ref_date - created_at).days, 1)


def normalize_twibot_record(raw_item: dict):
    profile  = raw_item.get("profile", {}) or {}
    raw_tweets = raw_item.get("tweet", []) or []

    created_at_raw = profile.get("created_at", "").strip()
    try:
        created_at_dt  = parse_created_at(created_at_raw)
        account_age_days = compute_account_age_days(created_at_dt)
    except Exception:
        account_age_days = 1

    followers_count = parse_int_safe(profile.get("followers_count", 0))
    friends_count    = parse_int_safe(profile.get("friends_count", 0))

    user = {
        "id":               profile.get("id", "").strip(),
        "screen_name":      profile.get("screen_name", "").strip(),
        "description":      profile.get("description", "").strip(),
        "url":              None if profile.get("url", "").strip() in ("None", "")
                            else profile.get("url", "").strip(),
        "followers_count":  followers_count,
        "following_count":  friends_count,
        "listed_count":     parse_int_safe(profile.get("listed_count", 0)),
        "statuses_count":   parse_int_safe(profile.get("statuses_count", 0)),
        "verified":         parse_bool_safe(profile.get("verified", "False")),
        "has_profile_image": not parse_bool_safe(profile.get("default_profile_image", "False")),
        "account_age_days": account_age_days,
    }

    tweets = build_tweet_objects(raw_tweets)

    raw_label = str(raw_item.get("label", "0")).strip()
    # TwiBot-20: '1' = bot, '0' = human
    label_int = 1 if raw_label == "1" else 0

    return {
        "user":   user,
        "tweets": tweets,
        "label":  label_int,
    }

