import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


# ─────────────────────────────────────────
# 1. Budowanie tweetów 
# ─────────────────────────────────────────

def build_tweet_objects(raw_tweets):
    """
    raw_tweets: lista stringów z TwiBot-20 ('tweet').
    Zwraca listę słowników z polami: text, is_retweet, is_reply.
    """
    tweets = []

    for text in raw_tweets:
        if not isinstance(text, str):
            text = str(text)

        tweets.append({
            "text":       text,
            "is_retweet": text.strip().startswith("RT "),
            "is_reply":   text.strip().startswith("@"),
        })

    return tweets


# ─────────────────────────────────────────
# 2. Grupa A – cechy statyczne profilu
# ─────────────────────────────────────────

def extract_group_a(user: dict) -> dict:
    followers = user.get("followers_count", 0)
    following = user.get("following_count", 0)

    return {
        "account_age_days":  user.get("account_age_days", 0),
        "ff_ratio":          followers / (following + 1),
        "listed_count":      user.get("listed_count", 0),
        "has_profile_image": int(user.get("has_profile_image", False)),
        "has_description":   int(bool(user.get("description", "").strip())),
        "has_url":           int(bool(user.get("url", None))),
        "verified":          int(user.get("verified", False)),
    }


# ─────────────────────────────────────────
# 3. Grupa B – cechy behawioralne
# ─────────────────────────────────────────

def extract_group_b(user: dict, tweets: list) -> dict:
    age = max(user.get("account_age_days", 1), 1)
    n   = max(len(tweets), 1)


    retweets = sum(1 for t in tweets if t.get("is_retweet", False))
    replies  = sum(1 for t in tweets if t.get("is_reply",   False))

    return {
        "tweets_per_day":       user.get("statuses_count", 0) / age,
        "retweet_ratio":        retweets / n,
        "reply_ratio":          replies  / n,
    }


# ─────────────────────────────────────────
# 4. Grupa C – cechy semantyczne (NLP)
# ─────────────────────────────────────────

def extract_group_c(tweets: list) -> dict:
    _empty = {k: 0.0 for k in [
        "avg_sentiment", "std_sentiment", "ttr",
        "avg_hashtags", "avg_mentions", 
        "url_post_ratio", "duplication_ratio"
    ]}

    texts = [t.get("text", "") for t in tweets if t.get("text", "")]
    if not texts:
        return _empty

    scores = [analyzer.polarity_scores(t)["compound"] for t in texts]

    all_words = " ".join(texts).lower().split()
    ttr = len(set(all_words)) / max(len(all_words), 1)

    avg_ht  = float(np.mean([len(re.findall(r"#\w+",  t)) for t in texts]))
    avg_mn  = float(np.mean([len(re.findall(r"@\w+",  t)) for t in texts]))

    duplication = 1.0 - (len(set(texts)) / len(texts))

    return {
        "avg_sentiment":     float(np.mean(scores)),
        "std_sentiment":     float(np.std(scores)),
        "ttr":               float(ttr),
        "avg_hashtags":      avg_ht,
        "avg_mentions":      avg_mn,
        "url_post_ratio":    float(np.mean([1 if "http" in t else 0 for t in texts])),
        "duplication_ratio": float(duplication),
    }


# ─────────────────────────────────────────
# 5. Główna funkcja – łączy wszystkie grupy
# ─────────────────────────────────────────

def extract_all_features(user: dict, tweets: list) -> dict:
    features = {}
    features.update(extract_group_a(user))
    features.update(extract_group_b(user, tweets))
    features.update(extract_group_c(tweets))
    return features






