# Fake Profiles Detector

> English version below (click to expand) | Wersja polska poniżej (kliknij, aby rozwinąć)

<details open>
<summary><b>English</b></summary>


## Project summary

The growing automation of social media activity is a serious threat to the credibility of published content. This project aims to detect fake profiles (bots) on X (formerly Twitter) by combining:

- **ML classification** - three models (Logistic Regression, Random Forest, XGBoost) trained on 17 profile features,
- **OSINT verification** - a manual, exploratory analysis of selected profiles using open-source intelligence tools (Sherlock, Wayback Machine, reverse image search, Botometer X).

The experiments were run on the **TwiBot-20** benchmark (11,826 labelled profiles: 8,278 train / 2,365 dev / 1,183 test, a 7:2:1 split). The models reached a ROC-AUC between **0.877 and 0.923** on the test set.

### Key finding

Machine learning models detect fake profiles effectively, but their high performance relies largely on a single feature: the account verification status (`verified`). An ablation experiment (removing this feature from XGBoost) lowered accuracy from 0.8555 to 0.7921, F1 from 0.8711 to 0.8057, and ROC-AUC from 0.9230 to 0.8782. This means the system is less effective for profiles where verification information is unavailable.

## Data architecture

### Data source

**TwiBot-20** (Feng, Wan, Wang, Li, Luo) is one of the largest Twitter bot detection benchmarks, covering 229,573 users, 33,488,192 tweets, 8,723,736 profile attributes and 455,958 follow relations (collected July–September 2020). Only 11,826 profiles carry `bot`/`human` labels and belong to the official train/dev/test split.

The dataset is available for research purposes only, after obtaining permission from the authors (contact by email → Google Drive link).

### Record structure

| Field | Contents |
|---|---|
| `ID` | user identifier |
| `profile` | profile metadata from the API |
| `tweet` | the user's last 200 tweets |
| `neighbour` | 20 random followers/following |
| `domain` | activity domain (politics, business, entertainment, sports) |
| `label` | label: 1 = bot, 0 = human |

### Data processing pipeline

```
TwiBot-20 (train.json / dev.json / test.json)
        │
        ▼
 data_loading.py        → record normalisation (types, dates, missing data)
        │
        ▼
 feature_extraction.py  → extraction of 17 features (Groups A / B / C)
 build_features.py      → aggregation to DataFrame → data/processed/*.parquet
        │
        ▼
 train_prep.py          → StandardScaler (fit on train only!) → models/prepared_data.joblib
        │
        ▼
 train_models.py        → training: Logistic Regression, Random Forest, XGBoost
                           (+ XGBoost without the "verified" feature - ablation)
        │
        ▼
 osint_sample_check.py  → sampling 4×20 test profiles → osint/sample_to_check.csv
                           (manual verification: Sherlock, Wayback Machine,
                           reverse image search, Botometer X)
```

Key rule: `StandardScaler` is fitted **only** on the training set and then applied (`transform`) to the validation and test sets, which prevents *data leakage*.

### Feature groups (17 attributes)

**Group A - static profile features**

| Feature | Description |
|---|---|
| `account_age_days` | account age in days relative to the reference date 2020-09-01 |
| `ff_ratio` | followers_count / (following_count + 1) |
| `listed_count` | number of lists the user appears on |
| `has_profile_image` | whether the account has a profile picture |
| `has_description` | whether the account has a bio |
| `has_url` | whether the profile contains a URL |
| `verified` | account verification status |

**Group B - behavioural features**

| Feature | Description |
|---|---|
| `tweets_per_day` | statuses_count / account age |
| `retweet_ratio` | share of retweets in activity |
| `reply_ratio` | share of replies in activity |

**Group C - semantic features (tweet text analysis, VADER)**

| Feature | Description |
|---|---|
| `avg_sentiment` | mean VADER compound sentiment score |
| `std_sentiment` | standard deviation of sentiment |
| `ttr` | Type-Token Ratio - vocabulary richness |
| `avg_hashtags` | average number of hashtags per post |
| `avg_mentions` | average number of mentions (@) per post |
| `url_post_ratio` | share of posts containing at least one URL |
| `duplication_ratio` | share of duplicated tweets |

## Methodology and OSINT verification

The automatic classification is complemented by a manual analysis of a sample of **79 unique profiles** from the test set, split into 4 categories:

| Group | Criterion | Purpose |
|---|---|---|
| `pewny_bot` (certain bot) | label=1 & ai_prob > 0.90 | verifying confident bots |
| `pewny_human` (certain human) | label=0 & ai_prob < 0.10 | verifying confident humans |
| `graniczny` (borderline) | 0.40 ≤ ai_prob ≤ 0.60 | ambiguous cases |
| `sprzeczność` (contradiction) | prediction ≠ dataset label | classifier error analysis |

The OSINT assessment uses a scoring system (Botometer X, reverse image search, Wayback Machine, Sherlock) modelled on the bot characteristics described by Ferrara et al. in *"The Rise of Social Bots"*. The total score determines the `osint_label`: `human` (≤0), `niejednoznaczny` / ambiguous (1–3), `bot` (≥4).

## Tech stack

The project is implemented in **Python 3.12.2**.

| Category | Libraries | Purpose |
|---|---|---|
| Data processing | `pandas`, `numpy` | data manipulation, feature extraction |
| Progress tracking | `tqdm` | feature extraction progress bar |
| Sentiment analysis | `vaderSentiment` | avg_sentiment, std_sentiment |
| Modelling (classical ML) | `scikit-learn` | Logistic Regression, Random Forest, StandardScaler, metrics |
| Modelling (boosting) | `xgboost` | XGBClassifier with class imbalance handling |
| Visualisation | `matplotlib`, `seaborn` | confusion matrices, ROC curves, feature importance |
| Data storage | `pyarrow`, `joblib` | Parquet storage, model serialisation |
| Paths | `pathlib` | file path management |

### Model configuration

- **Logistic Regression** - `C=1.0`, `max_iter=1000`, `class_weight="balanced"`, `random_state=42`.
- **Random Forest** - `n_estimators=200`, `max_depth=15`, `class_weight="balanced"`, `n_jobs=-1`, `random_state=42`.
- **XGBoost** - `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=3`, `scale_pos_weight=neg/pos`, `early_stopping_rounds=50`.

## Model results

| Model | Accuracy (val) | F1 (val) | ROC-AUC (val) | Accuracy (test) | F1 (test) | ROC-AUC (test) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8262 | 0.8518 | 0.8952 | 0.8276 | 0.8504 | 0.9002 |
| Random Forest | 0.8490 | 0.8720 | 0.9147 | 0.8529 | 0.8721 | 0.9211 |
| **XGBoost** | 0.8478 | 0.8671 | 0.9186 | **0.8555** | 0.8711 | **0.9230** |
| XGBoost (without `verified`) | 0.8000 | 0.8186 | 0.8774 | 0.7921 | 0.8057 | 0.8782 |

Random Forest achieved the highest F1 score, while XGBoost achieved the highest accuracy and ROC-AUC. The variant without `verified` shows a drop in every metric, confirming the feature's dominant influence on classification.

### Confusion matrix (test set)

| Model | TP | FN | FP | TN | Precision | Recall | Specificity |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 580 | 60 | 144 | 399 | 0.801 | 0.906 | 0.735 |
| Random Forest | 593 | 47 | 127 | 416 | 0.824 | 0.927 | 0.766 |
| XGBoost | 578 | 62 | 109 | 434 | 0.841 | 0.903 | 0.799 |
| XGBoost (without `verified`) | 510 | 130 | 116 | 427 | 0.815 | 0.797 | 0.786 |

## Repository structure

```
fake-profiles-detector/
├── data/
│   ├── raw/                    # train.json, dev.json, test.json (TwiBot-20) - NOT in repo
│   └── processed/              # features_train/dev/test.parquet - NOT in repo
├── models/
│   ├── prepared_data.joblib    # NOT in repo (large binary file)
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── lr_balanced.pkl         # NOT in repo (large binary file)
│   ├── rf_balanced.pkl         # NOT in repo (large binary file)
│   └── xgb_balanced.pkl        # NOT in repo (large binary file)
├── notebooks/
│   └── check_features_train.ipynb
├── osint/
│   └── sample_to_check.csv     # sample of 79 profiles for manual OSINT verification
├── results/
│   ├── model_results.csv       # aggregated metrics for all models
│   └── *.png                   # ROC curves, confusion matrices, feature importance
├── data_loading.py
├── feature_extraction.py
├── build_features.py
├── train_prep.py
├── train_models.py
├── osint_sample_check.py
└── README.md
```

## Reproducing the results

The repository contains only source code, configuration, lightweight artefacts (`scaler.pkl`, `feature_names.pkl`) and result files in CSV/PNG format. To reproduce the full pipeline, you need to obtain access to TwiBot-20 yourself (by contacting the benchmark authors) and run the scripts in this order: `data_loading.py` → `feature_extraction.py`/`build_features.py` → `train_prep.py` → `train_models.py` → `osint_sample_check.py`.

## Limitations

- The models' high performance depends largely on the `verified` feature, which is heavily imbalanced - most profiles do not have it.
- Sentiment analysis (VADER) relies on an English-language lexicon, which reduces the reliability of `avg_sentiment` and `std_sentiment` for tweets in other languages.
- TwiBot-20 dates from September 2020 - some accounts may have been deleted or suspended since, which limits OSINT verification (e.g. in the Wayback Machine).
- The OSINT scoring system is an auxiliary method developed for this work, not a validated diagnostic scale.

</details>

<details>
<summary><b>Polski</b></summary>


## Podsumowanie projektu

Rosnąca automatyzacja aktywności w mediach społecznościowych stanowi istotne zagrożenie dla wiarygodności publikowanych treści. Celem projektu jest wykrywanie fałszywych profili (botów) na platformie X (dawniej Twitter) poprzez połączenie:

- **klasyfikacji ML** - trzech modeli (Logistic Regression, Random Forest, XGBoost) trenowanych na 17 cechach profilu,
- **weryfikacji OSINT** - ręcznej, eksploracyjnej analizy wybranych profili z użyciem narzędzi białego wywiadu (Sherlock, Wayback Machine, reverse image search, Botometer X).

Badania przeprowadzono na benchmarku **TwiBot-20** (11 826 oznaczonych profili: 8 278 train / 2 365 dev / 1 183 test, podział 7:2:1). Modele osiągnęły ROC-AUC w przedziale **0.877–0.923** na zbiorze testowym.

### Główny wniosek

Modele uczenia maszynowego umożliwiają skuteczną detekcję fałszywych profili, jednak ich wysoka efektywność opiera się w dużej mierze na jednej cesze - statusie weryfikacji konta (`verified`). Eksperyment ablacyjny (usunięcie tej cechy z XGBoost) obniżył accuracy z 0.8555 do 0.7921, F1 z 0.8711 do 0.8057, a ROC-AUC z 0.9230 do 0.8782. Oznacza to, że skuteczność systemu jest ograniczona dla profili, gdzie informacja o weryfikacji jest niedostępna.

## Architektura danych

### Źródło danych

Zbiór **TwiBot-20** (Feng, Wan, Wang, Li, Luo) - jeden z największych benchmarków do detekcji botów na Twitterze, obejmujący 229 573 użytkowników, 33 488 192 tweetów, 8 723 736 atrybutów profili i 455 958 relacji obserwowania (dane zbierane VII–IX 2020). Tylko 11 826 profili posiada etykiety `bot`/`human` i wchodzi w oficjalny podział train/dev/test.

Zbiór jest udostępniany wyłącznie do celów naukowych po uzyskaniu zgody od współautorów (kontakt mailowy → link do Google Drive).

### Struktura pojedynczego rekordu

| Pole | Zawartość |
|---|---|
| `ID` | identyfikator użytkownika |
| `profile` | metadane profilu z API |
| `tweet` | ostatnie 200 tweetów użytkownika |
| `neighbour` | 20 losowych followers/following |
| `domain` | domena aktywności (polityka, biznes, rozrywka, sport) |
| `label` | etykieta: 1 = bot, 0 = człowiek |

### Pipeline przetwarzania danych

```
TwiBot-20 (train.json / dev.json / test.json)
        │
        ▼
 data_loading.py        → normalizacja rekordów (typy, daty, braki danych)
        │
        ▼
 feature_extraction.py  → ekstrakcja 17 cech (Grupy A / B / C)
 build_features.py      → agregacja do DataFrame → data/processed/*.parquet
        │
        ▼
 train_prep.py          → StandardScaler (fit tylko na train!) → models/prepared_data.joblib
        │
        ▼
 train_models.py        → trening: Logistic Regression, Random Forest, XGBoost
                           (+ XGBoost bez cechy "verified" - ablacja)
        │
        ▼
 osint_sample_check.py  → losowanie 4×20 profili testowych → osint/sample_to_check.csv
                           (ręczna weryfikacja: Sherlock, Wayback Machine,
                           reverse image search, Botometer X)
```

Kluczowa zasada: `StandardScaler` jest fitowany **wyłącznie** na zbiorze treningowym, a następnie stosowany (`transform`) do zbiorów walidacyjnego i testowego - zapobiega to wyciekowi danych (*data leakage*).

### Grupy cech (17 atrybutów)

**Grupa A - cechy statyczne profilu**

| Cecha | Opis |
|---|---|
| `account_age_days` | wiek konta w dniach względem daty referencyjnej 2020-09-01 |
| `ff_ratio` | followers_count / (following_count + 1) |
| `listed_count` | liczba list, na których widnieje użytkownik |
| `has_profile_image` | czy konto ma zdjęcie profilowe |
| `has_description` | czy konto ma opis |
| `has_url` | czy w profilu podany jest URL |
| `verified` | status weryfikacji konta |

**Grupa B - cechy behawioralne**

| Cecha | Opis |
|---|---|
| `tweets_per_day` | statuses_count / wiek konta |
| `retweet_ratio` | udział retweetów w aktywności |
| `reply_ratio` | udział odpowiedzi w aktywności |

**Grupa C - cechy semantyczne (analiza tekstu tweetów, VADER)**

| Cecha | Opis |
|---|---|
| `avg_sentiment` | średni wynik sentymentu compound (VADER) |
| `std_sentiment` | odchylenie standardowe sentymentu |
| `ttr` | Type-Token Ratio - bogactwo słownika |
| `avg_hashtags` | średnia liczba hashtagów na wpis |
| `avg_mentions` | średnia liczba wzmianek (@) na wpis |
| `url_post_ratio` | odsetek wpisów z co najmniej jednym URL |
| `duplication_ratio` | udział zduplikowanych tweetów |

## Metodologia i weryfikacja OSINT

Uzupełnieniem klasyfikacji automatycznej jest ręczna analiza próbki **79 unikalnych profili** ze zbioru testowego, podzielonej na 4 kategorie:

| Grupa | Kryterium | Cel |
|---|---|---|
| `pewny_bot` | label=1 & ai_prob > 0.90 | weryfikacja pewnych botów |
| `pewny_human` | label=0 & ai_prob < 0.10 | weryfikacja pewnych ludzi |
| `graniczny` | 0.40 ≤ ai_prob ≤ 0.60 | przypadki niejednoznaczne |
| `sprzeczność` | predykcja ≠ etykieta datasetu | analiza błędów klasyfikatora |

Ocena OSINT opiera się na systemie punktowym (Botometer X, reverse image search, Wayback Machine, Sherlock), wzorowanym na cechach botów opisanych przez Ferrarę i in. w *"The Rise of Social Bots"*. Suma punktów decyduje o etykiecie `osint_label`: `human` (≤0), `niejednoznaczny` (1–3), `bot` (≥4).

## Stos technologiczny

Projekt zaimplementowano w **Python 3.12.2**.

| Kategoria | Biblioteki | Zastosowanie |
|---|---|---|
| Przetwarzanie danych | `pandas`, `numpy` | manipulacja danymi, ekstrakcja cech |
| Postęp przetwarzania | `tqdm` | pasek postępu ekstrakcji cech |
| Analiza sentymentu | `vaderSentiment` | avg_sentiment, std_sentiment |
| Modelowanie (klasyczne ML) | `scikit-learn` | Logistic Regression, Random Forest, StandardScaler, metryki |
| Modelowanie (boosting) | `xgboost` | XGBClassifier z obsługą class imbalance |
| Wizualizacja | `matplotlib`, `seaborn` | macierze pomyłek, krzywe ROC, ważność cech |
| Przechowywanie danych | `pyarrow`, `joblib` | zapis Parquet, serializacja modeli |
| Ścieżki | `pathlib` | zarządzanie ścieżkami plików |

### Konfiguracja modeli

- **Logistic Regression** - `C=1.0`, `max_iter=1000`, `class_weight="balanced"`, `random_state=42`.
- **Random Forest** - `n_estimators=200`, `max_depth=15`, `class_weight="balanced"`, `n_jobs=-1`, `random_state=42`.
- **XGBoost** - `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=3`, `scale_pos_weight=neg/pos`, `early_stopping_rounds=50`.

## Wyniki modeli

| Model | Accuracy (val) | F1 (val) | ROC-AUC (val) | Accuracy (test) | F1 (test) | ROC-AUC (test) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8262 | 0.8518 | 0.8952 | 0.8276 | 0.8504 | 0.9002 |
| Random Forest | 0.8490 | 0.8720 | 0.9147 | 0.8529 | 0.8721 | 0.9211 |
| **XGBoost** | 0.8478 | 0.8671 | 0.9186 | **0.8555** | 0.8711 | **0.9230** |
| XGBoost (bez `verified`) | 0.8000 | 0.8186 | 0.8774 | 0.7921 | 0.8057 | 0.8782 |

Random Forest osiągnął najwyższy F1-score, a XGBoost - najwyższą dokładność i ROC-AUC. Wariant bez cechy `verified` wykazuje spadek wszystkich metryk, co potwierdza jej dominujący wpływ na klasyfikację.

### Macierz pomyłek (zbiór testowy)

| Model | TP | FN | FP | TN | Precision | Recall | Specificity |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 580 | 60 | 144 | 399 | 0.801 | 0.906 | 0.735 |
| Random Forest | 593 | 47 | 127 | 416 | 0.824 | 0.927 | 0.766 |
| XGBoost | 578 | 62 | 109 | 434 | 0.841 | 0.903 | 0.799 |
| XGBoost (bez `verified`) | 510 | 130 | 116 | 427 | 0.815 | 0.797 | 0.786 |

## Struktura repozytorium

```
fake-profiles-detector/
├── data/
│   ├── raw/                    # train.json, dev.json, test.json (TwiBot-20) - NIE w repo
│   └── processed/              # features_train/dev/test.parquet - NIE w repo
├── models/
│   ├── prepared_data.joblib    # NIE w repo (duży plik binarny)
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── lr_balanced.pkl         # NIE w repo (duży plik binarny)
│   ├── rf_balanced.pkl         # NIE w repo (duży plik binarny)
│   └── xgb_balanced.pkl        # NIE w repo (duży plik binarny)
├── notebooks/
│   └── check_features_train.ipynb
├── osint/
│   └── sample_to_check.csv     # próbka 79 profili do ręcznej weryfikacji OSINT
├── results/
│   ├── model_results.csv       # zbiorcze metryki wszystkich modeli
│   └── *.png                   # krzywe ROC, macierze pomyłek, ważność cech
├── data_loading.py
├── feature_extraction.py
├── build_features.py
├── train_prep.py
├── train_models.py
├── osint_sample_check.py
└── README.md
```

## Reprodukcja wyników

W repozytorium pozostawiono jedynie kod źródłowy, konfigurację, lekkie artefakty (`scaler.pkl`, `feature_names.pkl`) oraz pliki wynikowe w formacie CSV/PNG. Aby odtworzyć pełny pipeline, należy samodzielnie pozyskać dostęp do TwiBot-20 (kontakt z autorami benchmarku) i uruchomić skrypty w kolejności: `data_loading.py` → `feature_extraction.py`/`build_features.py` → `train_prep.py` → `train_models.py` → `osint_sample_check.py`.

## Ograniczenia

- Wysoka skuteczność modeli w dużej mierze zależy od cechy `verified`, która jest silnie niezbalansowana - większość profili jej nie posiada.
- Analiza sentymentu (VADER) bazuje na leksykonie anglojęzycznym, co obniża wiarygodność `avg_sentiment` i `std_sentiment` dla tweetów w innych językach.
- Zbiór TwiBot-20 pochodzi z września 2020 r. - część kont mogła zostać usunięta lub zawieszona, co ogranicza możliwości weryfikacji OSINT (np. w Wayback Machine).
- System punktowy OSINT to metoda pomocnicza opracowana na potrzeby pracy, nie jest zwalidowaną skalą diagnostyczną.

</details>
