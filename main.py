from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from build_features import build_features_for_split
from train_prep import main as prep_main
from train_models import main as train_main
from osint_sample_check import main as osint_main


if __name__ == "__main__":
    print("=" * 55)
    print("KROK 1/4 – Budowanie cech (train / dev / test)")
    print("=" * 55)
    build_features_for_split("train")
    build_features_for_split("dev")
    build_features_for_split("test")

    print("\n" + "=" * 55)
    print("KROK 2/4 – Skalowanie i zapis danych treningowych")
    print("=" * 55)
    prep_main()

    print("\n" + "=" * 55)
    print("KROK 3/4 – Trening modeli (LR / RF / XGBoost)")
    print("=" * 55)
    train_main()

    print("\n" + "=" * 55)
    print("KROK 4/4 – Generowanie próbki OSINT")
    print("=" * 55)
    osint_main()

    print("\n Pipeline zakończony!")
    print("  data/processed/ → pliki parquet")
    print("  models/         → wytrenowane modele")
    print("  results/        → wykresy i tabela wyników")
    print("  osint/          → sample_to_check.csv")



    
