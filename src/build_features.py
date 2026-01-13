import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_PATH = Path("data/raw/churn.csv")
PROCESSED_DIR = Path("data/processed")

def build_features(
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    logger.info("Starting feature engineering")

    df = pd.read_csv(RAW_DATA_PATH)

    # Basic cleaning
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # Target encoding
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    # Categorical encoding
    X = pd.get_dummies(X, drop_first=True)

    # 1. Split out test set (frozen)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 2. Split train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save train
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)

    # Save validation
    X_val.to_csv(PROCESSED_DIR / "X_val.csv", index=False)
    y_val.to_csv(PROCESSED_DIR / "y_val.csv", index=False)

    # Save test (frozen)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    logger.info(
        "Feature engineering completed | train=%d | val=%d | test=%d",
        X_train.shape[0],
        X_val.shape[0],
        X_test.shape[0],
    )

if __name__ == "__main__":
    build_features()
