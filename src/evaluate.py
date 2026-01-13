import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/artifacts/model.pkl")
TEST_METRICS_PATH = Path("models/test_metrics.json")

def evaluate_model():
    logger.info("Starting model evaluation on test set")

    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    with open(TEST_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Test evaluation completed | metrics=%s", metrics)

if __name__ == "__main__":
    evaluate_model()
