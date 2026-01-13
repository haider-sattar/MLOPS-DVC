import json
import joblib
import pandas as pd
from pathlib import Path
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models/artifacts")
METRICS_PATH = Path("models/metrics.json")
PARAMS_PATH = Path("params.yml")


def load_params():
    with PARAMS_PATH.open() as f:
        return yaml.safe_load(f)

def train_model():
    logger.info("Starting model training")

    params = load_params()
    data_cfg = params["data"]
    model_cfg = params["model"]

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()

    X_val = pd.read_csv(PROCESSED_DIR / "X_val.csv")
    y_val = pd.read_csv(PROCESSED_DIR / "y_val.csv").squeeze()

    model = LogisticRegression(
        max_iter=model_cfg["max_iter"],
        solver=model_cfg["solver"],
        penalty=model_cfg.get("penalty", "l2"),
        C=model_cfg.get("C", 1.0),
        random_state=data_cfg.get("random_state"),
    )

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "f1_score": f1_score(y_val, y_val_pred),
        "roc_auc": roc_auc_score(y_val, y_val_proba),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "model.pkl")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Training completed | metrics=%s", metrics)

if __name__ == "__main__":
    train_model()
