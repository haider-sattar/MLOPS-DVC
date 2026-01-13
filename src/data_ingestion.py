import pandas as pd
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_PATH = Path("data/raw/churn.csv")

def ingest_data():
    logger.info("Starting data ingestion")

    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

    df = pd.read_csv(url)

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)

    logger.info(
        "Data ingestion completed | rows=%d | cols=%d",
        df.shape[0],
        df.shape[1]
    )

if __name__ == "__main__":
    ingest_data()
