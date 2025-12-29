import os
import glob

# Project root = folder that contains /src
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Automatically find the CSV file inside /data
csv_files = glob.glob(os.path.join(PROJECT_ROOT, "data", "*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV file found in {os.path.join(PROJECT_ROOT, 'data')}"
    )

# Take the first CSV found (or the Telco one if multiple)
DATA_PATH = csv_files[0]

DB_PATH = os.path.join(PROJECT_ROOT, "outputs", "telco_churn.db")
RANDOM_STATE = 42
