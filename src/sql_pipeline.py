import os
import sqlite3
import pandas as pd

def build_sqlite_db(df: pd.DataFrame, db_path: str, table_name: str = "telco_churn") -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    if os.path.exists(db_path):
        os.remove(db_path)

    con = sqlite3.connect(db_path)
    df.to_sql(table_name, con, index=False, if_exists="replace")

    cur = con.cursor()
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_churn ON {table_name}(Churn)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_contract ON {table_name}(Contract)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_internet ON {table_name}(InternetService)")
    con.commit()
    con.close()

def run_sql_query(db_path: str, query: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    out = pd.read_sql_query(query, con)
    con.close()
    return out
