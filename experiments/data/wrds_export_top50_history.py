"""
Second WRDS export (optional, for the 50-stock zero-shot experiment): daily returns of each
year's top-50 constituents over the LOOKBACK years BEFORE the selection year, so that their
mean and covariance can be estimated without look-ahead.

Run on WRDS Jupyter in the same folder as the first export (it reads
wrds_export/top50_by_year.csv) and upload wrds_top50_history.zip.
CRSP data are licensed: do not commit the output to a public repository.
"""
import os
import shutil

import pandas as pd
import wrds

LOOKBACK = 3
OUT = "wrds_top50_history"
os.makedirs(OUT, exist_ok=True)
db = wrds.Connection()
tabs = set(db.list_tables(library="crsp"))
daily, dcol, rcol = ("crsp.dsf_v2", "dlycaldt", "dlyret") if "dsf_v2" in tabs else ("crsp.dsf", "date", "ret")

top = pd.read_csv(os.path.join("wrds_export", "top50_by_year.csv"))
parts = []
for y, g in top.groupby("year"):
    pl = ", ".join(str(int(p)) for p in g.permno)
    df = db.raw_sql(f"select permno, {dcol} as date, {rcol} as ret from {daily} where permno in ({pl}) "
                    f"and {dcol} between '{y - LOOKBACK}-01-01' and '{y - 1}-12-31'", date_cols=["date"])
    df.insert(0, "year", y)
    parts.append(df)
    print(f"year {y}: {len(df):,} rows", flush=True)
pd.concat(parts, ignore_index=True).to_csv(os.path.join(OUT, "top50_history.csv"), index=False)
shutil.make_archive(OUT, "zip", OUT)
print(f"Done. Upload {OUT}.zip")
