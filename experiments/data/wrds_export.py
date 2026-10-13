"""
Export the CRSP / Fama-French data used by the empirical study from WRDS.

Run on WRDS (Jupyter Hub on WRDS Cloud: `wrds.Connection()` authenticates automatically) or
locally with `pip install wrds` (asks for your WRDS username and password the first time):

    python wrds_export.py            # or paste into a notebook cell and run

Works with both CRSP formats: the legacy SIZ tables (crsp.dsi, crsp.dsf, ...) and the new
CIZ tables (crsp.dsf_v2, crsp.inddlyseriesdata, ...). It inspects which tables and columns
exist and uses those, and it writes schema.txt so that a failure can be diagnosed.

Output: wrds_export/ and wrds_export.zip containing
    ff_daily.csv            date, mktrf, rf (Fama-French daily factors)
    market_index.csv        date, vwretd (CRSP value-weighted, with dividends), sprtrn (S&P 500)
                            [SIZ]  or  index_header.csv + index_daily.csv  [CIZ: candidate series]
    sector_etf_daily.csv    permno, ticker, date, ret  (9 Select Sector SPDRs, 1998-12 onward)
    top50_by_year.csv       year, permno, mcap  (50 largest common stocks at the end of year-1)
    top50_daily.csv         permno, date, ret   (each stock only in the year it is selected)
    top50_delist.csv        permno, dlstdt, dlret  [SIZ only; CIZ returns already include delisting]
    schema.txt              tables and columns that were found

CRSP data are licensed: do not commit the output to a public repository.
"""
import os
import shutil
import traceback

import pandas as pd
import wrds

START = "1962-01-01"
START_STOCKS = 1990                     # first year of the top-50 universe
ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
OUT = "wrds_export"
os.makedirs(OUT, exist_ok=True)
db = wrds.Connection()
LOG = open(os.path.join(OUT, "schema.txt"), "w")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n"); LOG.flush()


_tables = {}


def tables(lib):
    if lib not in _tables:
        try:
            _tables[lib] = set(db.list_tables(library=lib))
        except Exception:
            _tables[lib] = set()
    return _tables[lib]


def has(lib, tab):
    return tab in tables(lib)


def columns(lib, tab):
    d = db.describe_table(library=lib, table=tab)
    cols = [c.lower() for c in d["name"]]
    log(f"  {lib}.{tab}: {', '.join(cols)}")
    return cols


def pick(cols, *cands, required=True):
    for c in cands:
        if c in cols:
            return c
    if required:
        raise KeyError(f"none of {cands} in columns")
    return None


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)
    log(f"wrote {name}: {len(df):,} rows")


def step(name, fn):
    log(f"\n=== {name} ===")
    try:
        fn()
    except Exception:
        log("FAILED:\n" + traceback.format_exc())


# ---------------------------------------------------------------- 1. Fama-French daily
def ff():
    lib, tab = ("ff", "factors_daily") if has("ff", "factors_daily") else ("ff_all", "factors_daily")
    cols = columns(lib, tab)
    d = pick(cols, "date", "dateff")
    save(db.raw_sql(f"select {d} as date, mktrf, rf from {lib}.{tab} where {d} >= '{START}' order by 1",
                    date_cols=["date"]), "ff_daily.csv")


# ---------------------------------------------------------------- 2. market index
def market():
    if has("crsp", "dsi"):
        cols = columns("crsp", "dsi")
        sel = ", ".join(c for c in ["vwretd", "vwretx", "ewretd", "sprtrn", "spindx"] if c in cols)
        save(db.raw_sql(f"select date, {sel} from crsp.dsi where date >= '{START}' order by date",
                        date_cols=["date"]), "market_index.csv")
        return
    # CIZ: dump the index header (small) and the daily series of plausible candidates
    hdr_tab = next(t for t in ["indseriesinfohdr", "indseriesinfohdr_v2"] if has("crsp", t))
    hcols = columns("crsp", hdr_tab)
    hdr = db.raw_sql(f"select * from crsp.{hdr_tab}")
    save(hdr, "index_header.csv")
    name_col = pick(hcols, "indname", "indexname", "indnm")
    m = hdr[name_col].str.contains(r"Value-Weighted|Value Weighted|S&P 500|S&P Composite", case=False, na=False)
    codes = sorted(hdr.loc[m, "indno"].dropna().astype(int).unique().tolist())
    log(f"candidate index series: {len(codes)}")
    dat_tab = next(t for t in ["inddlyseriesdata", "inddlyseriesdata_v2"] if has("crsp", t))
    dcols = columns("crsp", dat_tab)
    d = pick(dcols, "dlycaldt", "caldt", "date")
    keep = [c for c in dcols if c in ("indno", d) or "ret" in c or c in ("dlytotret", "dlyprcret", "dlyretx")]
    save(db.raw_sql(f"select {', '.join(keep)} from crsp.{dat_tab} where indno in ({', '.join(map(str, codes))}) "
                    f"and {d} >= '{START}' order by indno, {d}"), "index_daily.csv")


# ---------------------------------------------------------------- stock-file helpers
def stock_tables():
    """Return (daily, monthly, names) table descriptors for whichever format exists."""
    if has("crsp", "dsf_v2") and has("crsp", "msf_v2"):
        dc, mc = columns("crsp", "dsf_v2"), columns("crsp", "msf_v2")
        daily = dict(tab="crsp.dsf_v2", date=pick(dc, "dlycaldt", "date"), ret=pick(dc, "dlyret", "ret"))
        monthly = dict(tab="crsp.msf_v2", date=pick(mc, "mthcaldt", "date"),
                       cap=pick(mc, "mthcap", required=False), prc=pick(mc, "mthprc", "prc"),
                       shr=pick(mc, "shrout", "mthshrout", required=False), cols=mc)
        ntab = next(t for t in ["stksecurityinfohist", "stksecurityinfohist_v2"] if has("crsp", t))
        nc = columns("crsp", ntab)
        names = dict(tab=f"crsp.{ntab}", start=pick(nc, "secinfostartdt", "namedt"),
                     end=pick(nc, "secinfoenddt", "nameendt"), cols=nc, fmt="CIZ")
        return daily, monthly, names
    dc, mc, nc = columns("crsp", "dsf"), columns("crsp", "msf"), columns("crsp", "dsenames")
    return (dict(tab="crsp.dsf", date="date", ret="ret"),
            dict(tab="crsp.msf", date="date", cap=None, prc="prc", shr="shrout", cols=mc),
            dict(tab="crsp.dsenames", start="namedt", end="nameendt", cols=nc, fmt="SIZ"))


def common_stock_filter(names, p="n."):
    """Ordinary common shares on NYSE / AMEX / NASDAQ, columns qualified with prefix p."""
    c = names["cols"]
    if names["fmt"] == "SIZ":
        return f"{p}shrcd in (10, 11) and {p}exchcd in (1, 2, 3)"
    conds = []
    if "sharetype" in c: conds.append(f"{p}sharetype = 'NS'")
    if "securitytype" in c: conds.append(f"{p}securitytype = 'EQTY'")
    if "securitysubtype" in c: conds.append(f"{p}securitysubtype = 'COM'")
    if "usincflg" in c: conds.append(f"{p}usincflg = 'Y'")
    if "primaryexch" in c: conds.append(f"{p}primaryexch in ('N', 'A', 'Q')")
    return " and ".join(conds) or "true"


# ---------------------------------------------------------------- 3. sector ETFs
def etfs():
    daily, _, names = stock_tables()
    tl = ", ".join(f"'{t}'" for t in ETFS)
    ids = db.raw_sql(f"select distinct permno, ticker from {names['tab']} where ticker in ({tl})")
    log(f"ETF permnos: {ids.to_dict('records')}")
    pl = ", ".join(str(int(p)) for p in ids.permno.unique())
    df = db.raw_sql(f"select permno, {daily['date']} as date, {daily['ret']} as ret from {daily['tab']} "
                    f"where permno in ({pl}) and {daily['date']} >= '1998-12-01' order by permno, 2",
                    date_cols=["date"])
    # one permno per ticker: the one with the longest history
    n = df.groupby("permno").size().rename("n").reset_index().merge(ids, on="permno")
    best = n.sort_values("n", ascending=False).drop_duplicates("ticker")
    save(df.merge(best[["permno", "ticker"]], on="permno")[["permno", "ticker", "date", "ret"]], "sector_etf_daily.csv")


# ---------------------------------------------------------------- 4. top-50 universe
def top50():
    daily, monthly, names = stock_tables()
    cap = monthly["cap"] or f"abs({monthly['prc']}) * {monthly['shr']}"
    md = monthly["date"]
    q = f"""select m.permno, m.{md} as date, {cap} as mcap
            from {monthly['tab']} m join {names['tab']} n on m.permno = n.permno
              and m.{md} between n.{names['start']} and n.{names['end']}
            where extract(month from m.{md}) = 12 and m.{md} >= '{START_STOCKS - 1}-12-01'
              and {common_stock_filter(names)}"""
    m = db.raw_sql(q, date_cols=["date"]).dropna(subset=["mcap"])
    m["year"] = m["date"].dt.year + 1
    top = (m.sort_values("mcap", ascending=False).drop_duplicates(["year", "permno"])
             .groupby("year").head(50)[["year", "permno", "mcap"]])
    save(top, "top50_by_year.csv")
    parts = []
    for y, g in top.groupby("year"):
        pl = ", ".join(str(int(p)) for p in g.permno)
        parts.append(db.raw_sql(
            f"select permno, {daily['date']} as date, {daily['ret']} as ret from {daily['tab']} "
            f"where permno in ({pl}) and {daily['date']} between '{y}-01-01' and '{y}-12-31'", date_cols=["date"]))
        log(f"  year {y}: {len(parts[-1]):,} rows")
    save(pd.concat(parts, ignore_index=True), "top50_daily.csv")
    if names["fmt"] == "SIZ" and has("crsp", "dsedelist"):
        pl = ", ".join(str(int(p)) for p in top.permno.unique())
        save(db.raw_sql(f"select permno, dlstdt, dlret from crsp.dsedelist where permno in ({pl})"), "top50_delist.csv")


step("Fama-French daily factors", ff)
step("Market index", market)
step("Sector ETFs", etfs)
step("Top-50 universe", top50)
LOG.close()
shutil.make_archive(OUT, "zip", OUT)
print(f"\nDone. Upload {OUT}.zip (and check {OUT}/schema.txt if any step FAILED).")
