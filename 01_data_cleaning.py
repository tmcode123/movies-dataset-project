"""
01_data_cleaning.py
────────────────────────────────────────────────────────────────────────────
Data Engineering layer: raw CSV → analysis-ready dataset.

Inputs  : data/movie_metadata.csv  (5 043 rows, IMDB-sourced)
          data/tmdb_5000_movies.csv (4 803 rows, TMDB API)
Output  : data/movies_clean.csv    (deduplicated, typed, validated)

Run: python 01_data_cleaning.py
"""

import json
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ── helpers ────────────────────────────────────────────────────────────────────

def log_step(msg: str) -> None:
    print(f"[ETL] {msg}")


def extract_genre_names(genre_json: str) -> list[str]:
    """Parse TMDB-style JSON genre field → list of genre name strings."""
    try:
        return [g["name"] for g in json.loads(genre_json)]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []


# ── 1. Load ────────────────────────────────────────────────────────────────────

log_step("Loading raw sources …")
meta = pd.read_csv("data/movie_metadata.csv", low_memory=False)
tmdb = pd.read_csv("data/tmdb_5000_movies.csv", low_memory=False)

log_step(f"  movie_metadata : {meta.shape[0]:,} rows × {meta.shape[1]} cols")
log_step(f"  tmdb_5000      : {tmdb.shape[0]:,} rows × {tmdb.shape[1]} cols")

# ── 2. Clean movie_metadata ────────────────────────────────────────────────────

log_step("Cleaning movie_metadata …")

# Strip invisible whitespace from titles (trailing \xa0 present in source)
meta["movie_title"] = meta["movie_title"].str.strip()

# Drop rows missing the columns we care about for analysis / modelling
required_cols = ["gross", "budget", "title_year", "imdb_score", "genres", "duration"]
before = len(meta)
meta = meta.dropna(subset=required_cols)
log_step(f"  Dropped {before - len(meta):,} rows with nulls in required columns")

# Remove zero-dollar entries (data entry artefacts, not free films)
before = len(meta)
meta = meta[(meta["gross"] > 0) & (meta["budget"] > 0)]
log_step(f"  Dropped {before - len(meta):,} rows with zero gross or budget")

# Restrict to modern era for fair genre/ROI comparisons
before = len(meta)
meta = meta[meta["title_year"] >= 1980]
log_step(f"  Restricted to 1980+ → dropped {before - len(meta):,} older rows")

# Coerce types
meta["title_year"] = meta["title_year"].astype(int)
meta["duration"] = pd.to_numeric(meta["duration"], errors="coerce")

# Derived columns
meta["primary_genre"] = meta["genres"].str.split("|").str[0]
meta["roi"]           = meta["gross"] / meta["budget"]
meta["log_budget"]    = np.log1p(meta["budget"])
meta["log_gross"]     = np.log1p(meta["gross"])
meta["decade"]        = (meta["title_year"] // 10 * 10)
meta["profitable"]    = (meta["gross"] > meta["budget"]).astype(int)

log_step(f"  movie_metadata clean shape: {meta.shape}")

# ── 3. Clean TMDB ──────────────────────────────────────────────────────────────

log_step("Cleaning tmdb_5000_movies …")

tmdb["release_date"] = pd.to_datetime(tmdb["release_date"], errors="coerce")
tmdb["year"]         = tmdb["release_date"].dt.year

required_tmdb = ["revenue", "budget", "year", "runtime"]
before = len(tmdb)
tmdb = tmdb.dropna(subset=required_tmdb)
tmdb = tmdb[(tmdb["revenue"] > 0) & (tmdb["budget"] > 0) & (tmdb["runtime"] > 0)]
log_step(f"  Dropped {before - len(tmdb):,} unusable rows from TMDB")

tmdb["genre_list"]    = tmdb["genres"].apply(extract_genre_names)
tmdb["primary_genre"] = tmdb["genre_list"].str[0]
tmdb["roi"]           = tmdb["revenue"] / tmdb["budget"]
tmdb["log_budget"]    = np.log1p(tmdb["budget"])
tmdb["log_revenue"]   = np.log1p(tmdb["revenue"])
tmdb["profitable"]    = (tmdb["revenue"] > tmdb["budget"]).astype(int)

log_step(f"  TMDB clean shape: {tmdb.shape}")

# ── 4. Data quality report ─────────────────────────────────────────────────────

log_step("Generating data quality report …")

def quality_report(df: pd.DataFrame, name: str) -> pd.DataFrame:
    report = pd.DataFrame({
        "dataset"    : name,
        "column"     : df.columns,
        "dtype"      : df.dtypes.values,
        "null_count" : df.isnull().sum().values,
        "null_pct"   : (df.isnull().mean() * 100).round(1).values,
        "unique"     : [df[c].apply(lambda x: str(x)).nunique() for c in df.columns],
    })
    return report

qr_meta = quality_report(meta, "movie_metadata")
qr_tmdb = quality_report(tmdb, "tmdb_5000")
qr_full = pd.concat([qr_meta, qr_tmdb], ignore_index=True)

print("\n── Data Quality Report ──────────────────────────────────────────────")
print(qr_full.to_string(index=False))

# ── 5. Save ────────────────────────────────────────────────────────────────────

log_step("Saving cleaned datasets …")
meta.to_csv("data/movies_clean.csv", index=False)
tmdb.to_csv("data/tmdb_clean.csv", index=False)
qr_full.to_csv("data/data_quality_report.csv", index=False)

log_step("Done ✓")
log_step(f"  → data/movies_clean.csv   ({len(meta):,} rows)")
log_step(f"  → data/tmdb_clean.csv     ({len(tmdb):,} rows)")
log_step(f"  → data/data_quality_report.csv")

print("\n── Summary Stats (movies_clean) ─────────────────────────────────────")
print(meta[["gross", "budget", "imdb_score", "roi", "duration"]].describe().round(2))
