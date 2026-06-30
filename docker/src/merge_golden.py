#!/usr/bin/env python3
"""
merge_golden.py – Add golden question papers into the SQLite article database.
Input:  /data/golden_data/golden69_papers.pkl
Output: /data/database/database.db (updated in-place)
"""

import os
import pickle
import pandas as pd
from pathlib import Path
import sqlite3
from config import DB_PATH, GOLDEN_DIR
from schema import ARTICLE_COLUMNS
from tqdm import tqdm

# =============================================================================
# LOAD GOLDEN PAPERS
# =============================================================================
with open(GOLDEN_DIR / "golden69_papers.pkl", "rb") as f:
    df_golden = pickle.load(f)
print(f"Loaded {len(df_golden)} golden papers from {GOLDEN_DIR / 'golden69_papers.pkl'}")

# Keep only the columns that match the database schema
df_golden = df_golden[ARTICLE_COLUMNS]

# =============================================================================
# PREPARE DATA FOR DATABASE
# =============================================================================
# Convert any list (e.g. tables) to a flat string for SQLite TEXT columns
if 'tables' in df_golden.columns:
    df_golden['tables'] = df_golden['tables'].apply(
        lambda x: " \n\n---NEW_TABLE---\n\n ".join(x) if isinstance(x, list) else str(x)
    )
    print("Converted 'tables' list to string for storage.")

# =============================================================================
# UPSERT INTO SQLITE
# =============================================================================
conn = sqlite3.connect(str(DB_PATH))

# Ensure the table exists
col_defs = ", ".join([f'"{col}" TEXT' for col in ARTICLE_COLUMNS])
conn.execute(f"CREATE TABLE IF NOT EXISTS articles ({col_defs}, PRIMARY KEY (pmcid))")

# Count existing articles
old_len = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
print(f"Articles in database before merge: {old_len}")

# Insert or replace each golden paper
placeholders = ", ".join(["?"] * len(ARTICLE_COLUMNS))
for row in df_golden.itertuples(index=False, name=None):
    conn.execute(f"INSERT OR REPLACE INTO articles VALUES ({placeholders})", row)

conn.commit()
new_total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
conn.close()

added = new_total - old_len  # truly new papers added
# updated = len(df_golden) - added  # not needed; only new ones matter here

# =============================================================================
# SUMMARY
# =============================================================================
tqdm.write(f"Golden papers merged successfully.")
tqdm.write(f"   Articles before merge  : {old_len}")
tqdm.write(f"   Golden papers loaded   : {len(df_golden)}")
tqdm.write(f"   Newly added            : {added}")
tqdm.write(f"   Total articles now     : {new_total}")
tqdm.write(f"   Database path (in container): {DB_PATH}")
host_dir = os.environ.get("HOST_OUTPUT_DIR", "Host path unknown")
tqdm.write(f"   Host-side path: {host_dir}/database.db")
print(f"{added} new golden papers added / updated in the database.")