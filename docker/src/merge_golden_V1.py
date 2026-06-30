#!/usr/bin/env python3
import os
import pickle
import pandas as pd
from pathlib import Path
from config import DB_PATH, GOLDEN_DIR
import sqlite3
from schema import ARTICLE_COLUMNS
from tqdm import tqdm


# ── Golden Papers laden ──────────────────────────────────────────
with open(GOLDEN_DIR / "golden69_papers.pkl", "rb") as f:
    df_golden = pickle.load(f)
print(f"Golden Papers: {len(df_golden)}")

df_golden = df_golden[ARTICLE_COLUMNS]

# Tables-Liste in String wandeln, falls vorhanden
if 'tables' in df_golden.columns:
    df_golden['tables'] = df_golden['tables'].apply(
        lambda x: " \n\n---NEW_TABLE---\n\n ".join(x) if isinstance(x, list) else str(x)
    )

# In SQLite einfügen (Upsert)
conn = sqlite3.connect(str(DB_PATH))
old_len = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
col_defs = ", ".join([f'"{col}" TEXT' for col in ARTICLE_COLUMNS])
conn.execute(f"CREATE TABLE IF NOT EXISTS articles ({col_defs}, PRIMARY KEY (pmcid))")

placeholders = ", ".join(["?"] * len(ARTICLE_COLUMNS))
for row in df_golden.itertuples(index=False, name=None):
    conn.execute(f"INSERT OR REPLACE INTO articles VALUES ({placeholders})", row)

conn.commit()
new_total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

conn.close()

added = new_total - old_len
# updated = len(df_golden) - added   # entfällt, da wir nur hinzufügen

tqdm.write(f"✅ Goldene Artikel in die Datenbank eingefügt.")
tqdm.write(f"   Artikel vor diesem Lauf : {old_len}")
tqdm.write(f"   Goldene Artikel geladen : {len(df_golden)}")
tqdm.write(f"   ── davon neu hinzugefügt: {added}")
tqdm.write(f"   Gesamt in DB jetzt      : {new_total}")
tqdm.write(f"   Pfad: {DB_PATH}")
host_dir = os.environ.get("HOST_OUTPUT_DIR", "Host-Pfad unbekannt")
tqdm.write(f"   Host-Pfad: {host_dir}/database.db")

print(f"✅ {added} goldene Artikel in die Datenbank eingefügt/aktualisiert.")