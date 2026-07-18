#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Consolidated post-processing for the polypharmacy processed tables

Runs SIX idempotent, non-destructive steps against the files in
'Drugbank 3.0'. Safe to re-run at any time.

  1. Drop 'action' column from structured_drug_interactions_clean.csv
  2. Keep ONLY ['drugbank_id', 'food_interaction'] in food_with_halflife.csv
  3. Ingestion adjustment for drugs.csv  (whitelist includes 'description')
  4. Column positioning for drugs_processed.csv  ('description' before 'simple_description')
  5. Column positioning for drug_master.csv      ('description' before 'simple_description')
  6. Verification safeguard  (print columns + first 5 rows of every touched file)

Engineering guarantees

* NON-DESTRUCTIVE: existing half-life / food merge values are never recomputed.
  Target CSVs are read with dtype='string' so numeric cells are preserved verbatim.
* SAFE WRITES: atomic temp-file swap + timestamped .bak backup + lock-aware skip
  (a file open in Excel is reported, never corrupted). Retries once on a lock.
* IDEMPOTENT: a step that is already applied detects "no change" and does nothing
  (no needless backup churn).

"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from datetime import datetime
from typing import List, Optional

import pandas as pd


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("postprocess")


# Configuration with relative paths (100% PORTABLE)
DEFAULT_PROCESSED_DIR = r"Processed_outputs"
DEFAULT_DRUGS_CSV = r"Raw_data\drugs.csv"

# ingest whitelist for drugs.csv (now includes the detailed text).
DRUGS_KEEP: List[str] = ["drugbank_id", "name", "description", "simple_description"]

# Explicit required output orders.
ORDER_DRUGS_PROCESSED: List[str] = ["drugbank_id", "name", "description", "simple_description"]
ORDER_DRUG_MASTER: List[str] = [
    "drugbank_id", "name", "description", "simple_description",
    "half_life", "half_life_unit", "half_life_unit_mixed", "half_life_n_records",
    "half_life_hours", "food_interaction_count", "food_interactions_text",
]


def _resolve(default: str, *fallbacks: str) -> str:
    """Return the first existing path among default/fallbacks, else the default."""
    for p in (default, *fallbacks):
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return default


# Shared robustness helpers: atomic, lock-aware, reversible writes #

def _safe_remove(path: str) -> None:
    """Remove a file if present, ignoring errors (temp/backup cleanup)."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _atomic_write(new_df: pd.DataFrame, path: str, backup: bool = True, retries: int = 1) -> bool:
    """
    Overwrite `path` with `new_df` atomically and reversibly:
      * write to a sibling .tmp first (a temp file is never locked),
      * snapshot the original to a timestamped .bak,
      * os.replace() the temp over the target (atomic on the same volume).
    Returns False WITHOUT raising if the target is locked,
    so the caller can keep processing the other files. Retries once for transient locks.
    """
    tmp = f"{path}.tmp"
    bak = f"{path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
    for attempt in range(retries + 1):
        try:
            new_df.to_csv(tmp, index=False)          # always safe: temp file
            if backup:
                shutil.copy2(path, bak)              # snapshot before the swap
            os.replace(tmp, path)                    # atomic; raises if path locked
            if backup:
                logger.info("  backup -> %s", os.path.basename(bak))
            return True
        except PermissionError:
            _safe_remove(tmp)
            _safe_remove(bak)                        # nothing changed -> drop snapshot
            if attempt < retries:
                logger.warning("  target locked, retrying in 1s ...")
                time.sleep(1)
                continue
            logger.error("  LOCKED - close the file (open in Excel?) and re-run.")
            return False
        except Exception as exc:  # noqa: BLE001 - report, clean up, continue
            _safe_remove(tmp)
            logger.error("  write failed (%s)", exc)
            return False
    return False


# Column operations #

def trim_csv(path: str, *, drop: Optional[List[str]] = None,
             keep: Optional[List[str]] = None) -> bool:
    """
    Reads dtype='string' to preserve values verbatim. Idempotent.
    """
    if (drop is None) == (keep is None):
        raise ValueError("Provide exactly one of drop= or keep=")
    label = os.path.basename(path)
    if not os.path.exists(path):
        logger.error("SKIP %s - file not found", label)
        return False
    try:
        df = pd.read_csv(path, dtype="string")
    except Exception as exc:  # noqa: BLE001
        logger.error("SKIP %s - could not read (%s)", label, exc)
        return False

    original_cols = list(df.columns)
    if keep is not None:
        present = [c for c in keep if c in df.columns]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            logger.warning("  %s - requested KEEP columns absent: %s", label, missing)
        if not present:
            logger.error("SKIP %s - none of the KEEP columns exist", label)
            return False
        new_df = df[present]                          # list-index enforces order
    else:
        present = [c for c in drop if c in df.columns]
        if [c for c in drop if c not in df.columns]:
            logger.info("  %s - DROP columns already absent (idempotent)", label)
        new_df = df.drop(columns=present) if present else df

    if list(new_df.columns) == original_cols:
        logger.info("  %s - already in desired state, no change", label)
        return True
    if not _atomic_write(new_df, path):
        return False
    logger.info("  %s - %s -> %s | columns: %s",
                label, (len(original_cols),), new_df.shape, list(new_df.columns))
    return True


def load_description_lookup(drugs_csv: str) -> pd.DataFrame:
    """
    Ingestion adjustment: read the DRUGS_KEEP whitelist from drugs.csv
    (text as 'string') and return a {drugbank_id -> description} lookup.
    """
    if not os.path.exists(drugs_csv):
        raise FileNotFoundError(f"Source drugs file not found: {drugs_csv}")
    df = pd.read_csv(drugs_csv, usecols=DRUGS_KEEP, dtype={c: "string" for c in DRUGS_KEEP})
    df["drugbank_id"] = df["drugbank_id"].str.strip()
    lut = (df[["drugbank_id", "description"]]
           .dropna(subset=["drugbank_id"])
           .drop_duplicates("drugbank_id", keep="first"))
    logger.info("  loaded whitelist %s | description lookup: %s ids (%s non-null)",
                DRUGS_KEEP, f"{len(lut):,}", f"{int(lut['description'].notna().sum()):,}")
    return lut


def add_description(path: str, desired_order: List[str], lut: pd.DataFrame) -> bool:
    """
    add 'description' (matched on drugbank_id) and enforce the exact
    `desired_order` so 'description' sits right before 'simple_description'.
    Non-destructive (dtype='string') and idempotent.
    """
    label = os.path.basename(path)
    if not os.path.exists(path):
        logger.error("SKIP %s - file not found", label)
        return False
    try:
        original = pd.read_csv(path, dtype="string")
    except Exception as exc:  # noqa: BLE001
        logger.error("SKIP %s - could not read (%s)", label, exc)
        return False
    if "drugbank_id" not in original.columns:
        logger.error("SKIP %s - no 'drugbank_id' to join on", label)
        return False

    # drop any existing 'description' so a re-run replaces (not duplicates) it
    work = original.drop(columns=["description"]) if "description" in original.columns else original
    merged = work.merge(lut, on="drugbank_id", how="left")

    ordered = [c for c in desired_order if c in merged.columns]
    extras = [c for c in merged.columns if c not in desired_order]
    if extras:
        logger.warning("  %s - extra columns kept at end: %s", label, extras)
    merged = merged[ordered + extras]

    # idempotency: skip write if the on-disk file is already exactly this
    if (list(merged.columns) == list(original.columns)
            and merged.reset_index(drop=True).equals(original[merged.columns].reset_index(drop=True))):
        logger.info("  %s - already in desired state, no change", label)
        return True

    n_desc = int(merged["description"].notna().sum())
    if not _atomic_write(merged, path):
        return False
    logger.info("  %s - columns -> %d | %s/%s rows have a description",
                label, len(merged.columns), f"{n_desc:,}", f"{len(merged):,}")
    return True



# Step 6 - verification safeguard                                              #

def verify_file(path: str, before_after: str = "") -> None:
    """Print the on-disk column list + first 5 rows (long text truncated)."""
    name = os.path.basename(path)
    print("\n" + "=" * 82)
    print(f"VERIFY: {name}   {before_after}")
    print("=" * 82)
    if not os.path.exists(path):
        print("  (file not found)")
        return
    df = pd.read_csv(path, dtype="string")
    cols = list(df.columns)
    print("Columns:", cols)
    # targeted positional assertion for the description tables
    if "description" in cols and "simple_description" in cols:
        di, si = cols.index("description"), cols.index("simple_description")
        print(f"'description' idx {di}, 'simple_description' idx {si} "
              f"-> description immediately before: {di == si - 1}")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.max_colwidth", 38):
        print("\nFirst 5 rows:\n", df.head(5).to_string(index=False))


def main() -> int:
    processed_dir = _resolve(
        os.environ.get("POLYPHARMACY_OUT", DEFAULT_PROCESSED_DIR),
        os.path.join(os.getcwd(), "processed_polypharmacy"),
        "processed_polypharmacy",
    )
    drugs_csv = _resolve(os.environ.get("DRUGS_CSV", DEFAULT_DRUGS_CSV))
    P = lambda f: os.path.join(processed_dir, f)

    logger.info("=" * 75)
    logger.info("POLYPHARMACY POST-PROCESS | dir=%s", processed_dir)
    logger.info("=" * 75)
    if not os.path.isdir(processed_dir):
        logger.error("Directory not found: %s", processed_dir)
        return 1

    ok = True

    
    # 1. Drop 'action' from structured_drug_interactions_clean.csv
    
    logger.info("[1] drop 'action' from structured_drug_interactions_clean.csv")
    ok &= trim_csv(P("structured_drug_interactions_clean.csv"), drop=["action"])

    
    # 2. Keep ONLY drugbank_id + food_interaction in food_with_halflife.csv
    #    (NB: this makes the file identical to food_interactions_processed.csv)
    
    logger.info("[2] keep ONLY [drugbank_id, food_interaction] in food_with_halflife.csv")
    ok &= trim_csv(P("food_with_halflife.csv"), keep=["drugbank_id", "food_interaction"])

    
    # 3. Ingestion adjustment for drugs.csv (whitelist now carries 'description')
    
    logger.info("[3] ingestion adjustment for drugs.csv (load 'description')")
    try:
        desc_lut = load_description_lookup(drugs_csv)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    
    # 4. Column positioning for drugs_processed.csv
    
    logger.info("[4] position 'description' before 'simple_description' in drugs_processed.csv")
    ok &= add_description(P("drugs_processed.csv"), ORDER_DRUGS_PROCESSED, desc_lut)


    # 5. Column positioning for drug_master.csv
    
    logger.info("[5] position 'description' before 'simple_description' in drug_master.csv")
    ok &= add_description(P("drug_master.csv"), ORDER_DRUG_MASTER, desc_lut)

    
    # 6. Verification safeguard - show the resulting on-disk structure
    
    logger.info("[6] verification safeguard")
    verify_file(P("structured_drug_interactions_clean.csv"), "(step 1: 'action' removed)")
    verify_file(P("food_with_halflife.csv"), "(step 2: 2 columns only)")
    verify_file(P("drugs_processed.csv"), "(step 4)")
    verify_file(P("drug_master.csv"), "(step 5)")

    logger.info("=" * 75)
    logger.info("POST-PROCESS FINISHED%s.", "" if ok else " (one or more files skipped - see log)")
    logger.info("=" * 75)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
