#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 3 - Relational Data Enrichment & Multi-Modal Schema Assembly
"MAKING POLYPHARMACY SAFER: AN AI-BASED TRANSLATIVE MODEL FOR DRUG INTERACTION
 RISK DETECTION AND PATIENT GUIDANCE"

Deterministic knowledge-base synthesis: maps standalone drug
attributes from `drug_master.csv` onto BOTH ends of every interaction pair in
`structured_drug_interactions_clean.csv`, producing a single self-contained
multi-modal text corpus: `final_nlp_interaction_corpus.csv`.


DATA NOTE - severity encoding (verified against the real data)

Rule described severity as the strings 'minor'/'moderate'/'major', but in this
dataset `severity` is INTEGER-encoded (0/1/2). The integer->label mapping below
was confirmed empirically against the dataset overview in the cohort:
    2 == major    (Phenelzine[MAOI]+SSRI serotonin syndrome; Apixaban+Warfarin;
                   Methotrexate+NSAID  -> all severity 2)
    1 == moderate (Liraglutide+Insulin; Clopidogrel+Apixaban  -> the bulk, 1353)
    0 == minor    (Pseudoephedrine blunting antihypertensives -> 210)
The mapper is defensive: it accepts either the integer code OR an already-string
severity, so the script is correct whichever form the column takes.

"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from datetime import datetime
from typing import List

import pandas as pd

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("nlp_corpus")


# Configuration (override via env vars if paths differ)
 
# Configuration - with relative paths (100% PORTABLE)
DEFAULT_PROCESSED_DIR = r"Processed_outputs"

PROCESSED_DIR = os.environ.get("POLYPHARMACY_OUT", DEFAULT_PROCESSED_DIR)

INTERACTIONS_FILE = "structured_drug_interactions_clean.csv"
MASTER_FILE = "drug_master.csv"
OUTPUT_FILE = "final_nlp_interaction_corpus.csv"

# Severity integer code -> canonical clinical label (verified).
SEVERITY_INT_TO_LABEL = {0: "minor", 1: "moderate", 2: "major"}

# Rule - canonical label -> high-visibility patient-facing risk level.
# NB: the 'major' string keeps the exact trailing space given.
RISK_LEVEL_MAP = {
    "minor": "observe and adjust",
    "moderate": "adjustment should be considered, consult your physician/GP",
    "major": "combination should be avoided, consult your GP/physician for a review ",
}

# Evidence-level integer code -> clinical definition (verified: corpus has 1/2 only).
EVIDENCE_MAP = {
    1: "Mentioned in the drug monograph (FDA, Health Canada, EMA, etc.) and has "
       "been confirmed in clinical studies (cohort, case-control, case study etc.).",
    2: "Has been confirmed in at least 1 cohort, case-control, or case study and "
       "may or may not be mentioned in a drug monograph.",
}

# Only these attributes are pulled from the master for each side of the pair.
PROFILE_COLS = ["drugbank_id", "description", "simple_description", "half_life_hours"]

# Final, explicit column order of the corpus (keys kept for a relational KB).
FINAL_COLS: List[str] = [
    # --- foreign keys (self-contained relational knowledge base) ---
    "subject_drug_drugbank_id", "affected_drug_drugbank_id",
    # --- Interaction Core ---
    "subject_drug_name", "affected_drug_name",
    "severity", "severity_label", "risk_level", "evidence_level", "evidence_definition",
    "summary", "extended_description", "management",
    # --- Subject Drug Profile ---
    "subject_description", "subject_simple_description", "subject_half_life_hours",
    # --- Affected Drug Profile ---
    "affected_description", "affected_simple_description", "affected_half_life_hours",
]

# Text columns to down-cast to pandas 'string' (numeric cols stay numeric).
TEXT_COLS: List[str] = [
    "subject_drug_drugbank_id", "affected_drug_drugbank_id",
    "subject_drug_name", "affected_drug_name", "severity_label", "risk_level",
    "evidence_definition", "summary", "extended_description", "management",
    "subject_description", "subject_simple_description",
    "affected_description", "affected_simple_description",
]



# Safe, atomic, lock-aware write (consistent with the rest of the pipeline)    #

def _safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _atomic_write(df: pd.DataFrame, path: str, retries: int = 1) -> bool:
    """Write atomically; back up an existing target; skip cleanly if it is locked."""
    tmp = f"{path}.tmp"
    bak = f"{path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
    for attempt in range(retries + 1):
        try:
            df.to_csv(tmp, index=False)
            if os.path.exists(path):                 # back up only if overwriting
                shutil.copy2(path, bak)
                logger.info("  backup -> %s", os.path.basename(bak))
            os.replace(tmp, path)
            return True
        except PermissionError:
            _safe_remove(tmp)
            _safe_remove(bak)
            if attempt < retries:
                logger.warning("  target locked, retrying in 1s ...")
                time.sleep(1)
                continue
            logger.error("  LOCKED - close %s (open in Excel?) and re-run.",
                         os.path.basename(path))
            return False
        except Exception as exc:  # noqa: BLE001
            _safe_remove(tmp)
            logger.error("  write failed (%s)", exc)
            return False
    return False



# STEP 1 - Ingestion & integrity checks                                        #

def load_interactions(path: str) -> pd.DataFrame:
    """Load the interaction pairs and verify 'action' was dropped."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Interactions file not found: {path}")
    # text columns as 'string'; severity/evidence_level remain integer codes.
    text = ["subject_drug_drugbank_id", "subject_drug_name",
            "affected_drug_drugbank_id", "affected_drug_name",
            "summary", "extended_description", "management"]
    df = pd.read_csv(path, dtype={c: "string" for c in text})
    if "action" in df.columns:
        raise ValueError("Integrity check FAILED: 'action' column is still present "
                         "in the interactions file - run the cleanup step first.")
    for req in ("subject_drug_drugbank_id", "affected_drug_drugbank_id", "severity"):
        if req not in df.columns:
            raise ValueError(f"Interactions file missing required column: {req}")
    logger.info("[1] interactions loaded: %s rows, %d cols | 'action' dropped: OK",
                f"{len(df):,}", df.shape[1])
    return df


def load_master(path: str) -> pd.DataFrame:
    """Load the per-drug master and confirm both description columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Master file not found: {path}")
    df = pd.read_csv(path, dtype={"drugbank_id": "string",
                                  "description": "string",
                                  "simple_description": "string"})
    for col in ("description", "simple_description"):
        if col not in df.columns:
            raise ValueError(f"Integrity check FAILED: drug_master is missing '{col}'.")
    if "half_life_hours" not in df.columns:
        raise ValueError("drug_master is missing 'half_life_hours'.")
    logger.info("[1] drug_master loaded: %s rows | 'description' & "
                "'simple_description' present: OK", f"{len(df):,}")
    return df



# STEP 2 - Two-sided relational join (reverse enrichment)                      #

def _side_frame(master: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    Build a one-sided lookup frame from the master for `side` in {subject, affected}.

    Takes only the profile columns, prefixes them (e.g. 'subject_description'),
    and renames the key so it matches the interaction file's foreign key
    ('subject_drug_drugbank_id' / 'affected_drug_drugbank_id').
    """
    frame = master[PROFILE_COLS].add_prefix(f"{side}_")
    return frame.rename(columns={f"{side}_drugbank_id": f"{side}_drug_drugbank_id"})


def to_severity_label(severity: pd.Series) -> pd.Series:
    """
    Normalise `severity` to {'minor','moderate','major'} whether it is stored as
    the integer code (0/1/2) or already as a string. Unknown -> <NA> (logged).
    """
    label = pd.Series(pd.NA, index=severity.index, dtype="string")
    num = pd.to_numeric(severity, errors="coerce")     # integer-coded rows
    is_num = num.notna()
    if is_num.any():
        label.loc[is_num] = (num[is_num].astype("int64")
                             .map(SEVERITY_INT_TO_LABEL).astype("string"))
    # rows that were genuine strings (not numeric)
    str_rows = ~is_num & severity.notna()
    if str_rows.any():
        label.loc[str_rows] = severity[str_rows].astype("string").str.strip().str.lower()

    unknown = severity.notna() & ~label.isin(list(RISK_LEVEL_MAP))
    if unknown.any():
        logger.warning("severity: %s value(s) did not map to minor/moderate/major",
                       f"{int(unknown.sum()):,}")
    return label


def build_corpus(ddi: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich each interaction row with BOTH drugs' profiles, then add the clinical
    risk-level feature. Left joins preserve every interaction row exactly.
    """
    # --- 2a. join the SUBJECT profile on subject_drug_drugbank_id --------------
    corpus = ddi.merge(_side_frame(master, "subject"),
                       on="subject_drug_drugbank_id", how="left")
    # --- 2b. join the AFFECTED profile on affected_drug_drugbank_id ------------
    corpus = corpus.merge(_side_frame(master, "affected"),
                          on="affected_drug_drugbank_id", how="left")

    # join-coverage QA: a missing description means the id was absent from master
    miss_subj = int(corpus["subject_description"].isna().sum())
    miss_aff = int(corpus["affected_description"].isna().sum())
    logger.info("[2] two-sided join done | subject profiles missing: %d | "
                "affected profiles missing: %d", miss_subj, miss_aff)

    # --- 3. explicit clinical value mapping (severity -> label -> risk_level) --
    corpus["severity_label"] = to_severity_label(corpus["severity"])
    corpus["risk_level"] = corpus["severity_label"].map(RISK_LEVEL_MAP).astype("string")
    logger.info("[3] severity mapped -> risk_level | label counts: %s",
                corpus["severity_label"].value_counts(dropna=False).to_dict())

    # evidence_level code -> clinical definition (placed right after evidence_level)
    corpus["evidence_definition"] = (
        pd.to_numeric(corpus["evidence_level"], errors="coerce").astype("Int64")
        .map(EVIDENCE_MAP).astype("string"))

    # --- enforce the explicit column order (extras, if any, go to the end) -----
    ordered = [c for c in FINAL_COLS if c in corpus.columns]
    extras = [c for c in corpus.columns if c not in FINAL_COLS]
    if extras:
        logger.info("    (extra columns appended: %s)", extras)
    corpus = corpus[ordered + extras]

    # --- 4. explicit dtype casting: text -> 'string' (numeric cols untouched) --
    for col in TEXT_COLS:
        if col in corpus.columns:
            corpus[col] = corpus[col].astype("string")
    return corpus



# STEP 4 - Verification & save                                                 #

def main() -> int:
    logger.info("=" * 75)
    logger.info("PHASE 3 - NLP INTERACTION CORPUS | dir=%s", PROCESSED_DIR)
    logger.info("=" * 75)

    try:
        ddi = load_interactions(os.path.join(PROCESSED_DIR, INTERACTIONS_FILE))
        master = load_master(os.path.join(PROCESSED_DIR, MASTER_FILE))
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    corpus = build_corpus(ddi, master)

    out_path = os.path.join(PROCESSED_DIR, OUTPUT_FILE)
    if not _atomic_write(corpus, out_path):
        return 1
    logger.info("[4] saved -> %s", out_path)

    # ---- rule 4 verification output ----
    print("\n" + "=" * 82)
    print("FINAL CORPUS VERIFICATION")
    print("=" * 82)
    print("Shape:", corpus.shape)
    print("\nColumns (%d):" % corpus.shape[1])
    for i, c in enumerate(corpus.columns):
        print(f"  {i:2d}. {c:30s} [{corpus[c].dtype}]")
    with pd.option_context("display.max_columns", None, "display.width", 220,
                           "display.max_colwidth", 26):
        print("\nFirst 5 rows (long text truncated for display):\n",
              corpus.head(5).to_string(index=False))

    logger.info("=" * 75)
    logger.info("DONE. Corpus rows=%s, cols=%d", f"{len(corpus):,}", corpus.shape[1])
    logger.info("=" * 75)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
