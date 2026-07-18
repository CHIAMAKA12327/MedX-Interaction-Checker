#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Phase 4 - Rule-Driven Patient-Facing NLP Simplification & Readability Analytics
"MAKING POLYPHARMACY SAFER: AN AI-BASED TRANSLATIVE MODEL FOR DRUG INTERACTION
 RISK DETECTION AND PATIENT GUIDANCE"

Reads the enriched interaction corpus and, for every row:

  (A) DETERMINISTIC guidance synthesis -> `patient_friendly_guidance`
        1. severity_label  -> a fixed patient sentence (prepended)
        2. statin guardrail -> appends "The statins should be taken at night."
                               iff either drug name partial-matches a target statin
        3. evidence_definition -> appended as "Clinical Validity: <text>"
      

  (B) READABILITY ANALYTICS (textstat, with a pure-Python fallback)
        Flesch Reading Ease (FRE) and Flesch-Kincaid Grade Level (FKGL) for the
        raw clinical baseline (extended_description, else summary) vs. the
        generated guidance, plus the directional per-row delta (Net Shift).

Outputs `final_patient_guidance_analytics.csv` (atomic write + .bak) and prints
an executive readability report.

Design: comprehensive per-row error boundaries, processing telemetry, atomic and
lock-aware persistence. 

"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd


# Logging / telemetry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("phase4_nlp")


 # Configuration - with relative paths (100% PORTABLE)
DEFAULT_PROCESSED_DIR = r"Processed_outputs"

PROCESSED_DIR = os.environ.get("POLYPHARMACY_OUT", DEFAULT_PROCESSED_DIR)
INPUT_FILE = "final_nlp_interaction_corpus.csv"
OUTPUT_FILE = "final_patient_guidance_analytics.csv"

# --- Rule 1a: severity_label -> fixed patient sentence (verbatim, no variation) --
SEVERITY_GUIDANCE: Dict[str, str] = {
    "minor": ("Patient Guidance: One of the medications should be spaced from the "
              "other. Please observe and adjust."),
    "moderate": ("Patient Guidance: Clinical adjustment should be considered. "
                 "Consult your GP or Physician for adjustment."),
    "major": ("Critical Patient Warning: Avoid simultaneous intake of these drugs. "
              "Consult your GP or Physician for an immediate review."),
}

# --- Rule 1b: statin chronotherapy guardrail ---------------------------------
STATIN_NAMES: Tuple[str, ...] = ("atorvastatin", "rosuvastatin", "simvastatin")
STATIN_SENTENCE = "The statins should be taken at night."

# --- Rule 1c: evidence label -------------------------------------------------
EVIDENCE_LABEL = "Clinical Validity:"



# READABILITY BACKEND - textstat primary, pure-Python fallback        #

try:
    import textstat as _textstat  # type: ignore
    _HAS_TEXTSTAT = True
except Exception:  # noqa: BLE001 - any import failure -> use fallback
    _textstat = None
    _HAS_TEXTSTAT = False


def _count_syllables(word: str) -> int:
    """Approximate English syllable count for a single word (heuristic)."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    vowel_groups = re.findall(r"[aeiouy]+", word)
    n = len(vowel_groups)
    if word.endswith("e") and not word.endswith(("le", "ye")) and n > 1:
        n -= 1                                    # drop typical silent 'e'
    return max(n, 1)


def _basic_counts(text: str) -> Tuple[int, int, int]:
    """Return (sentences, words, syllables) with safe minimums."""
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    n_sent = max(len(sentences), 1)
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    n_words = len(words)
    n_syll = sum(_count_syllables(w) for w in words)
    return n_sent, n_words, n_syll


def _fallback_fre(text: str) -> float:
    s, w, y = _basic_counts(text)
    if w == 0:
        return math.nan
    return 206.835 - 1.015 * (w / s) - 84.6 * (y / w)


def _fallback_fkgl(text: str) -> float:
    s, w, y = _basic_counts(text)
    if w == 0:
        return math.nan
    return 0.39 * (w / s) + 11.8 * (y / w) - 15.59


def flesch_reading_ease(text: str) -> float:
    """FRE via textstat, falling back to the pure-Python formula on any failure."""
    if not isinstance(text, str) or not text.strip():
        return math.nan
    if _HAS_TEXTSTAT:
        try:
            return float(_textstat.flesch_reading_ease(text))
        except Exception:  # noqa: BLE001
            return _fallback_fre(text)
    return _fallback_fre(text)


def flesch_kincaid_grade(text: str) -> float:
    """FKGL via textstat, falling back to the pure-Python formula on any failure."""
    if not isinstance(text, str) or not text.strip():
        return math.nan
    if _HAS_TEXTSTAT:
        try:
            return float(_textstat.flesch_kincaid_grade(text))
        except Exception:  # noqa: BLE001
            return _fallback_fkgl(text)
    return _fallback_fkgl(text)



# Rule 2 - deterministic guidance synthesis                                    #

def synthesize_guidance(severity_label, subject_name, affected_name,
                        evidence_definition) -> Tuple[str, bool, bool]:
    """
    Build the patient_friendly_guidance card by sequentially applying the three
    fixed rules. Returns (guidance_text, severity_mapped, statin_applied).
    """
    parts = []

    # 1. severity translation (prepended)
    key = str(severity_label).strip().lower() if pd.notna(severity_label) else ""
    severity_sentence = SEVERITY_GUIDANCE.get(key)
    severity_mapped = severity_sentence is not None
    if severity_mapped:
        parts.append(severity_sentence)

    # 2. statin chronotherapy guardrail (case-insensitive partial match, either side)
    subj = str(subject_name).lower() if pd.notna(subject_name) else ""
    aff = str(affected_name).lower() if pd.notna(affected_name) else ""
    statin_applied = any(t in subj or t in aff for t in STATIN_NAMES)
    if statin_applied:
        parts.append(STATIN_SENTENCE)

    # 3. evidence-level contextualization (verbatim, under a fixed label)
    if isinstance(evidence_definition, str) and evidence_definition.strip():
        parts.append(f"{EVIDENCE_LABEL} {evidence_definition.strip()}")

    return " ".join(parts), severity_mapped, statin_applied



# Per-row processing (fully error-boundaried)                                  #

def _pick_baseline(row) -> Tuple[str, str]:
    """Baseline clinical text = extended_description, else summary."""
    ext = row.get("extended_description")
    if isinstance(ext, str) and ext.strip():
        return ext, "extended_description"
    summ = row.get("summary")
    if isinstance(summ, str) and summ.strip():
        return summ, "summary"
    return "", "none"


def process_row(row) -> Dict[str, object]:
    """Synthesize guidance + compute both readability metrics for one record."""
    try:
        guidance, sev_ok, statin = synthesize_guidance(
            row.get("severity_label"), row.get("subject_drug_name"),
            row.get("affected_drug_name"), row.get("evidence_definition"))
        baseline, source = _pick_baseline(row)

        fre_b, fk_b = flesch_reading_ease(baseline), flesch_kincaid_grade(baseline)
        fre_s, fk_s = flesch_reading_ease(guidance), flesch_kincaid_grade(guidance)

        return {
            "patient_friendly_guidance": guidance,
            "baseline_text_source": source,
            "statin_guardrail_applied": statin,
            "fre_baseline": fre_b, "fre_simplified": fre_s,
            "fre_delta": (fre_s - fre_b) if pd.notna(fre_b) and pd.notna(fre_s) else math.nan,
            "fkgl_baseline": fk_b, "fkgl_simplified": fk_s,
            "fkgl_delta": (fk_s - fk_b) if pd.notna(fk_b) and pd.notna(fk_s) else math.nan,
            "_processed_ok": bool(guidance) and sev_ok,
        }
    except Exception as exc:  # noqa: BLE001 - never let one row abort the batch
        logger.error("row %s failed: %s", getattr(row, "name", "?"), exc)
        return {
            "patient_friendly_guidance": pd.NA, "baseline_text_source": "error",
            "statin_guardrail_applied": False,
            "fre_baseline": math.nan, "fre_simplified": math.nan, "fre_delta": math.nan,
            "fkgl_baseline": math.nan, "fkgl_simplified": math.nan, "fkgl_delta": math.nan,
            "_processed_ok": False,
        }



# Safe, atomic, lock-aware persistence                                         #

def _safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _atomic_write(df: pd.DataFrame, path: str, retries: int = 1) -> bool:
    tmp = f"{path}.tmp"
    bak = f"{path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
    for attempt in range(retries + 1):
        try:
            df.to_csv(tmp, index=False)
            if os.path.exists(path):
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



# Orchestration                                                                #

def main() -> int:
    in_path = os.path.join(PROCESSED_DIR, INPUT_FILE)
    out_path = os.path.join(PROCESSED_DIR, OUTPUT_FILE)

    logger.info("=" * 75)
    logger.info("PHASE 4 - NLP TRANSLATION & READABILITY ANALYTICS")
    logger.info("readability backend: %s", "textstat" if _HAS_TEXTSTAT
                else "pure-Python fallback")
    logger.info("=" * 75)

    # ---- Rule 1: ingestion quality control (all text columns as 'string') ----
    if not os.path.exists(in_path):
        logger.error("Input corpus not found: %s", in_path)
        return 2
    try:
        df = pd.read_csv(in_path, dtype="string")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read corpus: %s", exc)
        return 2
    required = ["severity_label", "evidence_definition", "summary",
                "subject_drug_name", "affected_drug_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Integrity check FAILED - missing columns: %s", missing)
        return 2
    logger.info("[1] corpus loaded: %s rows x %d cols", f"{len(df):,}", df.shape[1])

    # ---- Rule 2 + 3: synthesize guidance & measure readability, row by row ----
    logger.info("[2/3] synthesizing guidance and scoring readability ...")
    results = pd.DataFrame(list(df.apply(process_row, axis=1)))
    enriched = pd.concat([df.reset_index(drop=True),
                          results.reset_index(drop=True)], axis=1)

    # explicit dtypes: new text -> 'string', flags -> boolean, metrics -> float
    enriched["patient_friendly_guidance"] = enriched["patient_friendly_guidance"].astype("string")
    enriched["baseline_text_source"] = enriched["baseline_text_source"].astype("string")
    enriched["statin_guardrail_applied"] = enriched["statin_guardrail_applied"].astype("boolean")
    for m in ("fre_baseline", "fre_simplified", "fre_delta",
              "fkgl_baseline", "fkgl_simplified", "fkgl_delta"):
        enriched[m] = pd.to_numeric(enriched[m], errors="coerce")

    # ---- processing telemetry ----
    total = len(enriched)
    ok = int(enriched["_processed_ok"].sum())
    n_statin = int(enriched["statin_guardrail_applied"].sum())
    src_counts = enriched["baseline_text_source"].value_counts().to_dict()
    sev_counts = df["severity_label"].str.lower().value_counts().to_dict()
    logger.info("[telemetry] severity=%s | statin guardrail fired=%d | baseline source=%s",
                sev_counts, n_statin, src_counts)

    enriched = enriched.drop(columns=["_processed_ok"])  # keep the saved file clean

    # ---- Rule 4: persist (atomic + backup) ----
    if not _atomic_write(enriched, out_path):
        return 1
    logger.info("[4] saved -> %s (%s rows x %d cols)", out_path,
                f"{len(enriched):,}", enriched.shape[1])

    # ---- Rule 4: executive readability report ----
    valid = enriched.dropna(subset=["fkgl_baseline", "fkgl_simplified"])
    fkgl_lowered = int((valid["fkgl_delta"] < 0).sum())
    fre_raised = int((valid["fre_delta"] > 0).sum())

    report = pd.DataFrame({
        "Metric": ["Flesch Reading Ease (FRE)", "Flesch-Kincaid Grade (FKGL)"],
        "Baseline (avg)": [enriched["fre_baseline"].mean(), enriched["fkgl_baseline"].mean()],
        "Simplified (avg)": [enriched["fre_simplified"].mean(), enriched["fkgl_simplified"].mean()],
        "Net Shift (avg)": [enriched["fre_delta"].mean(), enriched["fkgl_delta"].mean()],
    }).round(2)

    print("\n" + "=" * 75)
    print("EXECUTIVE READABILITY REPORT")
    print("=" * 75)
    print(report.to_string(index=False))
    print("-" * 75)
    print(f"Readability backend            : {'textstat' if _HAS_TEXTSTAT else 'pure-Python fallback'}")
    print(f"Total text segments processed  : {total:,}")
    print(f"Successful transformations     : {ok:,} ({100*ok/max(total,1):.1f}%)")
    print(f"Reading difficulty lowered FKGL: {fkgl_lowered:,} ({100*fkgl_lowered/max(len(valid),1):.1f}%)")
    print(f"Reading ease raised (FRE up)   : {fre_raised:,} ({100*fre_raised/max(len(valid),1):.1f}%)")
    print("Interpretation: FRE higher = easier; FKGL lower = easier "
          "(Net Shift = Simplified - Baseline).")
    print("=" * 75)

    # small qualitative sample for eyeballing the transformation
    with pd.option_context("display.max_colwidth", 90, "display.width", 200):
        print("\nSample guidance cards:")
        print(enriched[["severity_label", "statin_guardrail_applied",
                        "patient_friendly_guidance"]].head(3).to_string(index=False))

    logger.info("=" * 75)
    logger.info("PHASE 4 COMPLETE.")
    logger.info("=" * 75)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
