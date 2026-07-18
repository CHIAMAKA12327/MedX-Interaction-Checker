#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export a small, stratified HUMAN-REVIEW sample of the Phase 4 guidance cards for
clinical sign-off ("near-zero clinically significant errors" QA).

The sample is chosen deterministically to cover every rule path:
  * each severity_label (minor / moderate / major)
  * statin guardrail fired vs. not
  * each evidence_level (1 / 2)
  * plus the best and worst readability cases (largest / smallest FKGL reduction)

Two artefacts are written to Drug_bank 3.0/:
  1. guidance_review_sample.csv  - structured, with blank reviewer-verdict columns
  2. guidance_review_sample.md   - one readable card per interaction for sign-off

"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger("review_sample")

# Configuration - with relative paths (100% PORTABLE)
DEFAULT_PROCESSED_DIR = r"Processed_outputs"
PROCESSED_DIR = os.environ.get("POLYPHARMACY_OUT", DEFAULT_PROCESSED_DIR)

INPUT_FILE = "final_patient_guidance_analytics.csv"
CSV_OUT = "guidance_review_sample.csv"
MD_OUT = "guidance_review_sample.md"

PER_CELL = 2        # rows per (severity x statin x evidence) stratum
N_EXTREMES = 2      # best & worst readability cases to add
SEED = 42           # deterministic / reproducible sample

METRICS = ["fkgl_baseline", "fkgl_simplified", "fkgl_delta",
           "fre_baseline", "fre_simplified", "fre_delta"]


def _safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _atomic_write_text(text: str, path: str) -> bool:
    """Atomic, lock-aware text write with .bak backup if overwriting."""
    tmp, bak = f"{path}.tmp", f"{path}.{datetime.now():%Y%m%d_%H%M%S}.bak"
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        if os.path.exists(path):
            shutil.copy2(path, bak)
            logger.info("  backup -> %s", os.path.basename(bak))
        os.replace(tmp, path)
        return True
    except PermissionError:
        _safe_remove(tmp)
        logger.error("  LOCKED - close %s and re-run.", os.path.basename(path))
        return False
    except Exception as exc:  # noqa: BLE001
        _safe_remove(tmp)
        logger.error("  write failed (%s)", exc)
        return False


def select_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministically pick a stratified + extremes sample of cards."""
    idx: set = set()

    # 1) stratified coverage of every rule path
    keys = ["severity_label", "statin_guardrail_applied", "evidence_level"]
    for _, grp in df.groupby(keys, dropna=False):
        idx.update(grp.sample(min(len(grp), PER_CELL), random_state=SEED).index)

    # 2) readability extremes (best = most-negative delta, worst = most-positive)
    valid = df.dropna(subset=["fkgl_delta"])
    idx.update(valid.nsmallest(N_EXTREMES, "fkgl_delta").index)   # biggest reduction
    idx.update(valid.nlargest(N_EXTREMES, "fkgl_delta").index)    # least/worst

    sample = df.loc[sorted(idx)].copy()
    # order for reviewer: severity (major first), then drug pair
    sev_order = {"major": 0, "moderate": 1, "minor": 2}
    sample["_sev"] = sample["severity_label"].str.lower().map(sev_order).fillna(9)
    sample = sample.sort_values(["_sev", "subject_drug_name", "affected_drug_name"]).drop(columns="_sev")
    return sample.reset_index(drop=True)


def build_csv(sample: pd.DataFrame) -> pd.DataFrame:
    """Assemble the reviewer CSV with rounded metrics + blank verdict columns."""
    out = pd.DataFrame({"review_id": range(1, len(sample) + 1)})
    for c in ["subject_drug_name", "affected_drug_name", "severity_label",
              "risk_level", "evidence_level", "statin_guardrail_applied",
              "summary", "management", "evidence_definition",
              "patient_friendly_guidance", "baseline_text_source"]:
        out[c] = sample[c].values
    for m in METRICS:
        out[m] = pd.to_numeric(sample[m], errors="coerce").round(1).values
    # blank columns for the human reviewer to fill in
    out["reviewer_clinically_accurate"] = ""     # Yes / No
    out["reviewer_grade6_readable"] = ""         # Yes / No
    out["reviewer_notes"] = ""
    return out


def build_markdown(sample: pd.DataFrame) -> str:
    """One readable card per row for clinical sign-off."""
    lines = [
        "# Patient-Guidance Cards - Human Review Sample",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} from `{INPUT_FILE}` "
        f"({len(sample)} cards, deterministic seed={SEED})._",
        "",
        "For each card, confirm the patient guidance is **clinically accurate** and "
        "does not omit critical safety advice, then tick the review boxes.",
        "",
    ]
    for i, r in sample.iterrows():
        fkgl_b = pd.to_numeric(r["fkgl_baseline"], errors="coerce")
        fkgl_s = pd.to_numeric(r["fkgl_simplified"], errors="coerce")
        fkgl_d = pd.to_numeric(r["fkgl_delta"], errors="coerce")
        fre_b = pd.to_numeric(r["fre_baseline"], errors="coerce")
        fre_s = pd.to_numeric(r["fre_simplified"], errors="coerce")
        statin = "yes" if str(r["statin_guardrail_applied"]).lower() == "true" else "no"
        lines += [
            f"## Card {i + 1} - {r['severity_label'].upper()} - "
            f"{r['subject_drug_name']} + {r['affected_drug_name']}",
            "",
            f"- **Severity / risk level:** {r['severity_label']} - {r['risk_level']}",
            f"- **Evidence level {r['evidence_level']}:** {r['evidence_definition']}",
            f"- **Statin guardrail applied:** {statin}",
            "",
            f"**Clinical source (summary):** {r['summary']}",
            "",
            f"**Clinical management:** {r['management']}",
            "",
            f"**> Generated patient guidance:**",
            "",
            f"> {r['patient_friendly_guidance']}",
            "",
            f"**Readability:** FKGL {fkgl_b:.1f} -> {fkgl_s:.1f} (delta {fkgl_d:+.1f}) | "
            f"FRE {fre_b:.1f} -> {fre_s:.1f}",
            "",
            "**Reviewer:** clinically accurate? [ ] Yes  [ ] No  &nbsp;&nbsp; "
            "grade-6 readable? [ ] Yes  [ ] No",
            "",
            "**Notes:** _______________________________________________",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    in_path = os.path.join(PROCESSED_DIR, INPUT_FILE)
    if not os.path.exists(in_path):
        logger.error("Input not found: %s (run Phase 4 first)", in_path)
        return 2

    # text columns preserved; metrics/flags inferred for sampling
    df = pd.read_csv(in_path)
    need = ["severity_label", "statin_guardrail_applied", "evidence_level",
            "patient_friendly_guidance", "fkgl_delta", "summary", "management"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        logger.error("Input missing required columns: %s", missing)
        return 2

    logger.info("Loaded analytics: %s rows. Selecting review sample ...", f"{len(df):,}")
    sample = select_sample(df)
    logger.info("Sample size: %d cards | severity mix=%s | statin cards=%d",
                len(sample),
                sample["severity_label"].str.lower().value_counts().to_dict(),
                int((sample["statin_guardrail_applied"].astype(str).str.lower() == "true").sum()))

    csv_df = build_csv(sample)
    csv_path = os.path.join(PROCESSED_DIR, CSV_OUT)
    md_path = os.path.join(PROCESSED_DIR, MD_OUT)

    ok_csv = _atomic_write_text(csv_df.to_csv(index=False), csv_path)
    if ok_csv:
        logger.info("saved -> %s (%d rows x %d cols)", csv_path, *csv_df.shape)
    ok_md = _atomic_write_text(build_markdown(sample), md_path)
    if ok_md:
        logger.info("saved -> %s", md_path)

    # console preview of the compact review table
    with pd.option_context("display.max_colwidth", 46, "display.width", 200):
        print("\n" + "=" * 90)
        print("HUMAN-REVIEW SAMPLE (preview)")
        print("=" * 90)
        print(csv_df[["review_id", "severity_label", "statin_guardrail_applied",
                      "subject_drug_name", "affected_drug_name",
                      "fkgl_baseline", "fkgl_simplified"]].to_string(index=False))
    return 0 if (ok_csv and ok_md) else 1


if __name__ == "__main__":
    raise SystemExit(main())
