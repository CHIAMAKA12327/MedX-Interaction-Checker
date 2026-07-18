#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Polypharmacy Safety - DrugBank Data Cleaning, Standardization & Merging Pipeline


This part ingests four DrugBank export files, cleans/standardizes them, and
merges them into analysis-ready tables for the ML + NLP framework.

    1. structured_pharmacology_half_lives.csv
    2. food_interactions.csv
    3. drugs.csv
    4. structured_drug_interactions.csv   


IMPORTANT DATA-ENGINEERING DECISION 

standardizing the numeric drug id with `f"DB{int(x):05d}"`.

  * In drugs.csv the integer `id` equals the DrugBank number for only 17.5% of
    rows. 
  * mapping the id through the drugs.csv `id -> drugbank_id`
    crosswalk matched the ground truth (the drugbank_id columns already present
    in structured_drug_interactions.csv) 100% of the time.
This pipeline therefore standardizes ids through the crosswalk built from the provided drugs.csv and only falls back to `DB{05d}` for ids missing from the
crosswalk.


CLINICAL NOTE ON HALF-LIFE UNITS

The `unit` column is heterogeneous (hours/days/minutes/weeks/seconds, plus casing
variants). After id standardization, 83 aggregated drugs mix >1 unit, so a raw mean
of `value` across those rows is not clinically meaningful. The definitive `half_life` is the
raw value->min mean, but a unit-canonicalized `half_life_hours`
(mean of per-row values converted to hours)
plus a `half_life_unit_mixed` flag so the heterogeneity is explicit and auditable.

"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# Regional Synonym Fallback Dictionary
SYNONYM_FALLBACK = {
    'aspirin': 'acetylsalicylic acid',
    'paracetamol': 'acetaminophen',
    'chlorphenamine': 'chlorpheniramine',
    'salbutamol': 'albuterol',
    'beclometasone': 'beclomethasone',
    'colecalciferol': 'cholecalciferol'
}

# Logging: every processing step reports the resulting data shape. #

logger = logging.getLogger("polypharmacy")


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def log_shape(name: str, df: pd.DataFrame, extra: str = "") -> None:
    """Uniform 'shape after step' logging required by the brief."""
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    msg = f"[{name}] shape={df.shape[0]:,} rows x {df.shape[1]} cols | mem={mem_mb:,.1f} MB"
    if extra:
        msg += f" | {extra}"
    logger.info(msg)


# SECTION 0 - Static configuration (column drop-lists) #

# Columns to DROP. We realise "drop on read" via usecols = header
# minus these, which never materialises the dropped columns in memory.
HALF_LIFE_DROP: Set[str] = {
    "id", "max", "normalized_value", "normalized_min", "normalized_max",
    "normalized_unit", "dosage_route_id", "dosage_form_id", "dose_value",
    "dose_min", "dose_max", "dose_unit", "normalized_dose_value",
    "normalized_dose_min", "normalized_dose_max", "normalized_dose_unit",
    "fda_label_reference", "sex", "age_group",
}
FOOD_DROP: Set[str] = {"id"}
INTERACTIONS_DROP: Set[str] = {
    "subject_dosage", "subject_drug_id", "affected_drug_id", "conditions",
    "affected_category_id", "subject_category_id", "affected_dosage", "id",
}
# drugs.csv: 'description' (the detailed text) is kept
# immediately before 'simple_description' (rule: preserve the original description).
DRUGS_KEEP: List[str] = ["drugbank_id", "name", "description", "simple_description"]

# --- Half-life unit canonicalization (values observed in the file) ------ #
_UNIT_CANON: Dict[str, str] = {}
for _syns, _canon in [
    (("s", "sec", "secs", "second", "seconds"), "seconds"),
    (("m", "min", "mins", "minute", "minutes"), "minutes"),
    (("h", "hr", "hrs", "hour", "hours"), "hours"),
    (("d", "day", "days"), "days"),
    (("w", "wk", "wks", "week", "weeks"), "weeks"),
    (("mo", "mon", "month", "months"), "months"),
    (("y", "yr", "yrs", "year", "years"), "years"),
]:
    for _s in _syns:
        _UNIT_CANON[_s] = _canon

# Conversion factor -> hours (only for canonical units we recognise).
_TO_HOURS: Dict[str, float] = {
    "seconds": 1.0 / 3600.0,
    "minutes": 1.0 / 60.0,
    "hours": 1.0,
    "days": 24.0,
    "weeks": 168.0,
    "months": 730.485,   # 30.4369 days * 24
    "years": 8765.82,
}

# --- Clinical synonym rules (Iron ~ Ferrous ~ Ferric) ------------------- #
# Bidirectional token groups used both for cohort resolution and text matching.
SYNONYM_GROUPS: List[Set[str]] = [
    {"iron", "ferrous", "ferric"},
]

# Cohort: the ~75 study drugs (parsed from "75 Most Prescribed Drugs.docx").
# Combination / class entries are split into individually-resolvable tokens.

COHORT_DRUGS: Dict[str, List[str]] = {
    "Cardiovascular & Blood Thinner": [
        "Atorvastatin", "Rosuvastatin", "Simvastatin", "Ramipril", "Bisoprolol",
        "Lisinopril", "Amlodipine", "Losartan", "Clopidogrel", "Apixaban",
        "Rivaroxaban", "Furosemide", "Bendroflumethiazide", "Aspirin", "Warfarin",
    ],
    "Endocrine & Metabolic": [
        "Metformin", "Levothyroxine", "Dapagliflozin", "Empagliflozin",
        "Semaglutide", "Tirzepatide", "Liraglutide", "Sitagliptin",
        "Insulin glargine", "Insulin aspart",
    ],
    "Central Nervous System": [
        "Sertraline", "Citalopram", "Escitalopram", "Amitriptyline", "Fluoxetine",
        "Pregabalin", "Gabapentin", "Codeine", "Morphine", "Paracetamol",
        "Ibuprofen", "Naproxen", "Methotrexate", "Quetiapine", "Olanzapine",
        "Dextromethorphan",
    ],
    "Allergic Reactions": [
        "Cetirizine", "Levocetirizine", "Loratadine", "Chlorphenamine",
        "Montelukast", "Fexofenadine", "Pseudoephedrine",
    ],
    "Respiratory": ["Salbutamol", "Beclometasone", "Budesonide", "Tiotropium"],
    "Gastrointestinal": [
        "Omeprazole", "Lansoprazole", "Rabeprazole", "Cimetidine", "Lactulose",
        "Amoxicillin", "Clarithromycin",
    ],
    "Infections": [
        "Nitrofurantoin", "Trimethoprim", "Doxycycline",
        "Phenoxymethylpenicillin", "Flucloxacillin", "Itraconazole",
    ],
    "Supplements": [
        "Colecalciferol", "Folic Acid", "Ferrous fumarate", "Ferrous sulfate",
        "Hydroxocobalamin", "Calcium carbonate", "Zinc", "Ascorbic acid",
        "Phenelzine",
    ],
}


# SECTION 1 - ID standardization (crosswalk-based) #

def build_id_crosswalk(drugs_path: str) -> "pd.Series[str]":
    """
    Build the authoritative {integer id -> 'DBxxxxx'} crosswalk from drugs.csv.

    This standardizes the numeric `drug_id` foreign keys in
    the half-life and food files. Returns a Series indexed
    by int id whose values are the DrugBank accession strings.
    """
    xw = pd.read_csv(drugs_path, usecols=["id", "drugbank_id"])
    xw["id"] = pd.to_numeric(xw["id"], errors="coerce").astype("Int64")
    xw = xw.dropna(subset=["id", "drugbank_id"]).drop_duplicates("id")
    crosswalk = pd.Series(
        xw["drugbank_id"].astype("string").values,
        index=xw["id"].astype("int64").values,
        name="drugbank_id",
    )
    logger.info(f"Built id->drugbank_id crosswalk: {len(crosswalk):,} entries")
    return crosswalk


def standardize_drug_id(
    values: "pd.Series",
    crosswalk: Optional["pd.Series"] = None,
    legacy_format: bool = False,
) -> "pd.Series":
    """
    Standardize a drug-id column to the 'DBxxxxx' DrugBank format.

    Strategy (robust to mixed/dirty input):
      * Values already shaped like 'DB\\d+' are kept as-is.
      * Numeric ids are mapped through `crosswalk` (correct) and only fall back to
        the literal `f"DB{int(x):05d}"` when the id is absent from the crosswalk.
      * `legacy_format=True` forces the literal specified formula for every id.

    Returns a string Series aligned to the input index; unmappable -> <NA>.
    """
    ser = pd.Series(values)
    orig_index = ser.index                       # preserve caller's index
    ser = ser.reset_index(drop=True)
    out = pd.Series(pd.array([pd.NA] * len(ser), dtype="string"))

    as_str = ser.astype("string").str.strip()
    already = as_str.str.fullmatch(r"DB\d+", na=False)
    out[already] = as_str[already]

    todo = ~already
    nums = pd.to_numeric(ser[todo], errors="coerce")
    valid = nums.notna()
    if valid.any():
        ints = nums[valid].astype("int64")
        literal = ints.map(lambda x: f"DB{int(x):05d}").astype("string")
        if legacy_format or crosswalk is None:
            mapped = literal
            n_fallback = 0
        else:

            mapped = ints.map(crosswalk).astype("string")
            missing = mapped.isna()
            n_fallback = int(missing.sum())
            mapped = mapped.where(~missing, literal)  # fallback only where needed
            if n_fallback:
                logger.warning(
                    f"standardize_drug_id: {n_fallback:,} numeric ids not in "
                    f"crosswalk -> used DB{{05d}} fallback")
        out.loc[ints.index] = mapped.values

    n_bad = int(ser.notna().sum() - out.notna().sum())
    if n_bad:
        logger.warning(f"standardize_drug_id: {n_bad:,} ids could not be standardized")
    out.index = orig_index                       # restore caller's index for alignment
    return out


# SECTION 2 - Half-life processing #


def _canonical_unit(units: "pd.Series") -> "pd.Series":
    """Lower-case, strip, and map unit synonyms to a canonical token."""
    u = units.astype("string").str.strip().str.lower().str.rstrip(".")
    return u.map(_UNIT_CANON).fillna(u)


def _mode_first(s: "pd.Series"):
    """Most frequent non-null value (ties -> first); <NA> if empty."""
    s = s.dropna()
    if s.empty:
        return pd.NA
    m = s.mode()
    return m.iat[0] if not m.empty else pd.NA


def process_half_lives(
    path: str,
    crosswalk: Optional["pd.Series"],
    legacy_format: bool = False,
    add_hours: bool = True,
) -> pd.DataFrame:
    """
    Rule - one definitive half-life per drug.

    Aggregation logic (documented step by step):
      1. Read only the surviving columns (drug_id, value, min, unit); every column
         in HALF_LIFE_DROP is dropped by simply not reading it.
      2. Standardize drug_id -> drugbank_id via the crosswalk.
      3. Row-level coalesce: half-life source = `value`, falling back to `min`
         when value is null (profiling confirmed value|min is never both-null).
      4. Group by drugbank_id and take the arithmetic MEAN of that per-row source.
         Drugs with a single row therefore keep their own value unchanged.
      5. Attach the (canonicalized) modal unit, a mixed-unit flag, and a record
         count; optionally add a unit-normalized `half_life_hours`.
    """
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except FileNotFoundError:
        logger.error("Half-life file not found: %s", path)
        raise
    usecols = [c for c in header if c not in HALF_LIFE_DROP]  # drop-on-read
    logger.info("[half_lives] reading columns: %s", usecols)

    df = pd.read_csv(
        path, usecols=usecols,
        dtype={"drug_id": "Int64", "unit": "string"},
    )
    # value/min may contain blanks -> force numeric, invalid -> NaN (exception-safe)
    for col in ("value", "min"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    log_shape("half_lives.raw", df)

    # Step 2: standardize id
    df["drugbank_id"] = standardize_drug_id(df["drug_id"], crosswalk, legacy_format)
    df = df.dropna(subset=["drugbank_id"])

    # Step 3: row-level value -> min coalesce
    df["hl_source"] = df["value"].where(df["value"].notna(), df["min"])
    df["unit_canon"] = _canonical_unit(df["unit"])
    df = df.dropna(subset=["hl_source"])  # keep only rows with a usable number
    if add_hours:
        df["hl_hours"] = df["hl_source"] * df["unit_canon"].map(_TO_HOURS)

    # Step 4 + 5: aggregate to one row per drug
    g = df.groupby("drugbank_id", sort=False)
    out = pd.DataFrame({
        "half_life": g["hl_source"].mean(),
        "half_life_unit": g["unit_canon"].agg(_mode_first),
        "half_life_unit_mixed": g["unit_canon"].nunique(dropna=True) > 1,
        "half_life_n_records": g["hl_source"].size().astype("int32"),
    })
    if add_hours:
        out["half_life_hours"] = g["hl_hours"].mean()
    out = out.reset_index()

    # explicit type casting
    out["half_life"] = out["half_life"].astype("float64")
    out["half_life_unit"] = out["half_life_unit"].astype("string")
    out["half_life_unit_mixed"] = out["half_life_unit_mixed"].astype("boolean")

    n_mixed = int(out["half_life_unit_mixed"].sum())
    if n_mixed:
        logger.warning(
            f"[half_lives] {n_mixed:,} drugs aggregate ACROSS MIXED UNITS - trust "
            f"`half_life_hours` over raw `half_life` for those.")
    log_shape("half_lives.processed", out, extra=f"unique drugs, mixed_unit={n_mixed}")
    return out



# SECTION 3 - Food interaction processing                     #

def process_food_interactions(
    path: str,
    crosswalk: Optional["pd.Series"],
    legacy_format: bool = False,
) -> pd.DataFrame:
    """Rule - drop `id`, standardize drug_id, tidy the description column."""
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except FileNotFoundError:
        logger.error("Food-interaction file not found: %s", path)
        raise
    usecols = [c for c in header if c not in FOOD_DROP]  # drops 'id'
    df = pd.read_csv(path, usecols=usecols, dtype={"drug_id": "Int64",
                                                   "description": "string"})
    log_shape("food.raw", df)

    df["drugbank_id"] = standardize_drug_id(df["drug_id"], crosswalk, legacy_format)
    df = df.dropna(subset=["drugbank_id"])
    # rename to disambiguate from a drug's own `simple_description` after merges
    df = df.rename(columns={"description": "food_interaction"})
    df = df[["drugbank_id", "food_interaction"]].reset_index(drop=True)
    log_shape("food.processed", df, extra=f"{df['drugbank_id'].nunique():,} unique drugs")
    return df


def merge_food_with_halflife(food: pd.DataFrame, half_lives: pd.DataFrame) -> pd.DataFrame:
    """
    Rule (final clause) - match food interactions to processed half-lives on the
    standardized id. LEFT join keeps every food-interaction row (food is many-per
    drug, half_lives is one-per-drug); drugs without a half-life get <NA>.
    """
    merged = food.merge(half_lives, on="drugbank_id", how="left")
    matched = int(merged["half_life"].notna().sum())
    log_shape("food+half_life", merged,
              extra=f"{matched:,}/{len(merged):,} food rows have a half-life")
    return merged



# SECTION 4 - Drugs metadata processing                      #


def process_drugs(path: str) -> pd.DataFrame:
    """Rule - keep drugbank_id, name, description, simple_description (in that order)."""
    try:
        df = pd.read_csv(path, usecols=DRUGS_KEEP,
                         dtype={c: "string" for c in DRUGS_KEEP})
    except FileNotFoundError:
        logger.error("Drugs file not found: %s", path)
        raise
    except ValueError as exc:  # a required column is missing
        logger.error("drugs.csv missing an expected column: %s", exc)
        raise
    df["drugbank_id"] = df["drugbank_id"].str.strip()
    df = df.dropna(subset=["drugbank_id"]).drop_duplicates("drugbank_id")
    df = df[df["drugbank_id"].str.fullmatch(r"DB\d+", na=False)]  # rule 4 format guard
    # enforce explicit order: 'description' right before 'simple_description'
    df = df[[c for c in DRUGS_KEEP if c in df.columns]]
    log_shape("drugs.processed", df,
              extra=f"{df['simple_description'].notna().sum():,} have simple_description")
    return df.reset_index(drop=True)


# SECTION 5 - Large drug-drug interaction file #

def process_drug_interactions(
    path: str,
    target_ids: Optional[Set[str]] = None,
    filter_mode: str = "both",
    chunksize: int = 500_000,
    out_csv: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Rule - stream the structured drug interaction file in chunks, dropping the listed columns on read.

    MEMORY MANAGEMENT
    -----------------
    * `usecols` = header minus INTERACTIONS_DROP -> the 8 dropped columns are never
      allocated.
    * The file is consumed in `chunksize`-row pieces via the pandas iterator, so
      peak RAM is ~one chunk, not the whole file.
    * Per-chunk downcasting: id/name/text -> pandas 'string'; the ordinal codes
      severity/evidence_level -> Int8; action -> 'category'.
    * Optional cohort filter keeps only interactions among the study drugs, which
      is what actually makes the result fit comfortably in memory.
        - filter_mode='both'  : keep a row only if BOTH ends are in `target_ids`
                                (the within-cohort DDI network - default).
        - filter_mode='either': keep a row if EITHER end is in `target_ids`.
    * If no filter is supplied the cleaned data is streamed straight to `out_csv`
      and NOT concatenated in RAM (returns None) to stay memory-safe.

    Returns the concatenated cleaned DataFrame when filtering (small), else None.
    """
    if filter_mode not in {"both", "either"}:
        raise ValueError("filter_mode must be 'both' or 'either'")

    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except FileNotFoundError:
        logger.error("Interactions file not found: %s", path)
        raise
    usecols = [c for c in header if c not in INTERACTIONS_DROP]  # drop-on-read
    logger.info("[interactions] keeping columns: %s", usecols)

    # dtype map restricted to the kept string/categorical columns
    str_cols = [c for c in (
        "subject_drug_drugbank_id", "subject_drug_name",
        "affected_drug_drugbank_id", "affected_drug_name",
        "summary", "extended_description", "management") if c in usecols]
    read_dtype = {c: "string" for c in str_cols}
    if "action" in usecols:
        read_dtype["action"] = "category"

    filtering = target_ids is not None
    if not filtering and out_csv is None:
        raise ValueError("Provide target_ids (to filter) or out_csv (to stream) or both.")

    collected: List[pd.DataFrame] = []
    total_in = total_out = 0
    wrote_header = False
    if out_csv and os.path.exists(out_csv):
        os.remove(out_csv)  # start clean for append mode

    reader = pd.read_csv(
        path, usecols=usecols, dtype=read_dtype,
        chunksize=chunksize, on_bad_lines="warn",
    )
    for i, chunk in enumerate(reader, start=1):
        total_in += len(chunk)

        # explicit downcast of the ordinal code columns (exception-safe)
        for col in ("severity", "evidence_level"):
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int8")

        # cohort filtering (synonyms are already baked into target_ids)
        if filtering:
            subj_in = chunk["subject_drug_drugbank_id"].isin(target_ids)
            aff_in = chunk["affected_drug_drugbank_id"].isin(target_ids)
            mask = (subj_in & aff_in) if filter_mode == "both" else (subj_in | aff_in)
            chunk = chunk[mask]

        total_out += len(chunk)
        if out_csv and len(chunk):
            chunk.to_csv(out_csv, mode="a", index=False, header=not wrote_header)
            wrote_header = True
        if filtering and len(chunk):
            collected.append(chunk)

        if i % 5 == 0 or (max_rows and total_in >= max_rows):
            logger.info(f"[interactions] chunk {i} | read={total_in:,} kept={total_out:,}")
        if max_rows and total_in >= max_rows:
            logger.info(f"[interactions] stopping early at max_rows={max_rows:,} (debug)")
            break

    pct = 100 * total_out / max(total_in, 1)
    logger.info(f"[interactions] DONE read={total_in:,} kept={total_out:,} ({pct:.4f}%)")

    if not filtering:
        logger.info("[interactions] streamed to %s (not held in RAM)", out_csv)
        return None

    result = (pd.concat(collected, ignore_index=True) if collected
              else pd.DataFrame(columns=usecols))
    log_shape("interactions.clean", result)
    return result


# SECTION 6 - Cohort resolution   #

def resolve_drug_name(term, vocabulary_df):
    """
    Resolves a drug term using exact matching, with a fallback
    to regional synonyms for transatlantic clinical safety.
    """
    # Clean the input term
    cleaned_term = str(term).strip().lower()
    
    # 1. Apply Synonym Fallback if the term is regional
    if cleaned_term in SYNONYM_FALLBACK:
        resolved_name = SYNONYM_FALLBACK[cleaned_term]
        print(f"[Synonym Fallback] Mapping regional name '{term}' -> '{resolved_name}'")
    else:
        resolved_name = cleaned_term

    # 2. Search vocabulary dataset for the resolved name (case-insensitive)
    matches = vocabulary_df[vocabulary_df['name'].str.lower() == resolved_name]
    
    if not matches.empty:
        # Return the verified primary DrugBank ID
        return matches.iloc[0]['drugbank_id'], "matched"
    
    return None, "unresolved"

def _expand_synonyms(term: str) -> Set[str]:
    """Generate synonym variants of a term (e.g. iron<->ferrous and regional fallbacks)."""
    low = term.lower().strip()
    
    # 1. Apply regional synonym fallback mapping if applicable
    if low in SYNONYM_FALLBACK:
        resolved = SYNONYM_FALLBACK[low]
        logger.info(f"[Synonym Fallback] Mapping regional name '{term}' -> '{resolved}'")
    else:
        resolved = low
        
    variants = {term, resolved}
    
    # 2. Add chemical class/token synonyms (like iron<->ferrous) for both terms
    for variant in list(variants):
        low_v = variant.lower()
        for group in SYNONYM_GROUPS:
            for a in group:
                if re.search(rf"\b{a}\b", low_v):
                    for b in group:
                        variants.add(re.sub(rf"\b{a}\b", b, low_v))
    return variants


def resolve_cohort_ids(
    drugs: pd.DataFrame,
    cohort: Dict[str, List[str]],
    drug_synonyms_path: Optional[str] = None,
) -> Tuple[Set[str], pd.DataFrame]:
    """
    Resolve cohort drug NAMES to DrugBank ids using drugs.csv `name` (+ optional
    drug_synonyms.csv), with case-insensitive exact match, a starts with fallback,
    and Iron<->Ferrous synonym expansion.

    Returns (set of drugbank_ids, tidy resolution report for auditing).
    """
    name_map: Dict[str, Set[str]] = {}

    def _add(nm, dbid):
        if pd.isna(nm) or pd.isna(dbid):
            return
        name_map.setdefault(str(nm).strip().lower(), set()).add(str(dbid))

    for nm, dbid in zip(drugs["name"], drugs["drugbank_id"]):
        _add(nm, dbid)

    # optionally enrich the lexicon with the synonyms table (id -> drugbank_id)
    if drug_synonyms_path and os.path.exists(drug_synonyms_path):
        try:
            syn = pd.read_csv(drug_synonyms_path)
            id_col = next((c for c in ("drug_id", "id") if c in syn.columns), None)
            syn_col = next((c for c in ("synonym", "name") if c in syn.columns), None)
            if id_col and syn_col:
                # drug_synonyms references numeric drug_id -> map via drugs id crosswalk
                dmap = pd.read_csv(
                    os.path.join(os.path.dirname(drug_synonyms_path), "drugs.csv"),
                    usecols=["id", "drugbank_id"])
                dmap["id"] = pd.to_numeric(dmap["id"], errors="coerce")
                lut = dmap.dropna().set_index("id")["drugbank_id"].to_dict()
                for did, s in zip(syn[id_col], syn[syn_col]):
                    _add(s, lut.get(did))
                logger.info("Cohort lexicon enriched with drug_synonyms.csv")
        except Exception as exc:  # non-fatal enrichment
            logger.warning("Could not use drug_synonyms.csv (%s) - continuing", exc)

    all_names = list(name_map.keys())
    rows = []
    resolved: Set[str] = set()
    for category, terms in cohort.items():
        for term in terms:
            hits: Set[str] = set()
            for variant in _expand_synonyms(term):
                v = variant.strip().lower()
                if v in name_map:                       # exact (case-insensitive)
                    hits |= name_map[v]
                else:                                    # startswith fallback
                    for nm in all_names:
                        if nm.startswith(v):
                            hits |= name_map[nm]
            resolved |= hits
            rows.append({
                "category": category, "term": term,
                "n_matches": len(hits),
                "drugbank_ids": ";".join(sorted(hits)) if hits else "",
                "status": "matched" if hits else "UNRESOLVED",
            })

    report = pd.DataFrame(rows)
    n_unres = int((report["status"] == "UNRESOLVED").sum())
    logger.info(f"Cohort resolution: {len(report):,} terms -> {len(resolved):,} "
                f"DrugBank ids | {n_unres:,} unresolved")
    if n_unres:
        logger.warning("Unresolved cohort terms: %s",
                        ", ".join(report.loc[report.status == "UNRESOLVED", "term"]))
    return resolved, report


# SECTION 7 - Per-drug master table (the merge deliverable) #

def build_drug_master(
    drugs: pd.DataFrame,
    half_lives: pd.DataFrame,
    food: pd.DataFrame,
    keep_ids: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Assemble one analysis-ready row per drug:
        drugbank_id, name, simple_description,
        half_life (+ unit/flags/hours), food_interaction_count, food_interactions_text

    `keep_ids` restricts the master to the study cohort.
    """
    base = drugs if keep_ids is None else drugs[drugs["drugbank_id"].isin(keep_ids)].copy()

    # collapse the many-per-drug food rows into one cell + a count
    food_agg = (food.groupby("drugbank_id")["food_interaction"]
                .agg(food_interaction_count="size",
                     food_interactions_text=lambda s: " || ".join(map(str, s)))
                .reset_index())

    master = (base
              .merge(half_lives, on="drugbank_id", how="left")
              .merge(food_agg, on="drugbank_id", how="left"))
    master["food_interaction_count"] = (
        master["food_interaction_count"].fillna(0).astype("int32"))

    # enforce explicit final order: 'description' right before 'simple_description'
    final_order = [
        "drugbank_id", "name", "description", "simple_description",
        "half_life", "half_life_unit", "half_life_unit_mixed", "half_life_n_records",
        "half_life_hours", "food_interaction_count", "food_interactions_text",
    ]
    ordered = [c for c in final_order if c in master.columns]
    extras = [c for c in master.columns if c not in final_order]  # e.g. if add_hours=False
    master = master[ordered + extras]

    log_shape("drug_master", master,
              extra=f"{master['half_life'].notna().sum():,} with half-life, "
                    f"{(master['food_interaction_count']>0).sum():,} with food data")
    return master


# SECTION 8 - Orchestration                                                   #

def _save(df: pd.DataFrame, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False)
    logger.info(f"saved -> {path} ({len(df):,} rows)")


def run_pipeline(
    base_dir: str,
    out_dir: str,
    chunksize: int = 500_000,
    cohort_filter: bool = True,
    filter_mode: str = "both",
    legacy_id_format: bool = False,
    max_interaction_rows: Optional[int] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    P = lambda f: os.path.join(base_dir, f)

    logger.info("=" * 70)
    logger.info("POLYPHARMACY PIPELINE | base=%s", base_dir)
    logger.info("=" * 70)

    # 0) authoritative crosswalk (from drugs.csv) for correct id standardization
    crosswalk = None if legacy_id_format else build_id_crosswalk(P("drugs.csv"))

    # 1) drugs metadata 
    drugs = process_drugs(P("drugs.csv"))

    # 2) half-lives 
    half_lives = process_half_lives(
        P("structured_pharmacology_half_lives.csv"), crosswalk, legacy_id_format)

    # 3) food interactions + explicit merge with half-lives
    food = process_food_interactions(
        P("food_interactions.csv"), crosswalk, legacy_id_format)
    food_hl = merge_food_with_halflife(food, half_lives)

    # 6) cohort resolution (synonyms) -> id set used to filter structured_drug_interaction csv
    target_ids: Optional[Set[str]] = None
    if cohort_filter:
        target_ids, report = resolve_cohort_ids(
            drugs, COHORT_DRUGS, drug_synonyms_path=P("drug_synonyms.csv"))
        _save(report, out_dir, "cohort_resolution_report.csv")

    # 5) large drug-drug interactions (chunked + filtered)
    #    - filtering ON  : keep the (small) result in RAM, saved once below.
    #    - filtering OFF : stream every cleaned row straight to CSV (memory-safe).
    ddi_out = os.path.join(out_dir, "structured_drug_interactions_clean.csv")
    interactions = process_drug_interactions(
        P("structured_drug_interactions.csv"),
        target_ids=target_ids,
        filter_mode=filter_mode,
        chunksize=chunksize,
        out_csv=None if target_ids is not None else ddi_out,
        max_rows=max_interaction_rows,
    )

    # 7) per-drug master (the merge deliverable)
    master = build_drug_master(drugs, half_lives, food, keep_ids=target_ids)

    # persist processed tables
    _save(half_lives, out_dir, "half_lives_processed.csv")
    _save(food, out_dir, "food_interactions_processed.csv")
    _save(food_hl, out_dir, "food_with_halflife.csv")
    _save(drugs if target_ids is None
          else drugs[drugs["drugbank_id"].isin(target_ids)], out_dir, "drugs_processed.csv")
    _save(master, out_dir, "drug_master.csv")
    if interactions is not None:
        _save(interactions, out_dir, "structured_drug_interactions_clean.csv")

    logger.info("=" * 75)
    logger.info("PIPELINE COMPLETE. Outputs in: %s", out_dir)
    logger.info("=" * 75)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DrugBank polypharmacy preprocessing pipeline")
    p.add_argument("--base-dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="Folder containing the DrugBank CSVs")
    p.add_argument("--out-dir", default=None,
                   help="Output folder (default: <base-dir>/processed_polypharmacy)")
    p.add_argument("--chunksize", type=int, default=500_000,
                   help="Rows per chunk for the large interactions file")
    p.add_argument("--no-cohort-filter", action="store_true",
                   help="Process ALL drug-drug interactions (streams to CSV, memory-safe)")
    p.add_argument("--filter-mode", choices=("both", "either"), default="both",
                   help="Keep interactions where BOTH / EITHER end is in the cohort")
    p.add_argument("--legacy-id-format", action="store_true",
                   help="Force literal DB{05d} id formatting (NOT recommended - see docstring)")
    p.add_argument("--max-interaction-rows", type=int, default=None,
                   help="Debug: stop after reading N rows of the big file")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    _setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    out_dir = args.out_dir or os.path.join(args.base_dir, "processed_polypharmacy")
    try:
        run_pipeline(
            base_dir=args.base_dir,
            out_dir=out_dir,
            chunksize=args.chunksize,
            cohort_filter=not args.no_cohort_filter,
            filter_mode=args.filter_mode,
            legacy_id_format=args.legacy_id_format,
            max_interaction_rows=args.max_interaction_rows,
        )
    except FileNotFoundError as exc:
        logger.error("Aborting - required file missing: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - top-level guard for a batch script
        logger.exception("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
