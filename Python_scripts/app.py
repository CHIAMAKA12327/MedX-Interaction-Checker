# -*- coding: utf-8 -*-
"""
============================================================================
 MedX  |  Drug Interaction Checker
============================================================================
A clinical-grade Streamlit application that surfaces potentially harmful
drug-drug, food, supplement and alcohol interactions using a DETERMINISTIC,
non-generative text-mapping pipeline (no generative AI, no hallucinations).

MedX Health Informatics
Runtime : streamlit run app.py
============================================================================
"""

from __future__ import annotations

import os
import csv
import html
import random
from datetime import datetime

import pandas as pd
import streamlit st

# ===========================================================================
# 1. SYSTEM CHANNELS & LOCAL PATH SEGREGATION  (exact paths, no exception)
# ===========================================================================
INGESTION_PATH = "Processed_outputs/final_patient_guidance_analytics.csv"
AUDIT_PATH = "Processed_outputs/user_testing_audits.csv"
FOOD_PATH = "Processed_outputs/food_with_halflife.csv"

AUDIT_COLUMNS = [
    "timestamp",
    "reviewer_id",
    "user_type",
    "clinical_accuracy_preserved",
    "linguistic_suitability",
    "drugs_evaluated",
    "matched_interactions",
    "highest_severity",
]

SEV_META = {
    "major": {
        "cls": "sev-major",
        "badge": "🚨🚨🚨 Critical Alert",
        "title": "Major Interaction",
        "rank": 3,
    },
    "moderate": {
        "cls": "sev-moderate",
        "badge": "🚨 Alert",
        "title": "Moderate Interaction",
        "rank": 2,
    },
    "minor": {
        "cls": "sev-minor",
        "badge": "Minor Interaction Observed",
        "title": "Minor Interaction",
        "rank": 1,
    },
}

# ===========================================================================
# 2. WELLNESS & LIFESTYLE CONTENT
# ===========================================================================
WELLNESS_CARDS = [
    {"emoji": "🫐", "produce": "Blueberries",
     "benefit": "Packed with anthocyanin antioxidants that support memory, healthy blood vessels and heart function.",
     "tip": "Catch 10–15 minutes of early-morning sunlight — it anchors your circadian rhythm for deeper, better sleep."},
    {"emoji": "🥬", "produce": "Spinach",
     "benefit": "A rich source of iron, folate and vitamin K for healthy blood, energy and strong bones.",
     "tip": "A brisk 20-minute daily walk gently lowers blood pressure and lifts your mood."},
    {"emoji": "🥦", "produce": "Broccoli",
     "benefit": "Contains sulforaphane, which supports the body's natural detoxifying enzymes and cell protection.",
     "tip": "Light jogging strengthens your heart and lungs — start with 2 minutes and build up gradually."},
    {"emoji": "🍊", "produce": "Oranges",
     "benefit": "High in vitamin C to support immunity, collagen and healthy skin.",
     "tip": "Morning daylight within an hour of waking improves daytime alertness and night-time melatonin."},
    {"emoji": "🍅", "produce": "Tomatoes",
     "benefit": "Their lycopene is linked to heart and skin health, especially when lightly cooked.",
     "tip": "Five minutes of slow, deep breathing measurably lowers stress hormones."},
    {"emoji": "🥕", "produce": "Carrots",
     "benefit": "Beta-carotene supports eye health, immunity and a healthy glow.",
     "tip": "Stand up and stretch every hour to protect your back and keep circulation flowing."},
    {"emoji": "🥑", "produce": "Avocado",
     "benefit": "Rich in heart-friendly monounsaturated fats that help balance cholesterol.",
     "tip": "Aim for 7–9 hours of consistent sleep — it powers recovery, focus and a steady metabolism."},
    {"emoji": "🍌", "produce": "Bananas",
     "benefit": "A great source of potassium for healthy blood pressure and muscle function.",
     "tip": "A gentle 20-minute walk after meals helps steady your blood sugar."},
    {"emoji": "🍎", "produce": "Apples",
     "benefit": "Their soluble fibre (pectin) supports gut health and healthy cholesterol.",
     "tip": "An early-morning walk in natural daylight helps set your body clock for the day."},
    {"emoji": "🍠", "produce": "Sweet Potato",
     "benefit": "Slow-release carbohydrates plus beta-carotene for steady energy and immunity.",
     "tip": "Brisk 20-minute walks on most days are one of the simplest ways to protect your heart."},
    {"emoji": "🫑", "produce": "Bell Peppers",
     "benefit": "Exceptionally high in vitamin C — even more than citrus — to power your immune system.",
     "tip": "Take short movement breaks through the day to offset long periods of sitting."},
    {"emoji": "🥬", "produce": "Kale",
     "benefit": "Loaded with vitamins A, C and K for immunity, vision and bone strength.",
     "tip": "A few minutes of gentle stretching before bed can noticeably improve sleep quality."},
    {"emoji": "🫒", "produce": "Beetroot",
     "benefit": "Natural dietary nitrates support healthy blood flow and physical stamina.",
     "tip": "Jogging 2–3 times a week gradually improves your lung capacity and endurance."},
    {"emoji": "🧄", "produce": "Garlic",
     "benefit": "Its allicin compound is associated with healthy circulation and immune support.",
     "tip": "Keep well hydrated — sip water steadily rather than in large bursts."},
    {"emoji": "🫚", "produce": "Ginger",
     "benefit": "A natural anti-inflammatory that soothes digestion and eases nausea.",
     "tip": "Morning sunlight also supports natural vitamin D production and a brighter mood."},
]

# ===========================================================================
# 3. PAGE CONFIGURATION
# ===========================================================================
st.set_page_config(
    page_title="MedX · Drug Interaction Checker",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===========================================================================
# 4. GLOBAL STYLESHEET
# ===========================================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        :root {
            --brand:      #0066cc;
            --brand-dark: #004a99;
            --brand-soft: #e8f1fc;
            --ink:        #1e293b;
            --ink-soft:   #475569;
            --muted:      #64748b;
            --line:       #e2e8f0;
            --bg:         #f5f8fc;
            --card:       #ffffff;
            --radius:     18px;
            --shadow:     0 10px 30px rgba(15, 40, 80, 0.08);
            --shadow-sm:  0 4px 14px rgba(15, 40, 80, 0.06);
        }
        html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
            color: var(--ink);
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(1100px 520px at 12% -8%, #eaf2fd 0%, rgba(234,242,253,0) 55%),
                radial-gradient(1000px 520px at 105% 0%, #e9fbf6 0%, rgba(233,251,246,0) 50%),
                var(--bg);
        }
        [data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
        .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }
        .medx-wordmark {
            font-size: 4.4rem; font-weight: 800; letter-spacing: -2px; line-height: 1;
            background: linear-gradient(120deg, var(--brand) 0%, #2aa9e0 55%, #14b8a6 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; margin: 0;
        }
        .medx-sub { font-size: 1.35rem; font-weight: 600; color: var(--ink-soft); margin: .35rem 0 0 0; letter-spacing: .2px; }
        .medx-pill {
            display:inline-block; margin-bottom: .9rem; padding: .35rem .9rem;
            background: var(--brand-soft); color: var(--brand-dark);
            border-radius: 999px; font-size: .8rem; font-weight: 700;
            letter-spacing: .8px; text-transform: uppercase;
        }
        .medx-lede { font-size: 1.06rem; color: var(--ink-soft); line-height: 1.65; max-width: 760px; margin: 1rem 0 0 0; }
        .surface { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow-sm); padding: 1.6rem 1.7rem; }
        .section-title { display:flex; align-items:center; gap:.7rem; font-size: 1.32rem; font-weight: 700; color: var(--ink); margin: 2.1rem 0 1rem 0; }
        .section-title::before { content:""; width: 6px; height: 26px; border-radius: 6px; background: linear-gradient(180deg, var(--brand), #14b8a6); }
        .section-note { color: var(--muted); font-size: .95rem; margin:-.4rem 0 1rem 0; }
        .search-hero { background: var(--card); border: 1px solid var(--line); border-radius: 22px; box-shadow: var(--shadow); padding: 1.9rem 2rem 1.4rem 2rem; margin-top: .4rem; }
        .search-label { font-weight: 700; font-size: 1.05rem; color: var(--ink); margin-bottom:.2rem; }
        .stButton > button, .stFormSubmitButton > button { border-radius: 12px; font-weight: 700; font-size: 1rem; padding: .62rem 1.3rem; border: 1px solid var(--line); background: #ffffff; color: var(--brand-dark); transition: all .18s ease; }
        .stButton > button:hover, .stFormSubmitButton > button:hover { border-color: var(--brand); color: var(--brand); transform: translateY(-1px); box-shadow: var(--shadow-sm); }
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"], [data-testid="baseButton-primary"], [data-testid="stBaseButton-primary"] { background: linear-gradient(120deg, var(--brand) 0%, var(--brand-dark) 100%); color: #ffffff !important; border: none; box-shadow: 0 8px 20px rgba(0, 102, 204, 0.28); }
        .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover { transform: translateY(-2px); box-shadow: 0 12px 26px rgba(0, 102, 204, 0.36); color: #ffffff !important; }
        [data-baseweb="tag"] { background-color: var(--brand) !important; border-radius: 8px !important; }
        [data-baseweb="select"] > div { border-radius: 12px !important; border-color: var(--line) !important; min-height: 52px; }
        .wellness { border-radius: 20px; padding: 1.5rem 1.7rem; background: linear-gradient(120deg, #ecfdf5 0%, #eff6ff 100%); border: 1px solid #cfece0; box-shadow: var(--shadow-sm); }
        .wellness-tag { font-size:.78rem; font-weight:700; letter-spacing:.7px; text-transform:uppercase; color:#0f766e; }
        .wellness-head { display:flex; align-items:center; gap:.8rem; margin:.3rem 0 .5rem 0; }
        .wellness-emoji { font-size: 2.5rem; line-height:1; }
        .wellness-produce { font-size: 1.4rem; font-weight: 800; color:#0f172a; }
        .wellness-benefit { color: var(--ink-soft); line-height:1.6; margin-bottom:.7rem; }
        .wellness-tip { background: rgba(255,255,255,.75); border-left: 4px solid #14b8a6; border-radius: 10px; padding:.7rem .9rem; color:#0f172a; font-weight:500; }
        .wellness-foot { color:#0f766e; font-size:.8rem; margin-top:.7rem; }
        .stat-row { display:flex; gap:1rem; flex-wrap:wrap; margin: .2rem 0 .4rem 0; }
        .stat { flex:1; min-width: 150px; background: var(--card); border:1px solid var(--line); border-radius: 16px; padding: 1rem 1.2rem; box-shadow: var(--shadow-sm); }
        .stat .n { font-size: 2rem; font-weight: 800; line-height:1; }
        .stat .l { color: var(--muted); font-size:.85rem; font-weight:600; margin-top:.35rem; text-transform:uppercase; letter-spacing:.5px; }
        .n-total { color: var(--brand); }
        .n-major { color: #b91c1c; }
        .n-moderate { color: #d97706; }
        .n-minor { color: #ca8a04; }
        .ix-card { border-radius: 16px; padding: 1.25rem 1.4rem; margin-bottom: .4rem; box-shadow: var(--shadow-sm); }
        .ix-head { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin-bottom:.6rem; }
        .ix-pair { font-size: 1.16rem; font-weight: 800; }
        .ix-badge { font-weight: 800; font-size: .92rem; padding:.32rem .8rem; border-radius: 999px; white-space: nowrap; }
        .ix-guidance { line-height: 1.62; font-size: 1.01rem; }
        .sev-none { background:#f0fdf4; border:1px solid #bbf7d0; }
        .sev-none .ix-pair { color:#166534; }
        .sev-none .ix-badge { background:#dcfce7; color:#166534; }
        .sev-none .ix-guidance { color:#15803d; }
        .sev-minor { background:#fffbeb; border:1px solid #fde68a; }
        .sev-minor .ix-pair { color:#92400e; }
        .sev-minor .ix-badge { background:#fef3c7; color:#92400e; }
        .sev-minor .ix-guidance { color:#78350f; }
        .sev-moderate { background:#fff5f5; border:2px solid #ef4444; }
        .sev-moderate .ix-pair { color:#b91c1c; }
        .sev-moderate .ix-badge { background:#fee2e2; color:#b91c1c; }
        .sev-moderate .ix-guidance { color:#7f1d1d; }
        .sev-major { background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); border:2px solid #450a0a; }
        .sev-major .ix-pair { color:#ffffff; }
        .sev-major .ix-badge { background:#450a0a; color:#fecaca; }
        .sev-major .ix-guidance { color:#fee2e2; }
        .food-drug { font-weight:800; font-size:1.06rem; color:var(--ink); margin:.2rem 0 .5rem 0; }
        .food-item { display:flex; gap:.65rem; align-items:flex-start; background:#ffffff; border:1px solid var(--line); border-radius:12px; padding:.7rem .9rem; margin-bottom:.5rem; box-shadow: var(--shadow-sm); }
        .food-ic { font-size:1.15rem; line-height:1.4; }
        .food-tx { color:var(--ink-soft); line-height:1.55; }
        .legend { display:flex; gap:.6rem; flex-wrap:wrap; margin:.2rem 0 1rem 0; }
        .legend span { font-size:.82rem; font-weight:700; padding:.3rem .7rem; border-radius:999px; }
        .disclaimer { margin-top: 2.4rem; padding: 1.1rem 1.3rem; border-radius: 14px; background:#fff7ed; border:1px solid #fed7aa; color:#7c2d12; font-size:.92rem; line-height:1.6; }
        .disclaimer b { color:#9a3412; }
        [data-testid="stExpander"] { border:1px solid var(--line); border-radius: 12px; background:#fff; }
        hr.soft { border:none; border-top:1px solid var(--line); margin:1.6rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# 5. DATA LAYER
# ===========================================================================
@st.cache_data(show_spinner=False)
def load_interactions(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    df.columns = [c.strip() for c in df.columns]
    for col in ("subject_drug_name", "affected_drug_name", "subject_drug_drugbank_id", "affected_drug_drugbank_id", "severity_label"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_food(path: str, mtime: float) -> pd.DataFrame | None:
    try:
        fd = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        fd.columns = [c.strip() for c in fd.columns]
        if "drugbank_id" not in fd.columns or "food_interaction" not in fd.columns:
            return None
        fd["drugbank_id"] = fd["drugbank_id"].astype("string").str.strip()
        fd = fd.dropna(subset=["drugbank_id", "food_interaction"])
        return fd
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def build_indexes(df: pd.DataFrame):
    names: set[str] = set()
    name2id: dict[str, str] = {}
    pairs = [("subject_drug_name", "subject_drug_drugbank_id"), ("affected_drug_name", "affected_drug_drugbank_id")]
    for nc, ic in pairs:
        if nc not in df.columns:
            continue
        for nm, did in zip(df[nc], df.get(ic, pd.Series([None] * len(df)))):
            if isinstance(nm, str) and nm.strip():
                nm = nm.strip()
                names.add(nm)
                if isinstance(did, str) and did.strip():
                    name2id.setdefault(nm, did.strip())
    return sorted(names, key=str.lower), name2id

@st.cache_data(show_spinner=False)
def build_food_map(fd: pd.DataFrame | None):
    if fd is None:
        return {}
    fmap: dict[str, list[str]] = {}
    for did, txt in zip(fd["drugbank_id"], fd["food_interaction"]):
        if not isinstance(txt, str) or not txt.strip():
            continue
        bucket = fmap.setdefault(str(did), [])
        t = txt.strip()
        if t not in bucket:
            bucket.append(t)
    return fmap

# ===========================================================================
# 6. SMALL HELPERS
# ===========================================================================
def _rerun() -> None:
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def _sev_key(label) -> str:
    return str(label).strip().lower() if isinstance(label, str) else ""

def _fmt_hours(val) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        f = float(s)
    except (ValueError, TypeError):
        return s
    if f != f:
        return None
    if f >= 100:
        return f"{f:.0f}"
    return f"{f:.1f}" if f >= 10 else f"{f:.2f}"

def _food_icon(text: str) -> str:
    t = text.lower()
    if "alcohol" in t: return "🍷"
    if "grapefruit" in t: return "🍊"
    if any(k in t for k in ("supplement", "st. john", "st john", "herb", "vitamin", "calcium", "iron", "zinc", "ginkgo", "garlic", "ginger", "potassium", "magnesium", "folate", "folic")): return "🌿"
    if any(k in t for k in ("caffeine", "coffee", "tea")): return "☕"
    if t.startswith("take") or "with food" in t or "with a meal" in t or "after a meal" in t: return "🍽️"
    return "🥗"

def render_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer">
        <b>Disclaimer:</b> The content on MedX is intended for information only and should
        <b>NOT</b> be considered professional medical advice. Speak to your healthcare
        provider for guidance.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# 7. AUDIT & TRAFFIC NETWORK WRITERS
# ===========================================================================
def append_audit(row: dict) -> tuple[bool, str]:
    import requests
    import json
    try:
        GOOGLE_WEBAPP_URL = "https://script.google.com/macros/s/AKfycbzio2rXSCnd0z6Sg5-WnoLvMyraAn51_xbK5wRxXkaMIJuFa302VNLLR5ROAXdPfCTR/exec"
        response = requests.post(GOOGLE_WEBAPP_URL, data=json.dumps(row), headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            return True, "Evaluation recorded safely in the cloud spreadsheet ledger."
        return False, f"Spreadsheet connection returned status code: {response.status_code}"
    except Exception as exc:
        return False, f"Could not sync with cloud ledger ({type(exc).__name__})."

def log_traffic_event(event_type: str, details: str = "") -> None:
    import requests
    import json
    from datetime import datetime
    try:
        GOOGLE_WEBAPP_URL = "https://script.google.com/macros/s/AKfycbzio2rXSCnd0z6Sg5-WnoLvMyraAn51_xbK5wRxXkaMIJuFa302VNLLR5ROAXdPfCTR/exec"
        payload = {
            "log_type": "traffic",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            "details": details
        }
        requests.post(GOOGLE_WEBAPP_URL, data=json.dumps(payload), headers={"Content-Type": "application/json"})
    except Exception as e:
        st.session_state["traffic_debug_error"] = str(e)

# ===========================================================================
# 8. VIEW 1  —  THE MEDX LANDING PORTAL
# ===========================================================================
def render_landing(names: list[str]) -> None:
    SYNONYM_LOOKUP = {
        "acetaminophen": "Acetaminophen / Paracetamol",
        "acetylsalicylic acid": "Acetylsalicylic acid / Aspirin",
        "ascorbic acid": "Ascorbic acid / Vitamin C"
    }
    
    display_to_canonical = {}
    dropdown_options = []
    for n in names:
        disp = SYNONYM_LOOKUP.get(n.lower().strip(), n)
        dropdown_options.append(disp)
        display_to_canonical[disp] = n
    dropdown_options = sorted(list(set(dropdown_options)))

    st.markdown('<div class="medx-pill">Polypharmacy Safety Platform</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="medx-wordmark">MedX</h1>', unsafe_allow_html=True)
    st.markdown('<p class="medx-sub">Drug Interaction Checker</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="medx-lede">Welcome to MedX, your intelligent drug interaction '
        'checker. Find potentially harmful drug, food, supplement, and alcohol '
        'interactions instantly to manage patient safety.</p>',
        unsafe_allow_html=True,
    )

    st.write("")

    with st.container():
        if "traffic_debug_error" in st.session_state:
            st.error(f"⚠️ Hidden Traffic Logger Error: {st.session_state['traffic_debug_error']}")
            del st.session_state["traffic_debug_error"]
            
        st.markdown('<div class="search-hero">', unsafe_allow_html=True)
        st.markdown('<div class="search-label">🔎 Search medications</div>', unsafe_allow_html=True)
        st.caption("Select one medicine to view its full interaction profile, or two or more to check how they interact with each other.")

        selected = st.multiselect(
            label="Search medications",
            options=dropdown_options,
            default=st.session_state.get("ms_drugs", []),
            key="ms_drugs",
            placeholder="Start typing a drug name — e.g. Paracetamol, Aspirin, Vitamin C…",
            label_visibility="collapsed",
        )

        col_a, col_b = st.columns([1, 2.4])
        with col_a:
            check = st.button("🔍  Check Interactions", type="primary", use_container_width=True)
        with col_b:
            if selected:
                st.markdown(
                    f'<div style="padding-top:.55rem;color:var(--muted);font-weight:600;">'
                    f'{len(selected)} selected · '
                    f'{"drug-drug interaction mode" if len(selected) > 1 else "single-drug profile mode"}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)

    if check:
        if not selected:
            st.warning("Please select at least one medication to check.")
        else:
            searched_drugs_string = ", ".join(selected)
            log_traffic_event(event_type="Interaction Lookup", details=f"Queried Combination: {searched_drugs_string}")
            st.session_state["checked_drugs"] = [display_to_canonical[disp] for disp in selected]
            st.session_state["view"] = "results"
            _rerun()
        
    st.markdown('<div class="section-title">Daily Wellness & Lifestyle</div>', unsafe_allow_html=True)
    idx = st.session_state.get("wellness_idx", 0)
    idx = max(0, min(idx, len(WELLNESS_CARDS) - 1))
    card = WELLNESS_CARDS[idx]

    st.markdown(
        f"""
        <div class="wellness">
            <div class="wellness-tag">🌱 Wellness spotlight</div>
            <div class="wellness-head">
                <div class="wellness-emoji">{card['emoji']}</div>
                <div class="wellness-produce">{html.escape(card['produce'])}</div>
            </div>
            <div class="wellness-benefit">{html.escape(card['benefit'])}</div>
            <div class="wellness-tip">💡 <b>Lifestyle tip:</b> {html.escape(card['tip'])}</div>
            <div class="wellness-foot">General wellbeing guidance — always check specific foods against your own medicines using the checker above.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    if st.button("🔄  Show me another tip"):
        choices = [i for i in range(len(WELLNESS_CARDS)) if i != idx] or [idx]
        st.session_state["wellness_idx"] = random.choice(choices)
        _rerun()

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    with st.expander("ℹ️  About the MedX App"):
        st.markdown(
            """
**MedX** is a decision-support tool that helps patients and clinicians spot potentially harmful **drug–drug, food, supplement and alcohol interactions** before they cause harm.
            """
        )
    render_disclaimer()

# ===========================================================================
# 9. VIEW 2  —  INTERACTIVE RESULTS & DIAGNOSTIC PORTAL
# ===========================================================================
def _filter_interactions(df: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    sset = set(selected)
    subj = df["subject_drug_name"]
    aff = df["affected_drug_name"]
    if len(sset) == 1:
        mask = subj.isin(sset) | aff.isin(sset)
    else:
        mask = subj.isin(sset) & aff.isin(sset)

    sub = df[mask].copy()
    if sub.empty:
        return sub

    sub["_rank"] = sub["severity_label"].map(lambda s: SEV_META.get(_sev_key(s), {}).get("rank", 0))
    sub["_pair"] = [" || ".join(sorted([str(a), str(b)])) for a, b in zip(sub["subject_drug_name"], sub["affected_drug_name"])]
    sub = (sub.sort_values("_rank", ascending=False)
              .drop_duplicates("_pair", keep="first")
              .sort_values(["_rank", "subject_drug_name"], ascending=[False, True])
              .reset_index(drop=True))
    return sub

def _render_stat_row(sub: pd.DataFrame) -> None:
    total = len(sub)
    counts = {"major": 0, "moderate": 0, "minor": 0}
    for lbl in sub["severity_label"]:
        k = _sev_key(lbl)
        if k in counts: counts[k] += 1
    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat"><div class="n n-total">{total}</div><div class="l">Interactions found</div></div>
            <div class="stat"><div class="n n-major">{counts['major']}</div><div class="l">🚨🚨🚨 Major</div></div>
            <div class="stat"><div class="n n-moderate">{counts['moderate']}</div><div class="l">🚨 Moderate</div></div>
            <div class="stat"><div class="n n-minor">{counts['minor']}</div><div class="l">Minor</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _render_interaction_card(row: pd.Series, primary_set: set[str]) -> None:
    subj = str(row.get("subject_drug_name", "") or "")
    aff = str(row.get("affected_drug_name", "") or "")
    a, b = (subj, aff) if (subj in primary_set or aff not in primary_set) else (aff, subj)
    key = _sev_key(row.get("severity_label"))
    meta = SEV_META.get(key, {"cls": "sev-minor", "badge": "Interaction", "title": "Interaction"})
    guidance = row.get("patient_friendly_guidance")
    
    if not isinstance(guidance, str) or not guidance.strip():
        guidance = {"major": "Critical Patient Warning: Avoid simultaneous intake...", "moderate": "Patient Guidance: Clinical adjustment...", "minor": "Patient Guidance: One of the medications..."}.get(key, "Consult provider.")

    st.markdown(f'<div class="ix-card {meta["cls"]}"><div class="ix-head"><span class="ix-pair">{html.escape(a)} &nbsp;⇄&nbsp; {html.escape(b)}</span><span class="ix-badge">{meta["badge"]}</span></div><div class="ix-guidance">{html.escape(guidance)}</div></div>', unsafe_allow_html=True)

    with st.expander(f"🩺 Clinical detail · {a} ⇄ {b}"):
        summary = row.get("summary")
        ext = row.get("extended_description")
        mgmt = row.get("management")
        ev = row.get("evidence_definition")
        if isinstance(summary, str) and summary.strip(): st.markdown(f"**Clinical summary**  \n{summary.strip()}")
        if isinstance(ext, str) and ext.strip(): st.markdown(f"**Mechanism / detail**  \n{ext.strip()}")
        if isinstance(mgmt, str) and mgmt.strip(): st.markdown(f"**Management**  \n{mgmt.strip()}")
        if isinstance(ev, str) and ev.strip(): st.markdown(f"**Evidence level**  \n{ev.strip()}")

def _render_food_section(selected: list[str], name2id: dict, food_map: dict) -> None:
    st.markdown('<div class="section-title">Food, Supplement & Alcohol Interactions</div>', unsafe_allow_html=True)
    if not food_map:
        st.info("Food advisories are not available.")
        return
    any_found = False
    for name in selected:
        did = name2id.get(name)
        items = food_map.get(str(did), []) if did else []
        if not items: continue
        any_found = True
        st.markdown(f'<div class="food-drug">💊 {html.escape(name)}</div>', unsafe_allow_html=True)
        rows_html = "".join(f'<div class="food-item"><div class="food-ic">{_food_icon(it)}</div><div class="food-tx">{html.escape(it)}</div></div>' for it in items)
        st.markdown(rows_html, unsafe_allow_html=True)

    if not any_found:
        st.markdown('<div class="ix-card sev-none"><div class="ix-guidance">No specific food advisories documented.</div></div>', unsafe_allow_html=True)

def _render_evaluation_panel(selected: list[str], matched: int, highest: str) -> None:
    st.markdown('<div class="section-title">Expert Validation Logging</div>', unsafe_allow_html=True)
    with st.form("evaluation_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1: acc = st.checkbox("Clinical Accuracy Preserved", value=True)
        with c2: ling = st.checkbox("Linguistic Suitability for Patients", value=True)
        c3, c4 = st.columns(2)
        with c3: user_type = st.selectbox("User Type", ["Medical Professional", "Patient"])
        with c4: reviewer = st.text_input("Reviewer Name / ID")
        submitted = st.form_submit_button("✅  Submit Evaluation", type="primary")

    if submitted:
        if not reviewer.strip():
            st.warning("Please enter a Reviewer Name / ID.")
            return
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reviewer_id": reviewer.strip(),
            "user_type": user_type,
            "clinical_accuracy_preserved": "Yes" if acc else "No",
            "linguistic_suitability": "Yes" if ling else "No",
            "drugs_evaluated": "; ".join(selected),
            "matched_interactions": matched,
            "highest_severity": highest,
        }
        ok, msg = append_audit(record)
        if ok: st.success(f"✅ {msg}")
        else: st.error(f"⚠️ {msg}")

def render_results(df: pd.DataFrame, name2id: dict, food_map: dict) -> None:
    selected = st.session_state.get("checked_drugs", [])
    nav_l, nav_r = st.columns([3, 1])
    with nav_l:
        st.markdown('<h1 class="medx-wordmark" style="font-size:2.6rem;">MedX</h1>', unsafe_allow_html=True)
        st.markdown('<p class="medx-sub" style="font-size:1rem;margin-top:.1rem;">Interaction Results</p>', unsafe_allow_html=True)
    with nav_r:
        st.write("")
        if st.button("←  New Search", use_container_width=True):
            st.session_state["view"] = "landing"
            _rerun()

    if not selected:
        st.info("No medications selected.")
        return

    SYNONYM_LOOKUP = {"acetaminophen": "Acetaminophen / Paracetamol", "acetylsalicylic acid": "Acetylsalicylic acid / Aspirin", "ascorbic acid": "Ascorbic acid / Vitamin C"}
    display_tags = [SYNONYM_LOOKUP.get(n.lower().strip(), n) for n in selected]
    tags = "  ".join(f'<span style="background:var(--brand-soft);color:var(--brand-dark);padding:.3rem .8rem;border-radius:999px;font-weight:700;font-size:.9rem;">{html.escape(t)}</span>' for t in display_tags)
    st.markdown(f'<div style="margin:.4rem 0 1rem 0;">{tags}</div>', unsafe_allow_html=True)

    try:
        sub = _filter_interactions(df, selected)
    except Exception:
        st.error("Problem filtering.")
        return

    st.markdown('<div class="legend"><span style="background:#dcfce7;color:#166534;">🟢 None</span><span style="background:#fef3c7;color:#92400e;">🟡 Minor</span><span style="background:#fee2e2;color:#b91c1c;">🚨 Moderate</span><span style="background:#7f1d1d;color:#fecaca;">🚨🚨🚨 Major</span></div>', unsafe_allow_html=True)

    highest = "none"
    if sub.empty:
        st.markdown('<div class="ix-card sev-none"><div class="ix-guidance">No documented interactions found.</div></div>', unsafe_allow_html=True)
    else:
        _render_stat_row(sub)
        for cand in ("major", "moderate", "minor"):
            if any(_sev_key(x) == cand for x in sub["severity_label"]):
                highest = cand
                break
        st.markdown('<div class="section-title">Drug–Drug Interactions</div>', unsafe_allow_html=True)
        for _, row in sub.iterrows():
            _render_interaction_card(row, set(selected))

    _render_food_section(selected, name2id, food_map)
    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    _render_evaluation_panel(selected, matched=int(len(sub)), highest=highest)
    render_disclaimer()

# ===========================================================================
# 10. APPLICATION ENTRY POINT
# ===========================================================================
def main() -> None:
    inject_css()

    st.session_state.setdefault("view", "landing")
    if "wellness_idx" not in st.session_state:
        st.session_state["wellness_idx"] = random.randrange(len(WELLNESS_CARDS))

    # Core Page Visit logging initialization
    if "traffic_logged" not in st.session_state:
        log_traffic_event(event_type="Page Visit", details="User initialized MedX web app application interface.")
        st.session_state["traffic_logged"] = True

    if not os.path.isfile(INGESTION_PATH):
        st.error("Clinical dataset could not be located.")
        return

    try:
        df = load_interactions(INGESTION_PATH, os.path.getmtime(INGESTION_PATH))
    except Exception:
        st.error("Unable to read dataset.")
        return

    names, name2id = build_indexes(df)
    food_df = load_food(FOOD_PATH, os.path.getmtime(FOOD_PATH)) if os.path.isfile(FOOD_PATH) else None
    food_map = build_food_map(food_df)

    try:
        if st.session_state.get("view") == "results":
            render_results(df, name2id, food_map)
        else:
            render_landing(names)
    except Exception:
        st.error("Error rendering page components.")

if __name__ == "__main__":
    main()