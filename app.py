"""Clinical Trial Finder: live app (Streamlit).

Searches ClinicalTrials.gov (API v2) on each search. Plain-English summaries are
written on request with Gemini and cached in cache/live_summaries.json.

Run locally:  streamlit run app.py
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st

from summary_prompt import PROMPT_VERSION, build_prompt

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_PATH = PROJECT_DIR / "cache" / "live_summaries.json"
API_URL = "https://clinicaltrials.gov/api/v2/studies"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
SUMMARY_MODEL = "gemini-3.5-flash-lite"  # must match MODEL in summarize.py
RETRYABLE = {429, 500, 503, 504}


DISCLAIMER = (
    "This page uses AI to rewrite simple trial descriptions. It is not medical advice and does not "
    "check your eligibility to participate in any clinical trial. "
    "Talk to your doctor before you decide anything about participating in a clinical trial."
)


AFFILIATION_NOTE = (
    "This is an independent project. It is not affiliated with or endorsed by "
    "ClinicalTrials.gov or the National Library of Medicine."
)

MODIFICATIONS = """
- Plain-English summaries are AI-generated rewrites of each study's brief summary. They are shown
  in place of the brief summary and are labelled as AI-generated.
- Only studies with the status Recruiting or Not yet recruiting are shown, in two groups.
- Only these fields are shown: title, status, conditions, interventions, phase, primary outcome,
  and the AI summary.
- Eligibility criteria and contacts are not shown. Each trial links to its record on ClinicalTrials.gov.
- Studies are matched to the condition you enter, and studies with no listed site in the chosen
  country are excluded.
"""

COUNTRIES = [
    "Israel", "Poland", "United States", "United Kingdom", "Canada", "Germany", "France",
    "Japan", "Australia", "China", "Spain", "Italy", "Netherlands", "Brazil", "India",
]

# Plain-language expansions. A query in this table also matches its full term.
SYNONYMS = {
    "t1d": "type 1 diabetes",
    "t2d": "type 2 diabetes",
    "crohn": "crohn's disease",
}

OPEN_STATUSES = {"RECRUITING": "Recruiting", "NOT_YET_RECRUITING": "Not yet recruiting"}


def get_key(name):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner="Searching ClinicalTrials.gov...")
def search_studies(condition, country):
    params = {
        "query.cond": condition,
        "query.locn": country,
        "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING",
        "pageSize": 100,
        "format": "json",
        "sort": "LastUpdatePostDate:desc",
    }
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    studies = response.json().get("studies", [])
    retrieved = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M")
    return studies, retrieved


def flatten(study):
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    desc = p.get("descriptionModule", {})
    cond = p.get("conditionsModule", {})
    design = p.get("designModule", {})
    arms = p.get("armsInterventionsModule", {})
    outcomes = p.get("outcomesModule", {})
    locations = p.get("contactsLocationsModule", {}).get("locations", [])
    return {
        "nct_id": ident.get("nctId", ""),
        "brief_title": ident.get("briefTitle", ""),
        "status": status.get("overallStatus", ""),
        "last_update": status.get("lastUpdatePostDateStruct", {}).get("date", "not stated"),
        "conditions": cond.get("conditions", []),
        "interventions": [i.get("name", "") for i in arms.get("interventions", [])],
        "phases": design.get("phases", []),
        "primary_outcomes": [o.get("measure", "") for o in outcomes.get("primaryOutcomes", [])],
        "brief_summary": desc.get("briefSummary", ""),
        "countries": {loc.get("country") for loc in locations if loc.get("country")},
    }


def match_level(query, conditions):
    """Return 'exact', 'synonym', 'umbrella', or None for one trial's conditions."""
    q = query.strip().lower()
    if not q:
        return None
    conds = [c.lower() for c in conditions]
    if q in conds:
        return "exact"
    if q in SYNONYMS and any(SYNONYMS[q] in c for c in conds):
        return "synonym"
    if any(q in c for c in conds):
        return "umbrella"
    return None


def filter_results(studies, query, country):
    current, future = [], []
    for study in studies:
        t = flatten(study)
        if t["status"] not in OPEN_STATUSES:
            continue
        if country not in t["countries"]:
            continue
        level = match_level(query, t["conditions"])
        if level is None:
            continue
        t["level"] = level
        (current if t["status"] == "RECRUITING" else future).append(t)
    return current, future


def summary_key(nct_id, brief_summary):
    raw = f"{nct_id}|{SUMMARY_MODEL}|{PROMPT_VERSION}|{brief_summary}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_summary(brief_summary):
    api_key = get_key("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("The summary service is not configured (GEMINI_API_KEY is missing).")
    body = {
        "contents": [{"parts": [{"text": build_prompt(brief_summary)}]}],
        "generationConfig": {"maxOutputTokens": 1024},
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    url = GEMINI_URL.format(model=SUMMARY_MODEL)
    for attempt in range(4):
        response = requests.post(url, headers=headers, json=body, timeout=120)
        if response.status_code == 200:
            data = response.json()
            candidate = data["candidates"][0]
            text = "".join(p.get("text", "") for p in candidate["content"]["parts"]).strip()
            if candidate.get("finishReason") == "MAX_TOKENS":
                raise RuntimeError("The summary was cut off. Please try again.")
            return text
        if response.status_code in RETRYABLE:
            time.sleep(15 * (attempt + 1))
            continue
        raise RuntimeError(f"The summary service returned an error (HTTP {response.status_code}).")
    raise RuntimeError("The summary service is busy. Try again in a few minutes.")


def show_list(items):
    if not items:
        st.write("Not listed.")
    for item in items:
        st.write(f"- {item}")


def summary_block(t):
    st.markdown("**AI summary** (AI-generated, simple English)")
    if not t["brief_summary"]:
        st.caption("The registry has no brief summary for this study.")
        return
    key = summary_key(t["nct_id"], t["brief_summary"])
    cache = load_cache()
    if key in cache:
        st.write(cache[key]["summary"])
        return
    if st.button("Generate Summary", key=f"btn_{t['nct_id']}", type="primary"):
        with st.spinner("Writing summary..."):
            try:
                text = generate_summary(t["brief_summary"])
            except Exception as e:
                st.error(str(e))
                return
        cache[key] = {
            "summary": text,
            "generated_at": datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC"),
        }
        save_cache(cache)
        st.write(text)


def render_trial(t, retrieved):
    phase = ", ".join(t["phases"]) if t["phases"] else "Not listed"
    st.markdown(f"**{OPEN_STATUSES[t['status']]}** · match: {t['level']}")
    st.markdown("**Conditions**")
    show_list(t["conditions"])
    st.markdown("**Interventions**")
    show_list(t["interventions"])
    st.markdown(f"**Phase:** {phase}")
    st.markdown("**Primary outcome (as stated in the registry)**")
    show_list(t["primary_outcomes"])
    summary_block(t)
    st.markdown(
        f"[Eligibility criteria and contacts on ClinicalTrials.gov]"
        f"(https://clinicaltrials.gov/study/{t['nct_id']}) (scroll down on the page)"
    )
    st.caption(
        f"Source: ClinicalTrials.gov. Record last updated by ClinicalTrials.gov: "
        f"{t['last_update']}. Retrieved live: {retrieved} UTC."
    )


def show_results(query, country):
    try:
        studies, retrieved = search_studies(query, country)
    except requests.RequestException:
        st.error("ClinicalTrials.gov could not be reached. Try again in a few minutes.")
        return

    current, future = filter_results(studies, query, country)
    if not current and not future:
        st.info("No recruiting or not-yet-recruiting trial matches this condition in this country.")
        return

    for title, group in (("Currently recruiting", current), ("Not yet recruiting", future)):
        if not group:
            continue
        st.subheader(f"{title} ({len(group)})")
        for t in group:
            with st.expander(f"{t['nct_id']} · {t['brief_title'] or t['nct_id']}"):
                render_trial(t, retrieved)


def footer():
    st.divider()
    st.caption(
        "Source: ClinicalTrials.gov. Results are retrieved live when you search. The date on "
        "each trial is the date ClinicalTrials.gov last updated its record."
    )
    st.caption(AFFILIATION_NOTE)
    with st.expander("How this page changes the registry data"):
        st.markdown(MODIFICATIONS)


def main():
    st.set_page_config(page_title="Clinical Trial Finder", layout="centered")
    st.markdown(
        """<style>
        input, [data-baseweb="select"] *, [data-baseweb="input"] * { color: #0D1B2A !important; }
        input::placeholder { color: #455A64 !important; }
        button[kind="primary"] * { color: #0D1B2A !important; }
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("Clinical Trial Finder")
    st.warning(DISCLAIMER)

    with st.form("search"):
        query = st.text_input("Medical condition", placeholder="e.g. type 1 diabetes, T1D, Crohn's disease")
        country = st.selectbox("Country", COUNTRIES)
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        if not query.strip():
            st.warning("Enter a medical condition to search.")
        else:
            st.session_state["last_search"] = (query.strip(), country)

    if "last_search" in st.session_state:
        show_results(*st.session_state["last_search"])

    footer()


main()
