#!/usr/bin/env python3
"""
Streamlit SEC Filing QA Pipeline
Upload JSON credentials and API key file, then run the pipeline.
"""

import os
import re
import time
import json
import tempfile
import html
import streamlit as st
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import threading

# ----------------------------
# Streamlit App Configuration
# ----------------------------
st.set_page_config(
    page_title="SEC Filing QA Pipeline",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Core Functions: Imports
# ----------------------------
try:
    import requests
    from bs4 import BeautifulSoup
    from openai import OpenAI
    import tiktoken
    import gspread
    from google.oauth2.service_account import Credentials
    from gspread.exceptions import APIError

    CORE_LIBS_AVAILABLE = True
except ImportError as e:
    st.error(f"Missing required libraries: {e}")
    st.info("Please install required packages: pip install streamlit requests beautifulsoup4 openai tiktoken gspread google-auth")
    CORE_LIBS_AVAILABLE = False

# ----------------------------
# Defaults / Config
# ----------------------------
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_CONTEXT_TOKENS = 128000
USER_AGENT = (
    "SEC-Filing-QA/1.0 "
    "(Abhishek Kumar; abhishek.your_email@example.com)"
)

SLEEP_BETWEEN_CALLS = 0.6

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.5
DEFAULT_MAX_TOKENS_PER_QUESTION = 600

# Parallelism controls
MAX_GPT_CONCURRENCY = int(os.getenv("GPT_MAX_CONCURRENCY", "4"))
MAX_FILE_CONCURRENCY = int(os.getenv("FILE_MAX_CONCURRENCY", "3"))
gpt_semaphore = threading.Semaphore(MAX_GPT_CONCURRENCY)

# Rate-limit/Quota flags
GPT_DISABLED = threading.Event()      # permanently disable GPT when quota exceeded
GPT_RATE_LIMITED = threading.Event()  # transient rate-limit indicator

# OpenAI API key (env fallback if no file uploaded)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

SYSTEM_PROMPT = """You are an expert SEC filing analyst. Extract, verify and summarize information from the provided filing text.
When asked for structured output, return JSON only when explicitly requested. 
If none found or the information is not present in the text, return exactly 'none' (lowercase, no quotes, no other text).
Be precise and factual. Do not make up information."""

# Initialize OpenAI client (will be set after file upload)
client = None
tokenizer = None
CLIENTS: List[object] = []
thread_local = threading.local()

# Simple in-memory log buffer
LOG_BUFFER: List[str] = []
ATTEMPT_COUNTS: Dict[str, int] = {}

def log(msg: str):
    ts = datetime.now().isoformat()
    entry = f"[{ts}] {msg}"
    LOG_BUFFER.append(entry)
    # Keep buffer bounded
    if len(LOG_BUFFER) > 5000:
        del LOG_BUFFER[:2000]

# ----------------------------
# UI Helpers (Professional Dark Theme)
# ----------------------------
def inject_crazy_styles():
    """Inject a professional dark theme with clear hierarchy and semantics."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #080d1a;
            --panel: rgba(18, 26, 43, 0.75);
            --glass: rgba(255, 255, 255, 0.06);
            --border: #1c2947;
            --text: #eaf0ff;
            --muted: #a2b0d4;
            /* Semantics */
            --blue: #82b1ff;        /* primary */
            --green: #6fe3a2;       /* success */
            --yellow: #ffd36e;      /* warning */
            --red: #ff6b6b;         /* error */
            --grey: #7887a6;        /* disabled */
        }
                    .footer { margin-top: 12px; padding: 10px 12px; border-top: 1px solid var(--border); color: var(--muted); }
                    /* Stepper */
                    .stepper { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 8px 0 12px 0; }
                    .step { background: var(--glass); border:1px solid var(--border); border-radius: 8px; padding: 8px; text-align:center; color: var(--muted); }
                    .step.active { color: var(--text); border-color: var(--accent2); }
                    .step.done { color: var(--lime); border-color: var(--lime); }
                    .step.future { color: var(--muted); }
            background: radial-gradient(1200px 600px at 5% 0%, #0a1230 0%, #080d1a 40%, #060a14 100%);
            color: var(--text);
        }
        .navbar {
            position: relative;
            margin: 0.25rem 0 0.75rem 0;
            padding: 10px 14px;
            border-radius: 12px;
            background: var(--panel);
            border: 1px solid var(--border);
            display: flex; align-items: center; gap: 10px;
        }
        .navbar .title { font-weight: 700; letter-spacing: 0.4px; font-size: 1.05rem; }
        .chip {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 10px; border-radius: 999px; border: 1px solid var(--border);
            background: var(--glass); color: var(--blue); font-size: 0.8rem;
        }
        .panel {
            background: var(--panel);
            backdrop-filter: blur(6px);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px 14px;
        }
        .divider { margin: 10px 0; padding-top: 6px; border-top: 1px solid var(--border); font-weight: 600; color: var(--muted); }
        .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }
        .card { background: var(--glass); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
        .card h3 { margin: 0; font-size: 0.9rem; color: var(--muted); }
        .card .value { margin-top: 6px; font-size: 1.3rem; }
        .value.success { color: var(--green); }
        .value.danger { color: var(--red); }
        .value.accent { color: var(--blue); }
        /* Stepper */
        .stepper { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 8px 0 12px 0; }
        .step { background: var(--glass); border:1px solid var(--border); border-radius: 8px; padding: 8px; text-align:center; color: var(--grey); }
        .step.active { color: var(--text); border-color: var(--blue); }
        .step.done { color: var(--green); border-color: var(--green); }
        .step.future { color: var(--grey); }
        /* Progress */
        .progress { width: 100%; height: 10px; border-radius: 999px; background: #0d1630; border: 1px solid var(--border); overflow: hidden; }
        .progress > .bar { height: 100%; background: linear-gradient(90deg, #82b1ff, #6fe3a2); }
        /* Logs */
        .logbox { background: #0e1629; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
        .logline { padding: 2px 0; color: #cfe3ff; font-family: ui-monospace, monospace; }
        .logline.ok { color: var(--green); }
        .logline.fail { color: var(--red); }
        .logline.queue { color: var(--blue); }
        .footer { margin-top: 12px; padding: 10px 12px; border-top: 1px solid var(--border); color: var(--muted); }
        /* Primary buttons */
        .btn-primary { display:inline-block; background: linear-gradient(135deg, #2766ff, #82b1ff); color:#fff; padding:10px 16px; border-radius:10px; border:1px solid #2b5fe0; font-weight:600; }
        .btn-primary.disabled { background: #1a2540; color: var(--grey); border-color: var(--border); cursor:not-allowed; }
        .hint { color: var(--muted); font-size: 0.9rem; }
        .mini { font-size: 0.85rem; color: var(--muted); }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_navbar():
    st.markdown(
        """
        <div class="navbar">
            <div class="title">📄 SEC Filing QA Pipeline</div>
            <div class="chip">⚡ Concurrency</div>
            <div class="chip">🧠 GPT-4o-mini</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_stepper(current_step: int):
    labels = ["Upload Credentials", "Select Google Sheet", "Configure Inputs", "Process Filings"]
    items = []
    for i, label in enumerate(labels):
        cls = "step future"
        if i < current_step:
            cls = "step done"
        elif i == current_step:
            cls = "step active"
        icon = "✅" if i < current_step else ("🔷" if i == current_step else "•")
        items.append(f"<div class='{cls}'><span class='icon'>{icon}</span> {label}</div>")
    st.markdown("<div class='stepper'>" + "".join(items) + "</div>", unsafe_allow_html=True)

def render_status_card(openai_ok: bool, gsheet_ok: bool, tab_count: int, input_tab: str):
    icon_ai = "✅" if openai_ok else "❌"
    icon_gs = "✅" if gsheet_ok else "❌"
    color_ai = "success" if openai_ok else "danger"
    color_gs = "success" if gsheet_ok else "danger"
    st.markdown(
        f"""
        <div class="panel">
            <div class="grid">
                <div class="card"><h3>OpenAI API</h3><div class="value {color_ai}">{icon_ai} {('Connected' if openai_ok else 'Not Connected')}</div></div>
                <div class="card"><h3>Google Sheets</h3><div class="value {color_gs}">{icon_gs} {('Authorized' if gsheet_ok else 'Not Authorized')}</div></div>
                <div class="card"><h3>Sheet Tabs</h3><div class="value accent">{tab_count}</div></div>
                <div class="card"><h3>Input Tab</h3><div class="value">{(input_tab or '—')}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_context_panel(total_filings: int, model: str, concurrency: int, est_runtime_min: int, est_cost: Optional[float]):
    cost_txt = (f"${est_cost:.2f}" if isinstance(est_cost, (float, int)) else "—")
    st.markdown(
        f"""
        <div class="panel">
            <div class="grid">
                <div class="card"><h3>Total Filings</h3><div class="value accent">{total_filings:,}</div></div>
                <div class="card"><h3>Model</h3><div class="value">{model}</div></div>
                <div class="card"><h3>Concurrency</h3><div class="value">{concurrency}</div></div>
                <div class="card"><h3>Est. Runtime</h3><div class="value">{est_runtime_min} min</div></div>
                <div class="card"><h3>Est. Cost</h3><div class="value">{cost_txt}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_stats_showcase(total_files: int, successful: int, failed: int,
                          input_tokens: int, output_tokens: int, cached_tokens: int,
                          total_cost: float):
    st.markdown(
        f"""
        <div class="panel">
            <div class="grid">
                <div class="card"><h3>Total Files</h3><div class="value accent">{total_files:,}</div></div>
                <div class="card"><h3>Successful</h3><div class="value success">{successful:,}</div></div>
                <div class="card"><h3>Failed</h3><div class="value danger">{failed:,}</div></div>
                <div class="card"><h3>Input Tokens</h3><div class="value">{input_tokens:,}</div></div>
                <div class="card"><h3>Output Tokens</h3><div class="value">{output_tokens:,}</div></div>
                <div class="card"><h3>Cached Tokens</h3><div class="value">{cached_tokens:,}</div></div>
                <div class="card"><h3>Estimated Cost</h3><div class="value">${total_cost:.4f}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_footer(model: str, sheet_url: str):
    st.markdown(
        f"""
        <div class="footer">
          <span>Model: {model}</span>
          <span style="margin-left:10px">• Built with Streamlit • <a href="{sheet_url}" target="_blank" style="color: var(--blue);">Open Sheet</a></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_stepper(current_step: int):
        steps = [
            ("1", "Upload Credentials"),
            ("2", "Select Google Sheet"),
            ("3", "Configure Inputs"),
            ("4", "Process Filings"),
        ]
        html = ["<div class='stepper'>"]
        for i, (num, label) in enumerate(steps, start=1):
            cls = "step"
            if current_step > i:
                cls += " done"
            elif current_step == i:
                cls += " active"
            else:
                cls += " future"
            html.append(f"<div class='{cls}'><strong>{num}</strong><br/>{label}</div>")
        html.append("</div>")
        st.markdown("".join(html), unsafe_allow_html=True)

def render_status_summary(openai_ok: bool, sheets_ok: bool, tabs_count: int, input_tab: str):
        icon_ok = "✅"
        icon_err = "⚠️"
        oa = f"<div class='card'><h3>OpenAI</h3><div class='value {'success' if openai_ok else 'danger'}'>{icon_ok if openai_ok else icon_err} {'Connected' if openai_ok else 'Not Connected'}</div></div>"
        gs = f"<div class='card'><h3>Google Sheets</h3><div class='value {'success' if sheets_ok else 'danger'}'>{icon_ok if sheets_ok else icon_err} {'Authorized' if sheets_ok else 'Not Authorized'}</div></div>"
        tb = f"<div class='card'><h3>Sheet Tabs</h3><div class='value accent'>{tabs_count}</div></div>"
        it = f"<div class='card'><h3>Input Tab</h3><div class='value'>{input_tab or '—'}</div></div>"
        st.markdown(f"<div class='panel'><div class='grid'>{oa}{gs}{tb}{it}</div></div>", unsafe_allow_html=True)

 

# ----------------------------
# Remainder of the application (logic and UI)
# ----------------------------
def split_date_time(ts: str) -> Tuple[str, str]:
    """Convert ISO timestamp to (date, time) strings.
    - date: dd-mon-yyyy (month in lowercase, e.g., 13-jan-2026)
    - time: HH:MM (24h)
    """
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        dt = datetime.now()
    date_part = dt.strftime("%d-%b-%Y").lower()
    time_part = dt.strftime("%H:%M")
    return date_part, time_part

def get_openai_credit_grants(api_key: str) -> Optional[Dict[str, float]]:
    """Fetch OpenAI credit grants (remaining balance). Returns dict with totals or None."""
    try:
        url = "https://api.openai.com/v1/dashboard/billing/credit_grants"
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json() or {}
        total_granted = float(data.get("total_granted", 0.0))
        total_used = float(data.get("total_used", 0.0))
        total_remaining = float(data.get("total_remaining", max(0.0, total_granted - total_used)))
        return {
            "total_granted": total_granted,
            "total_used": total_used,
            "total_remaining": total_remaining,
        }
    except Exception as e:
        log(f"Failed to fetch OpenAI credits: {e}")
        return None

def get_openai_usage_tokens(api_key: str, days: int = 30) -> Optional[Dict[str, int]]:
    """Best-effort fetch of token usage via usage endpoint, aggregated over last `days`.
    Returns dict with input_tokens, output_tokens, total_tokens or None.
    """
    try:
        now = int(time.time())
        start = now - max(1, days) * 86400
        url = "https://api.openai.com/v1/usage"
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"start_time": start, "end_time": now}
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json() or {}
        total_input = 0
        total_output = 0
        for item in data.get("data", []):
            # Try common fields; fall back to 0 when missing
            in_tok = item.get("n_input_tokens") or item.get("n_context_tokens_total") or 0
            out_tok = item.get("n_output_tokens") or item.get("n_generated_tokens_total") or 0
            # Some responses may nest by model; defensively sum known keys
            try:
                in_tok = int(in_tok)
            except Exception:
                in_tok = 0
            try:
                out_tok = int(out_tok)
            except Exception:
                out_tok = 0
            total_input += in_tok
            total_output += out_tok
        return {"input_tokens": total_input, "output_tokens": total_output, "total_tokens": total_input + total_output}
    except Exception as e:
        log(f"Failed to fetch OpenAI usage tokens: {e}")
        return None


def get_service_account_email(path: str) -> str:
    """Return the service account email from the JSON file, if available."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("client_email", "")
    except Exception:
        return ""

def assert_sheet_writeable(worksheet) -> bool:
    """Probe the sheet for write permission by updating a harmless cell."""
    try:
        worksheet.update('Z1', [["ping"]])
        log("Write probe succeeded (Z1=ping).")
        return True
    except Exception as e:
        log(f"Write probe failed: {e}")
        st.error(f"Cannot write to the sheet: {e}. Ensure the sheet is shared (Editor) with your service account email.")
        return False

# ----------------------------
# Helper Functions
# ----------------------------
def initialize_openai(api_key: str):
    """Initialize one or more OpenAI clients with provided keys (OpenAI only)."""
    global client, CLIENTS
    CLIENTS = []

    # Collect keys: primary uploaded key + optional extra keys from multiple .txt uploads + env default
    extra_keys = st.session_state.get("extra_api_keys", []) or []

    keys: List[str] = []
    if api_key and api_key.strip():
        keys.append(api_key.strip())
    # Add additional keys (independent accounts/projects recommended)
    keys.extend([k for k in extra_keys if k not in keys])
    if not keys and LLM_API_KEY:
        keys = [LLM_API_KEY]

    if not keys:
        raise RuntimeError("No OpenAI API keys provided")

    # Build clients list (OpenAI only)
    for k in keys:
        CLIENTS.append(OpenAI(api_key=k))

    # Default client (used when no thread-local client is set)
    client = CLIENTS[0]

    # Initialize tokenizer
    global tokenizer
    try:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    except:
        tokenizer = tiktoken.get_encoding("cl100k_base")

def normalize_text_for_prefix(txt: str) -> str:
    """Canonical normalization to ensure identical prefix across calls."""
    if not txt:
        return ""
    # html is imported in __main__ before main() executes
    import html  # local import to ensure availability
    txt = html.unescape(txt)
    for ch in ("\u00a0", "\u2007", "\u2009", "\u202f", "\u2002", "\u2003", "\u200a"):
        txt = txt.replace(ch, " ")
    txt = txt.replace("\r", " ").replace("\t", " ")
    txt = re.sub(r"\n\s*\n+", "\n\n", txt)          # collapse multiple blank lines
    lines = [ln.strip() for ln in txt.splitlines()]
    txt = "\n".join(lines).strip()
    txt = re.sub(r"[ ]{2,}", " ", txt)
    return txt

def fetch_filing_html(url: str, timeout: int = 30, retry_forever: bool = False, status_text=None) -> Optional[str]:
    # Removed erroneous duplicate block; see actual implementation later in file
    pass

def get_google_sheet_client(service_account_file: str):
    """Authenticate and return Google Sheets client."""
    try:
        creds = Credentials.from_service_account_file(
            service_account_file,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        gc = gspread.authorize(creds)
        return gc
    except Exception as e:
        st.error(f"Failed to authenticate with Google Sheets: {e}")
        return None

def get_sheet_tabs(gc, spreadsheet_url: str) -> List[str]:
    """Get all sheet tabs from the Google Sheet."""
    try:
        spreadsheet = gc.open_by_url(spreadsheet_url)
        worksheets = spreadsheet.worksheets()
        tab_names = [ws.title for ws in worksheets]
        return tab_names
    except Exception as e:
        st.error(f"Failed to fetch sheet tabs: {e}")
        return []

def get_inputs_from_sheet(gc, spreadsheet_url: str, input_tab_name: str) -> Tuple[List[str], List[str]]:
    """Read URLs and questions from specified sheet tab."""
    try:
        spreadsheet = gc.open_by_url(spreadsheet_url)
        worksheet = spreadsheet.worksheet(input_tab_name)
        all_values = worksheet.get_all_values()
        
        if not all_values or len(all_values) < 2:
            st.error(f"No data found in tab '{input_tab_name}'")
            return [], []
        
        # First row contains questions
        question_row = all_values[0]
        questions = [q.strip() for q in question_row[1:] if q.strip()]
        
        if not questions:
            st.error(f"No questions found in first row of tab '{input_tab_name}'")
            return [], []
        
        # Extract URLs from column A
        urls = []
        for i, row in enumerate(all_values[1:], start=2):
            if row and row[0].strip():
                url = row[0].strip()
                if url.startswith(('http://', 'https://', 'www.')):
                    urls.append(url)
        
        if not urls:
            st.error(f"No valid URLs found in column A of tab '{input_tab_name}'")
            return [], []
        
        return urls, questions
        
    except Exception as e:
        st.warning(f"Failed to read from Google Sheet: {e}")
        return [], []

def update_questions_in_sheet(gc, spreadsheet_url: str, input_tab_name: str, questions: List[str]) -> bool:
    """Update questions in the Google Sheet."""
    try:
        spreadsheet = gc.open_by_url(spreadsheet_url)
        worksheet = spreadsheet.worksheet(input_tab_name)
        
        # Get current data to preserve URLs
        all_values = worksheet.get_all_values()
        
        if not all_values:
            return False
        
        # Create new first row: empty first cell, then questions
        new_first_row = [''] + questions
        
        # Update first row
        worksheet.update('A1', [new_first_row])
        
        return True
    except Exception as e:
        st.error(f"Failed to update questions in sheet: {e}")
        return False

def add_url_to_sheet(gc, spreadsheet_url: str, input_tab_name: str, url: str) -> bool:
    """Add a new URL to the Google Sheet."""
    try:
        spreadsheet = gc.open_by_url(spreadsheet_url)
        worksheet = spreadsheet.worksheet(input_tab_name)
        
        # Find the next empty row in column A
        all_values = worksheet.get_all_values()
        
        # Count existing rows with URLs
        existing_urls = []
        for row in all_values[1:]:  # Skip header row
            if row and row[0].strip():
                existing_urls.append(row[0].strip())
        
        # Add new URL in the next row
        next_row = len(existing_urls) + 2  # +1 for header, +1 for next empty row
        worksheet.update(f'A{next_row}', [[url]])
        
        return True
    except Exception as e:
        st.error(f"Failed to add URL to sheet: {e}")
        return False

def setup_output_tab(gc, spreadsheet_url: str, output_tab_name: str, num_questions: int):
    """Setup output tab with headers."""
    try:
        spreadsheet = gc.open_by_url(spreadsheet_url)
        
        try:
            worksheet = spreadsheet.worksheet(output_tab_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=output_tab_name, rows=1000, cols=50)
        
        headers = ["date", "time", "company_name", "ticker", "url", 
                  "input_tokens", "cached_tokens", "output_tokens"]
        
        for i in range(1, num_questions + 1):
            headers.append(f"answer_{i}")
        # Put headers only if there is no second row (i.e., sheet effectively empty)
        try:
            existing = worksheet.get_all_values()
        except Exception:
            existing = []
        if len(existing) < 2:
            worksheet.update('A1', [headers], value_input_option='USER_ENTERED')
        return worksheet
        
    except Exception as e:
        st.error(f"Failed to setup output tab: {e}")
        return None

def get_input_url_rows(gc, spreadsheet_url: str, input_tab_name: str) -> List[Tuple[str, int]]:
    """Return list of (url, row_index) from the input tab, starting at row 2."""
    try:
        spreadsheet = gc.open_by_url(spreadsheet_url)
        worksheet = spreadsheet.worksheet(input_tab_name)
        all_values = worksheet.get_all_values()
        out: List[Tuple[str, int]] = []
        for i, row in enumerate(all_values[1:], start=2):
            if row and row[0].strip():
                url = row[0].strip()
                if url.startswith(("http://", "https://", "www.")):
                    out.append((url, i))
        return out
    except Exception:
        return []

def setup_revised_output_tab(gc, spreadsheet_url: str, urls: List[str], tab_name: str = "revised input"):
    """Create or clear 'revised input' tab and write the de-duplicated URLs in column A (header 'url')."""
    try:
        spreadsheet = gc.open_by_url(spreadsheet_url)
        try:
            worksheet = spreadsheet.worksheet(tab_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=tab_name, rows=1000, cols=10)
        # Clear existing content (keep sheet)
        try:
            worksheet.clear()
        except Exception:
            pass
        # Write header and URLs
        header = ["url"]
        worksheet.update('A1', [header])
        if urls:
            rows = [[u] for u in urls]
            # Write starting at A2
            worksheet.update(f'A2', rows, value_input_option='USER_ENTERED')
        log(f"revised input tab '{tab_name}' populated with {len(urls)} unique URLs.")
        return worksheet
    except Exception as e:
        st.warning(f"Failed to setup revised input tab: {e}")
        return None

def write_result_to_sheet_row(worksheet, row_index: int, result: Dict) -> bool:
    """Write a single result to a specific row in the output worksheet instead of appending."""
    if not isinstance(row_index, int) or row_index < 2:
        # Fallback to append when row index invalid
        return write_single_result_to_sheet(worksheet, result)
    try:
        ts = result.get('timestamp') or datetime.now().isoformat()
        date_part, time_part = split_date_time(ts)
        row = [
            date_part,
            time_part,
            result['company_name'],
            result['ticker'],
            result['url'],
            str(result['input_tokens']),
            str(result['cached_tokens']),
            str(result['output_tokens'])
        ]
        for answer in result['answers']:
            if isinstance(answer, dict):
                clean_answer = json.dumps(answer, ensure_ascii=False)
            else:
                clean_answer = str(answer)
            clean_answer = clean_answer.replace('\n', ' ').replace('\r', ' ').strip()
            if len(clean_answer) > 990:
                clean_answer = clean_answer[:990] + "..."
            row.append(clean_answer)
        worksheet.update(f'A{row_index}', [row], value_input_option='USER_ENTERED')
        log(f"Row {row_index} updated for URL: {result['url']}")
        return True
    except Exception as e:
        st.error(f"Failed to write row {row_index} to sheet: {e}")
        log(f"Failed to write row {row_index} for URL {result.get('url','?')}: {e}")
        return False

def write_single_result_to_sheet(worksheet, result: Dict):
    """Write a single URL's result row to the sheet."""
    try:
        ts = result.get('timestamp') or datetime.now().isoformat()
        date_part, time_part = split_date_time(ts)
        row = [
            date_part,
            time_part,
            result['company_name'],
            result['ticker'],
            result['url'],
            str(result['input_tokens']),
            str(result['cached_tokens']),
            str(result['output_tokens'])
        ]
        
        for answer in result['answers']:
            if isinstance(answer, dict):
                clean_answer = json.dumps(answer, ensure_ascii=False)
            else:
                clean_answer = str(answer)
            
            clean_answer = clean_answer.replace('\n', ' ').replace('\r', ' ').strip()
            if len(clean_answer) > 990:
                clean_answer = clean_answer[:990] + "..."
            row.append(clean_answer)
        
        worksheet.append_row(row)
        log(f"Row appended for URL: {result['url']}")
        return True
        
    except Exception as e:
        st.error(f"Failed to write row to sheet: {e}")
        log(f"Failed to write row for URL {result.get('url','?')}: {e}")
        return False

def build_failure_result(url: str, num_questions: int) -> Dict:
    """Construct a placeholder result row for failed processing to preserve ordering."""
    return {
        'timestamp': datetime.now().isoformat(),
        'company_name': 'Unknown Company',
        'ticker': 'UNKN',
        'url': url,
        'input_tokens': 0,
        'cached_tokens': 0,
        'output_tokens': 0,
        'answers': ['none'] * num_questions
    }

def write_results_in_order(worksheet, url_order: List[str], results_map: Dict[str, Optional[Dict]], num_questions: int):
    """Write all results in the exact input order by appending at the end (no overwrite)."""
    # Build rows
    rows = []
    for url in url_order:
        res = results_map.get(url)
        if not res:
            res = build_failure_result(url, num_questions)
        ts = res.get('timestamp') or datetime.now().isoformat()
        date_part, time_part = split_date_time(ts)
        row = [
            date_part,
            time_part,
            res['company_name'],
            res['ticker'],
            res['url'],
            str(res['input_tokens']),
            str(res['cached_tokens']),
            str(res['output_tokens'])
        ]
        for answer in res['answers']:
            if isinstance(answer, dict):
                clean_answer = json.dumps(answer, ensure_ascii=False)
            else:
                clean_answer = str(answer)
            clean_answer = clean_answer.replace('\n', ' ').replace('\r', ' ').strip()
            if len(clean_answer) > 990:
                clean_answer = clean_answer[:990] + "..."
            row.append(clean_answer)
        rows.append(row)

    # Append rows in chunks to the bottom
    try:
        if hasattr(worksheet, 'append_rows'):
            # Use moderate chunk sizes to respect API limits
            chunk_size = 50
            for i in range(0, len(rows), chunk_size):
                chunk = rows[i:i+chunk_size]
                worksheet.append_rows(chunk, value_input_option='USER_ENTERED')
                log(f"Batch appended {len(chunk)} rows in order.")
            return True
    except Exception as e:
        log(f"Batch append failed: {e}. Falling back to per-row writes.")

    # Fallback to per-row append
    ok = True
    for url, row in zip(url_order, rows):
        try:
            worksheet.append_row(row, value_input_option='USER_ENTERED')
            log(f"Row appended for URL: {url}")
        except Exception as e:
            ok = False
            st.error(f"Failed to write row to sheet for {url}: {e}")
            log(f"Failed to write row for URL {url}: {e}")
            # Be polite to Sheets API
            time.sleep(0.5)
    return ok

def process_single_filing(url: str, questions: List[str], progress_bar=None, status_text=None) -> Optional[Dict]:
    """Process a single filing URL with all questions."""
    if status_text:
        status_text.text(f"Fetching: {url[:50]}...")
    
    html = fetch_filing_html(url, retry_forever=True, status_text=status_text)
    if not html:
        return None
    
    extracted = extract_plaintext(html)
    _ = len(extracted)  # unused metrics retained for potential UI
    _ = count_tokens(extracted)
    
    if status_text:
        status_text.text(f"Extracting company info...")
    
    company_name, ticker, _ = extract_company_info_with_gpt(extracted)
    
    processed_text = extracted
    
    # SEED cache (only if GPT available)
    seed_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
    if not GPT_DISABLED.is_set():
        if status_text:
            status_text.text(f"Seeding cache...")
        seed_usage = seed_prefix_cache(processed_text)
        time.sleep(SLEEP_BETWEEN_CALLS)
    
    # Process all questions with persistent retry until completion
    total_prompt = 0
    total_completion = 0
    total_cached = 0
    all_answers = [None] * len(questions)

    def _ask(idx_q: int, q_text: str) -> Tuple[int, Tuple[str, Dict]]:
        ans, use = ask_question_with_prefix(processed_text, q_text)
        return idx_q, (ans, use)

    per_question_workers = max(1, (1 if GPT_RATE_LIMITED.is_set() else min(len(questions), MAX_GPT_CONCURRENCY)))
    pending_indices = list(range(len(questions)))
    completed = 0

    while pending_indices:
        # Launch up to per_question_workers tasks
        batch = pending_indices[:per_question_workers]
        pending_indices = pending_indices[per_question_workers:]
        futures_map = {}
        with ThreadPoolExecutor(max_workers=len(batch) or 1) as executor:
            for idx in batch:
                futures_map[executor.submit(_ask, idx, questions[idx])] = idx
            for fut in as_completed(futures_map):
                idx_q = futures_map[fut]
                try:
                    _, (answer, usage) = fut.result()
                    all_answers[idx_q] = answer
                    total_prompt += usage.get("prompt_tokens", 0)
                    total_completion += usage.get("completion_tokens", 0)
                    total_cached += usage.get("cached_tokens", 0)
                    completed += 1
                    if progress_bar:
                        progress_bar.progress(completed / len(questions))
                except Exception:
                    # If globally disabled, mark as none and do not retry
                    if GPT_DISABLED.is_set():
                        all_answers[idx_q] = "none"
                        completed += 1
                        if progress_bar:
                            progress_bar.progress(completed / len(questions))
                    else:
                        # Rate limited or transient error: re-queue this question
                        pending_indices.append(idx_q)
                        # Gentle wait to respect rate limits when re-queuing
                        wait_time = SLEEP_BETWEEN_CALLS
                        if GPT_RATE_LIMITED.is_set():
                            wait_time = max(wait_time, 4.0)
                        time.sleep(wait_time)
    
    result = {
        'timestamp': datetime.now().isoformat(),
        'company_name': company_name,
        'ticker': ticker,
        'url': url,
        'input_tokens': max(0, total_prompt - total_cached),
        'cached_tokens': total_cached,
        'output_tokens': total_completion,
        'answers': all_answers
    }
    
    return result

# ----------------------------
# Streamlit UI
# ----------------------------

def split_date_time(ts: str) -> Tuple[str, str]:
    """Convert ISO timestamp to (date, time) strings.
    - date: dd-mon-yyyy (month in lowercase, e.g., 13-jan-2026)
    - time: HH:MM (24h)
    """
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        dt = datetime.now()
    date_part = dt.strftime("%d-%b-%Y").lower()
    time_part = dt.strftime("%H:%M")
    return date_part, time_part

def get_openai_credit_grants(api_key: str) -> Optional[Dict[str, float]]:
    """Fetch OpenAI credit grants (remaining balance). Returns dict with totals or None."""
    try:
        url = "https://api.openai.com/v1/dashboard/billing/credit_grants"
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json() or {}
        total_granted = float(data.get("total_granted", 0.0))
        total_used = float(data.get("total_used", 0.0))
        total_remaining = float(data.get("total_remaining", max(0.0, total_granted - total_used)))
        return {
            "total_granted": total_granted,
            "total_used": total_used,
            "total_remaining": total_remaining,
        }
    except Exception as e:
        log(f"Failed to fetch OpenAI credits: {e}")
        return None

def get_openai_usage_tokens(api_key: str, days: int = 30) -> Optional[Dict[str, int]]:
    """Best-effort fetch of token usage via usage endpoint, aggregated over last `days`.
    Returns dict with input_tokens, output_tokens, total_tokens or None.
    """
    try:
        now = int(time.time())
        start = now - max(1, days) * 86400
        url = "https://api.openai.com/v1/usage"
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"start_time": start, "end_time": now}
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json() or {}
        total_input = 0
        total_output = 0
        for item in data.get("data", []):
            # Try common fields; fall back to 0 when missing
            in_tok = item.get("n_input_tokens") or item.get("n_context_tokens_total") or 0
            out_tok = item.get("n_output_tokens") or item.get("n_generated_tokens_total") or 0
            # Some responses may nest by model; defensively sum known keys
            try:
                in_tok = int(in_tok)
            except Exception:
                in_tok = 0
            try:
                out_tok = int(out_tok)
            except Exception:
                out_tok = 0
            total_input += in_tok
            total_output += out_tok
        return {"input_tokens": total_input, "output_tokens": total_output, "total_tokens": total_input + total_output}
    except Exception as e:
        log(f"Failed to fetch OpenAI usage tokens: {e}")
        return None

def get_service_account_email(path: str) -> str:
    """Return the service account email from the JSON file, if available."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("client_email", "")
    except Exception:
        return ""

def assert_sheet_writeable(worksheet) -> bool:
    """Probe the sheet for write permission by updating a harmless cell."""
    try:
        worksheet.update('Z1', [["ping"]])
        log("Write probe succeeded (Z1=ping).")
        return True
    except Exception as e:
        log(f"Write probe failed: {e}")
        st.error(f"Cannot write to the sheet: {e}. Ensure the sheet is shared (Editor) with your service account email.")
        return False

# ----------------------------
# Helper Functions
# ----------------------------
def initialize_openai(api_key: str):
    """Initialize one or more OpenAI clients with provided keys (OpenAI only)."""
    global client, CLIENTS
    CLIENTS = []

    # Collect keys: primary uploaded key + optional extra keys from multiple .txt uploads + env default
    extra_keys = st.session_state.get("extra_api_keys", []) or []

    keys: List[str] = []
    if api_key and api_key.strip():
        keys.append(api_key.strip())
    # Add additional keys (independent accounts/projects recommended)
    keys.extend([k for k in extra_keys if k not in keys])
    if not keys and LLM_API_KEY:
        keys = [LLM_API_KEY]

    if not keys:
        raise RuntimeError("No OpenAI API keys provided")

    # Build clients list (OpenAI only)
    for k in keys:
        CLIENTS.append(OpenAI(api_key=k))

    # Default client (used when no thread-local client is set)
    client = CLIENTS[0]

    # Initialize tokenizer
    global tokenizer
    try:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    except:
        tokenizer = tiktoken.get_encoding("cl100k_base")

def normalize_text_for_prefix(txt: str) -> str:
    """Canonical normalization to ensure identical prefix across calls."""
    if not txt:
        return ""
    txt = html.unescape(txt)
    for ch in ("\u00a0", "\u2007", "\u2009", "\u202f", "\u2002", "\u2003", "\u200a"):
        txt = txt.replace(ch, " ")
    txt = txt.replace("\r", " ").replace("\t", " ")
    txt = re.sub(r"\n\s*\n+", "\n\n", txt)          # collapse multiple blank lines
    lines = [ln.strip() for ln in txt.splitlines()]
    txt = "\n".join(lines).strip()
    txt = re.sub(r"[ ]{2,}", " ", txt)
    return txt

def fetch_filing_html(url: str, timeout: int = 30, retry_forever: bool = False, status_text=None) -> Optional[str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive"
    }

    if "ix?doc=" in url:
        m = re.search(r'ix\?doc=(/.+)', url)
        if m:
            url = f"https://www.sec.gov{m.group(1)}"
    attempts = 0
    while True:
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            attempts += 1
            log(f"Fetch failed for {url}: {e} (attempt {attempts})")
            ATTEMPT_COUNTS[url] = attempts
            if not retry_forever:
                st.error(f"Failed to fetch {url}: {e}")
                return None
            # Exponential backoff with cap
            wait = min(60.0, 2 ** min(10, attempts - 1))
            time.sleep(wait)


def extract_plaintext(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    raw = soup.get_text(separator="\n", strip=True)
    return normalize_text_for_prefix(raw)

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    if tokenizer is None:
        # Rough estimate: 4 chars per token
        return len(text) // 4
    return len(tokenizer.encode(text))

def trim_text_to_token_limit(text: str, max_tokens: int, reserved_tokens: int = 10000) -> str:
    """Trim text to fit within token limit."""
    available_tokens = max_tokens - reserved_tokens
    token_count = count_tokens(text)
    
    if token_count <= available_tokens:
        return text
    
    # If text is too long, we need to truncate it
    if tokenizer is not None:
        tokens = tokenizer.encode(text)
        truncated_tokens = tokens[:available_tokens]
        truncated_text = tokenizer.decode(truncated_tokens)
    else:
        # Fallback: truncate by characters (rough estimate)
        chars_per_token = 4
        max_chars = available_tokens * chars_per_token
        truncated_text = text[:max_chars]
    
    return truncated_text

def api_call_with_retry(messages: List[dict], max_tokens: int, temperature: float = 0.0) -> object:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        # Stop immediately if GPT is globally disabled due to quota
        if GPT_DISABLED.is_set():
            raise RuntimeError("OpenAI disabled due to insufficient quota")
        try:
            # Limit concurrent GPT API calls
            gpt_semaphore.acquire()
            try:
                # Use thread-local client if assigned; else fallback to default
                active_client = getattr(thread_local, "client", None) or client
                resp = active_client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            finally:
                gpt_semaphore.release()
            # Clear rate-limit flag on success
            if GPT_RATE_LIMITED.is_set():
                GPT_RATE_LIMITED.clear()
            return resp
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            # Quota exhausted → stop all GPT calls
            if ("insufficient_quota" in msg) or ("quota" in msg and "exceeded" in msg):
                GPT_DISABLED.set()
                break
            # General rate-limit → slow down and reduce per-question workers
            wait = RETRY_BACKOFF_BASE ** (attempt - 1)
            # Parse suggested retry delay if present (e.g., "Please try again in 4.032s")
            m = re.search(r"try again in\s+([0-9.]+)s", msg)
            if m:
                try:
                    suggested = float(m.group(1))
                    wait = max(wait, suggested)
                except Exception:
                    pass
            GPT_RATE_LIMITED.set()
            time.sleep(wait)
    raise last_exc

def build_prefix(full_text: str) -> Tuple[dict, dict]:
    """Return canonical prefix messages (user, system)."""
    user_msg = {"role": "user", "content": full_text}
    sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
    return user_msg, sys_msg

def seed_prefix_cache(prefix_text: str) -> Dict:
    """Seed the prompt cache using the canonical prefix."""
    if GPT_DISABLED.is_set():
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
    user_msg, sys_msg = build_prefix(prefix_text)
    messages = [user_msg, sys_msg]
    resp = api_call_with_retry(messages=messages, max_tokens=1, temperature=0.0)
    usage = {}
    try:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens
        }
        if hasattr(resp.usage, "prompt_tokens_details"):
            details = resp.usage.prompt_tokens_details
            if hasattr(details, "cached_tokens"):
                usage["cached_tokens"] = details.cached_tokens
    except Exception:
        pass
    return usage

def ask_question_with_prefix(prefix_text: str, question: str, max_tokens: int = DEFAULT_MAX_TOKENS_PER_QUESTION) -> Tuple[str, Dict]:
    """Ask a question reusing the exact prefix."""
    if GPT_DISABLED.is_set():
        return "none", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
    user_msg, sys_msg = build_prefix(prefix_text)
    messages = [user_msg, sys_msg, {"role": "user", "content": question}]
    resp = api_call_with_retry(messages=messages, max_tokens=max_tokens, temperature=0.0)
    answer = ""
    usage = {}
    try:
        answer = resp.choices[0].message.content.strip()
    except Exception:
        pass
    try:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens
        }
        if hasattr(resp.usage, "prompt_tokens_details"):
            details = resp.usage.prompt_tokens_details
            if hasattr(details, "cached_tokens"):
                usage["cached_tokens"] = details.cached_tokens
    except Exception:
        pass
    
    # Normalize answer
    answer = answer.strip()
    if answer.lower() in ['', 'not found', 'information not available', 'n/a', 'na']:
        answer = 'none'
    elif 'not present in the text' in answer.lower() or 'not found in the text' in answer.lower():
        answer = 'none'
    elif 'i don\'t see' in answer.lower() or 'i cannot find' in answer.lower():
        answer = 'none'
    elif 'none' in answer.lower() and len(answer) < 10:
        answer = 'none'
    
    return answer, usage

def extract_company_info_with_gpt(full_text: str) -> Tuple[str, str, Dict]:
    """Extract company name and ticker using GPT."""
    if GPT_DISABLED.is_set():
        return "Unknown Company", "UNKN", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
    COMPANY_EXTRACTION_PROMPT = """Extract the company name and stock ticker symbol from this SEC filing text.
    Return ONLY a JSON object with two fields:
    - "company_name": Full legal name of the company
    - "ticker": Stock ticker symbol in uppercase
    
    If you cannot determine either field, use "Unknown" as the value."""
    
    sample_text = full_text[:10000]
    messages = [
        {"role": "user", "content": sample_text},
        {"role": "system", "content": COMPANY_EXTRACTION_PROMPT}
    ]
    
    try:
        resp = api_call_with_retry(messages=messages, max_tokens=100, temperature=0.0)
        answer = resp.choices[0].message.content.strip()
        
        json_match = re.search(r'\{.*\}', answer, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                company_name = data.get('company_name', 'Unknown Company')
                ticker = data.get('ticker', 'UNKN')
                
                if ticker != 'UNKN' and isinstance(ticker, str):
                    ticker = ticker.upper().strip()
                    ticker = re.sub(r'[^A-Z]', '', ticker)
                    if not ticker or len(ticker) > 5:
                        ticker = 'UNKN'
                
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                    "cached_tokens": 0
                }
                
                return company_name, ticker, usage
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    
    return "Unknown Company", "UNKN", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}

def get_google_sheet_client(service_account_file: str):
    """Authenticate and return Google Sheets client."""
    try:
        creds = Credentials.from_service_account_file(
            service_account_file,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        gc = gspread.authorize(creds)
        return gc
    except Exception as e:
        st.error(f"Failed to authenticate with Google Sheets: {e}")
        return None

def get_sheet_tabs(gc, spreadsheet_url: str) -> List[str]:
    """Get all sheet tabs from the Google Sheet."""
    try:
        spreadsheet = gc.open_by_url(spreadsheet_url)
        worksheets = spreadsheet.worksheets()
        tab_names = [ws.title for ws in worksheets]
        return tab_names
    except Exception as e:
        st.error(f"Failed to fetch sheet tabs: {e}")
        return []

def get_inputs_from_sheet(gc, spreadsheet_url: str, input_tab_name: str) -> Tuple[List[str], List[str]]:
    """Read URLs and questions from specified sheet tab."""
    try:
        spreadsheet = gc.open_by_url(spreadsheet_url)
        worksheet = spreadsheet.worksheet(input_tab_name)
        all_values = worksheet.get_all_values()
        
        if not all_values or len(all_values) < 2:
            st.error(f"No data found in tab '{input_tab_name}'")
            return [], []
        
        # First row contains questions
        question_row = all_values[0]
        questions = [q.strip() for q in question_row[1:] if q.strip()]
        
        if not questions:
            st.error(f"No questions found in first row of tab '{input_tab_name}'")
            return [], []
        
        # Extract URLs from column A
        urls = []
        for i, row in enumerate(all_values[1:], start=2):
            if row and row[0].strip():
                url = row[0].strip()
                if url.startswith(('http://', 'https://', 'www.')):
                    urls.append(url)
        
        if not urls:
            st.error(f"No valid URLs found in column A of tab '{input_tab_name}'")
            return [], []
        
        return urls, questions
        
    except Exception as e:
        st.error(f"Failed to read from Google Sheet: {e}")
        return [], []

def update_questions_in_sheet(gc, spreadsheet_url: str, input_tab_name: str, questions: List[str]) -> bool:
    """Update questions in the Google Sheet."""
    try:
        spreadsheet = gc.open_by_url(spreadsheet_url)
        worksheet = spreadsheet.worksheet(input_tab_name)
        
        # Get current data to preserve URLs
        all_values = worksheet.get_all_values()
        
        if not all_values:
            return False
        
        # Create new first row: empty first cell, then questions
        new_first_row = [''] + questions
        
        # Update first row
        worksheet.update('A1', [new_first_row])
        
        return True
    except Exception as e:
        st.error(f"Failed to update questions in sheet: {e}")
        return False

def add_url_to_sheet(gc, spreadsheet_url: str, input_tab_name: str, url: str) -> bool:
    """Add a new URL to the Google Sheet."""
    try:
        spreadsheet = gc.open_by_url(spreadsheet_url)
        worksheet = spreadsheet.worksheet(input_tab_name)
        
        # Find the next empty row in column A
        all_values = worksheet.get_all_values()
        
        # Count existing rows with URLs
        existing_urls = []
        for row in all_values[1:]:  # Skip header row
            if row and row[0].strip():
                existing_urls.append(row[0].strip())
        
        # Add new URL in the next row
        next_row = len(existing_urls) + 2  # +1 for header, +1 for next empty row
        worksheet.update(f'A{next_row}', [[url]])
        
        return True
    except Exception as e:
        st.error(f"Failed to add URL to sheet: {e}")
        return False

def setup_output_tab(gc, spreadsheet_url: str, output_tab_name: str, num_questions: int):
    """Setup output tab with headers."""
    try:
        spreadsheet = gc.open_by_url(spreadsheet_url)
        
        try:
            worksheet = spreadsheet.worksheet(output_tab_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=output_tab_name, rows=1000, cols=50)
        
        headers = ["date", "time", "company_name", "ticker", "url", 
                  "input_tokens", "cached_tokens", "output_tokens"]
        
        for i in range(1, num_questions + 1):
            headers.append(f"answer_{i}")
        # Put headers only if there is no second row (i.e., sheet effectively empty)
        try:
            existing = worksheet.get_all_values()
        except Exception:
            existing = []
        if len(existing) < 2:
            worksheet.update('A1', [headers], value_input_option='USER_ENTERED')
        return worksheet
        
    except Exception as e:
        st.error(f"Failed to setup output tab: {e}")
        return None

def write_single_result_to_sheet(worksheet, result: Dict):
    """Write a single URL's result row to the sheet."""
    try:
        ts = result.get('timestamp') or datetime.now().isoformat()
        date_part, time_part = split_date_time(ts)
        row = [
            date_part,
            time_part,
            result['company_name'],
            result['ticker'],
            result['url'],
            str(result['input_tokens']),
            str(result['cached_tokens']),
            str(result['output_tokens'])
        ]
        
        for answer in result['answers']:
            if isinstance(answer, dict):
                clean_answer = json.dumps(answer, ensure_ascii=False)
            else:
                clean_answer = str(answer)
            
            clean_answer = clean_answer.replace('\n', ' ').replace('\r', ' ').strip()
            if len(clean_answer) > 990:
                clean_answer = clean_answer[:990] + "..."
            row.append(clean_answer)
        
        worksheet.append_row(row)
        log(f"Row appended for URL: {result['url']}")
        return True
        
    except Exception as e:
        st.error(f"Failed to write row to sheet: {e}")
        log(f"Failed to write row for URL {result.get('url','?')}: {e}")
        return False

def build_failure_result(url: str, num_questions: int) -> Dict:
    """Construct a placeholder result row for failed processing to preserve ordering."""
    return {
        'timestamp': datetime.now().isoformat(),
        'company_name': 'Unknown Company',
        'ticker': 'UNKN',
        'url': url,
        'input_tokens': 0,
        'cached_tokens': 0,
        'output_tokens': 0,
        'answers': ['none'] * num_questions
    }

def write_results_in_order(worksheet, url_order: List[str], results_map: Dict[str, Optional[Dict]], num_questions: int):
    """Write all results in the exact input order by appending at the end (no overwrite)."""
    # Build rows
    rows = []
    for url in url_order:
        res = results_map.get(url)
        if not res:
            res = build_failure_result(url, num_questions)
        ts = res.get('timestamp') or datetime.now().isoformat()
        date_part, time_part = split_date_time(ts)
        row = [
            date_part,
            time_part,
            res['company_name'],
            res['ticker'],
            res['url'],
            str(res['input_tokens']),
            str(res['cached_tokens']),
            str(res['output_tokens'])
        ]
        for answer in res['answers']:
            if isinstance(answer, dict):
                clean_answer = json.dumps(answer, ensure_ascii=False)
            else:
                clean_answer = str(answer)
            clean_answer = clean_answer.replace('\n', ' ').replace('\r', ' ').strip()
            if len(clean_answer) > 990:
                clean_answer = clean_answer[:990] + "..."
            row.append(clean_answer)
        rows.append(row)

    # Append rows in chunks to the bottom
    try:
        if hasattr(worksheet, 'append_rows'):
            # Use moderate chunk sizes to respect API limits
            chunk_size = 50
            for i in range(0, len(rows), chunk_size):
                chunk = rows[i:i+chunk_size]
                worksheet.append_rows(chunk, value_input_option='USER_ENTERED')
                log(f"Batch appended {len(chunk)} rows in order.")
            return True
    except Exception as e:
        log(f"Batch append failed: {e}. Falling back to per-row writes.")

    # Fallback to per-row append
    ok = True
    for url, row in zip(url_order, rows):
        try:
            worksheet.append_row(row, value_input_option='USER_ENTERED')
            log(f"Row appended for URL: {url}")
        except Exception as e:
            ok = False
            st.error(f"Failed to write row to sheet for {url}: {e}")
            log(f"Failed to write row for URL {url}: {e}")
            # Be polite to Sheets API
            time.sleep(0.5)
    return ok

def process_single_filing(url: str, questions: List[str], progress_bar=None, status_text=None) -> Optional[Dict]:
    """Process a single filing URL with all questions."""
    if status_text:
        status_text.text(f"Fetching: {url[:50]}...")
    
    html = fetch_filing_html(url, retry_forever=True, status_text=status_text)
    if not html:
        return None
    
    extracted = extract_plaintext(html)
    text_length_chars = len(extracted)
    text_length_tokens = count_tokens(extracted)
    
    if status_text:
        status_text.text(f"Extracting company info...")
    
    company_name, ticker, _ = extract_company_info_with_gpt(extracted)
    
    # Prepare text for processing and fit within model context window
    # Approximate question token budget and reserved tokens for answers + system overhead
    max_question_tokens = max(len(q) // 4 for q in questions) + 100
    max_answer_tokens = DEFAULT_MAX_TOKENS_PER_QUESTION
    reserved_tokens = 5000 + (len(questions) * (max_question_tokens + max_answer_tokens))

    processed_text = trim_text_to_token_limit(extracted, max_tokens=MAX_CONTEXT_TOKENS, reserved_tokens=reserved_tokens)
    trimmed_tokens = count_tokens(processed_text)
    if trimmed_tokens < text_length_tokens:
        log(f"Trimmed filing text from {text_length_tokens} → {trimmed_tokens} tokens to fit context.")
    
    # SEED cache (only if GPT available)
    seed_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
    if not GPT_DISABLED.is_set():
        if status_text:
            status_text.text(f"Seeding cache...")
        seed_usage = seed_prefix_cache(processed_text)
        time.sleep(SLEEP_BETWEEN_CALLS)
    
    # Process all questions with persistent retry until completion
    total_prompt = 0
    total_completion = 0
    total_cached = 0
    all_answers = [None] * len(questions)

    def _ask(idx_q: int, q_text: str) -> Tuple[int, Tuple[str, Dict]]:
        ans, use = ask_question_with_prefix(processed_text, q_text)
        return idx_q, (ans, use)

    per_question_workers = max(1, (1 if GPT_RATE_LIMITED.is_set() else min(len(questions), MAX_GPT_CONCURRENCY)))
    pending_indices = list(range(len(questions)))
    completed = 0

    while pending_indices:
        # Launch up to per_question_workers tasks
        batch = pending_indices[:per_question_workers]
        pending_indices = pending_indices[per_question_workers:]
        futures_map = {}
        with ThreadPoolExecutor(max_workers=len(batch) or 1) as executor:
            for idx in batch:
                futures_map[executor.submit(_ask, idx, questions[idx])] = idx
            for fut in as_completed(futures_map):
                idx_q = futures_map[fut]
                try:
                    _, (answer, usage) = fut.result()
                    all_answers[idx_q] = answer
                    total_prompt += usage.get("prompt_tokens", 0)
                    total_completion += usage.get("completion_tokens", 0)
                    total_cached += usage.get("cached_tokens", 0)
                    completed += 1
                    if progress_bar:
                        progress_bar.progress(completed / len(questions))
                except Exception:
                    # If globally disabled, mark as none and do not retry
                    if GPT_DISABLED.is_set():
                        all_answers[idx_q] = "none"
                        completed += 1
                        if progress_bar:
                            progress_bar.progress(completed / len(questions))
                    else:
                        # Rate limited or transient error: re-queue this question
                        pending_indices.append(idx_q)
                        # Gentle wait to respect rate limits when re-queuing
                        wait = SLEEP_BETWEEN_CALLS
                        if GPT_RATE_LIMITED.is_set():
                            wait = max(wait, 4.0)
                        time.sleep(wait)
    
    result = {
        'timestamp': datetime.now().isoformat(),
        'company_name': company_name,
        'ticker': ticker,
        'url': url,
        # Input tokens are not limited; we still report usage-derived values when available
        'input_tokens': max(0, total_prompt - total_cached),
        'cached_tokens': total_cached,
        'output_tokens': total_completion,
        'answers': all_answers
    }
    
    return result

# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    inject_crazy_styles()
    render_navbar()
    # Guided stepper
    current_step = 0
    if st.session_state.get("gc") and CLIENTS:
        current_step = 1
    if st.session_state.get("sheet_tabs"):
        current_step = 2
    if st.session_state.get("processing"):
        current_step = 3
    render_stepper(current_step)
    st.markdown("<div class='divider'>Workflow</div>", unsafe_allow_html=True)
    # View selector for logs
    if 'view' not in st.session_state:
        st.session_state.view = 'app'
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    
    # Initialize session state for storing tabs
    if 'sheet_tabs' not in st.session_state:
        st.session_state.sheet_tabs = []
    if 'gc' not in st.session_state:
        st.session_state.gc = None
    if 'preview_expanded' not in st.session_state:
        st.session_state.preview_expanded = False
    if 'custom_questions' not in st.session_state:
        st.session_state.custom_questions = []
    if 'original_questions' not in st.session_state:
        st.session_state.original_questions = []
    if 'urls' not in st.session_state:
        st.session_state.urls = []
    if 'new_question_text' not in st.session_state:
        st.session_state.new_question_text = ""
    if 'new_url_text' not in st.session_state:
        st.session_state.new_url_text = ""
    if 'sheet_updated' not in st.session_state:
        st.session_state.sheet_updated = False
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("Configuration")
        st.markdown("<div class='mini'>Hover icons for tips.</div>", unsafe_allow_html=True)
        
        # File uploaders
        st.subheader("Upload Credentials 🪪")
        
        service_account_file = st.file_uploader(
            "Google Service Account JSON",
            type=["json"],
            help="Upload your Google Cloud Service Account JSON file",
            key="service_account"
        )
        
        api_key_file = st.file_uploader(
            "Primary OpenAI API Key (.txt)",
            type=["txt"],
            help="Upload a text file containing your OpenAI API key",
            key="api_key"
        )

        # Additional independent API keys via multiple .txt uploads
        st.subheader("Additional API Keys (.txt) 🔑")
        additional_api_key_files = st.file_uploader(
            "Upload one key per file",
            type=["txt"],
            accept_multiple_files=True,
            help="Each file should contain exactly one OpenAI API key. Used concurrently, one URL per key.",
            key="extra_api_keys_files"
        )
        extra_keys = []
        if additional_api_key_files:
            for f in additional_api_key_files:
                try:
                    data = f.getvalue()
                except Exception:
                    # Fallback to read()
                    data = f.read()
                try:
                    key_str = data.decode("utf-8").strip()
                except Exception:
                    key_str = data.decode("utf-8", errors="ignore").strip()
                if key_str:
                    extra_keys.append(key_str)
        st.session_state.extra_api_keys = extra_keys
        
        # Configuration options
        st.subheader("Settings ⚙️")
        spreadsheet_url = st.text_input(
            "Google Sheets URL",
            value="https://docs.google.com/spreadsheets/d/1DXoR2yqrnrCYuu4rTO27FIUPrLFPBRWX2h5vEi40xiI/edit?usp=sharing",
            help="URL of the Google Sheet containing URLs and questions",
            key="spreadsheet_url_input",
        )
        # Auto-refresh preview when the spreadsheet URL changes
        prev_url = st.session_state.get("prev_spreadsheet_url")
        if spreadsheet_url != prev_url:
            st.session_state.prev_spreadsheet_url = spreadsheet_url
            # Clear cached tabs, URLs, and questions so preview reloads
            st.session_state.sheet_tabs = []
            st.session_state.urls = []
            st.session_state.custom_questions = []
            st.session_state.original_questions = []
            st.session_state.sheet_updated = False
            # If authorized, refresh available tabs automatically
            if st.session_state.get("gc"):
                try:
                    st.session_state.sheet_tabs = get_sheet_tabs(st.session_state.gc, spreadsheet_url) or []
                except Exception:
                    st.session_state.sheet_tabs = []

        # Always append rows at the end; no overwrite mode, no toggle

        # Live logs are shown during processing; remove separate logs view
        
        # Button to fetch sheet tabs
        if st.button("📋 Fetch Sheet Tabs", key="fetch_tabs"):
            if spreadsheet_url and service_account_file and st.session_state.gc:
                with st.spinner("Fetching sheet tabs..."):
                    tabs = get_sheet_tabs(st.session_state.gc, spreadsheet_url)
                    if tabs:
                        st.session_state.sheet_tabs = tabs
                        st.success(f"Found {len(tabs)} tabs")
                    else:
                        st.error("No tabs found or couldn't access the sheet")
        
        # Display tabs if available
        if st.session_state.sheet_tabs:
            input_tab_name = st.selectbox(
                "Input Tab Name",
                options=st.session_state.sheet_tabs,
                index=0,
                help="Select the tab containing URLs and questions"
            )
            # Auto-refresh preview when the input tab changes
            prev_input_tab = st.session_state.get("prev_input_tab_name")
            if input_tab_name != prev_input_tab:
                st.session_state.prev_input_tab_name = input_tab_name
                st.session_state.urls = []
                st.session_state.custom_questions = []
                st.session_state.original_questions = []
            
            output_tab_name = st.selectbox(
                "Output Tab Name",
                options=st.session_state.sheet_tabs,
                index=min(1, len(st.session_state.sheet_tabs)-1) if len(st.session_state.sheet_tabs) > 1 else 0,
                help="Select the tab where results will be written"
            )
        else:
            input_tab_name = st.text_input(
                "Input Tab Name",
                value="Sheet1",
                help="Name of the tab containing URLs and questions"
            )
            # Auto-refresh preview when the input tab (text) changes
            prev_input_tab = st.session_state.get("prev_input_tab_name")
            if input_tab_name != prev_input_tab:
                st.session_state.prev_input_tab_name = input_tab_name
                st.session_state.urls = []
                st.session_state.custom_questions = []
                st.session_state.original_questions = []
            
            output_tab_name = st.text_input(
                "Output Tab Name",
                value="Results",
                help="Name of the tab where results will be written"
            )
    
    # Main content area
    if not CORE_LIBS_AVAILABLE:
        st.error("Required libraries not installed. Please check the error message above.")
        return

    # Early initialization so status reflects immediately
    # Initialize OpenAI once a key is available
    if not CLIENTS:
        candidate_key = None
        if 'api_key' in st.session_state and st.session_state.api_key is not None:
            try:
                data = st.session_state.api_key.getvalue()
            except Exception:
                data = st.session_state.api_key.read()
            try:
                candidate_key = data.decode("utf-8").strip()
            except Exception:
                candidate_key = data.decode("utf-8", errors="ignore").strip()
        elif LLM_API_KEY:
            candidate_key = LLM_API_KEY
        elif st.session_state.get("extra_api_keys"):
            candidate_key = st.session_state.get("extra_api_keys")[0]
        if candidate_key:
            try:
                initialize_openai(candidate_key)
                st.session_state.openai_connected = True
                log("Initialization: OpenAI API connected")
            except Exception:
                st.session_state.openai_connected = False
    # Initialize Google Sheets when JSON is uploaded
    if st.session_state.get("gc") is None and 'service_account' in st.session_state and st.session_state.service_account is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_json:
                tmp_json.write(st.session_state.service_account.read())
                st.session_state.service_account_temp_path = tmp_json.name
            st.session_state.gc = get_google_sheet_client(st.session_state.service_account_temp_path)
            if st.session_state.gc:
                st.session_state.sheets_authorized = True
                log("Initialization: Google Sheets authorized")
        except Exception:
            st.session_state.sheets_authorized = False
    # Auto-fetch tabs if authorized and URL present
    if st.session_state.get("gc") and spreadsheet_url and not st.session_state.get("sheet_tabs"):
        tabs = get_sheet_tabs(st.session_state.gc, spreadsheet_url)
        st.session_state.sheet_tabs = tabs or []

    # Status section
    st.markdown("<div class='divider'>Status</div>", unsafe_allow_html=True)
    openai_ok = bool(st.session_state.get("openai_connected") or CLIENTS)
    gsheet_ok = bool(st.session_state.get("sheets_authorized") or st.session_state.get("gc"))
    tab_count = len(st.session_state.get("sheet_tabs", []))
    selected_input_tab = input_tab_name if 'input_tab_name' in locals() and isinstance(input_tab_name, str) else ""
    render_status_summary(openai_ok, gsheet_ok, tab_count, selected_input_tab)
    
    # Check if files are uploaded (OpenAI only)
    extra_keys = st.session_state.get("extra_api_keys", []) or []
    if service_account_file and (api_key_file or extra_keys or LLM_API_KEY):
        # Save uploaded files to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_json:
            tmp_json.write(service_account_file.read())
            service_account_path = tmp_json.name
        
        api_key = ""
        api_key_path = None
        if api_key_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as txt_json:
                txt_json.write(api_key_file.read())
                api_key_path = txt_json.name
            # Read API key
            with open(api_key_path, "r") as f:
                api_key = f.read().strip()
        else:
            # No primary file; prefer env key, otherwise use first extra key
            if LLM_API_KEY:
                api_key = LLM_API_KEY
            elif extra_keys:
                api_key = extra_keys[0]
            else:
                st.error("No OpenAI API key provided. Upload key files (.txt) or set LLM_API_KEY env var.")
                return
        
        # Initialize OpenAI
        try:
            initialize_openai(api_key)
            st.success("✅ OpenAI API initialized successfully")
            log("Initialization: OpenAI API connected")
        except Exception as e:
            st.error(f"Failed to initialize OpenAI: {e}")
            return

        # Skip pre-run credits/usage display per user request
        
        # Initialize Google Sheets (if not already initialized)
        if st.session_state.gc is None:
            try:
                st.session_state.gc = get_google_sheet_client(service_account_path)
                if st.session_state.gc:
                    st.success("✅ Google Sheets API initialized successfully")
                    log("Initialization: Google Sheets authorized")
                else:
                    st.error("Failed to initialize Google Sheets")
                    return
            except Exception as e:
                st.error(f"Failed to initialize Google Sheets: {e}")
                return
        
        # Display sheet information
        if spreadsheet_url and st.session_state.gc:
            if st.session_state.sheet_tabs:
                st.info(f"Found {len(st.session_state.sheet_tabs)} sheet tabs.")
                log(f"Initialization: {len(st.session_state.sheet_tabs)} sheet tabs detected")
            else:
                st.info("Click 'Fetch Sheet Tabs' in the sidebar to see available tabs")
        
        # Preview section with expander
        if spreadsheet_url and input_tab_name and st.session_state.gc:
            preview_expander = st.expander("👁️ Preview & Edit Input Data", expanded=st.session_state.preview_expanded)
            
            with preview_expander:
                # Toggle button to expand/collapse
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader("Input Data Preview & Editor")
                with col2:
                    if st.button("🔄 Refresh from Sheet"):
                        # Clear cached data to force reload
                        if 'urls' in st.session_state:
                            del st.session_state.urls
                        if 'custom_questions' in st.session_state:
                            del st.session_state.custom_questions
                        if 'original_questions' in st.session_state:
                            del st.session_state.original_questions
                        st.rerun()
                
                try:
                    # Fetch data from sheet
                    urls, questions = get_inputs_from_sheet(st.session_state.gc, spreadsheet_url, input_tab_name)
                    
                    if urls and questions:
                        # Store original data
                        if not st.session_state.original_questions:
                            st.session_state.original_questions = questions.copy()
                        
                        # Initialize custom questions if not set
                        if not st.session_state.custom_questions:
                            st.session_state.custom_questions = questions.copy()
                        
                        # Store URLs
                        st.session_state.urls = urls
                        
                        st.markdown("<div class='divider'>Sheet Configuration</div>", unsafe_allow_html=True)
                        col1, col2 = st.columns([2,1])
                        
                        with col1:
                            st.subheader("🔗 URLs")
                            st.metric("Number of URLs", len(urls))
                            
                            # URL List
                            url_container = st.container()
                            with url_container:
                                for i, url in enumerate(urls[:20], 1):
                                    st.write(f"{i}. [{url[:80]}...]({url})")
                                
                                if len(urls) > 20:
                                    st.write(f"... and {len(urls) - 20} more URLs")
                            
                            # Add new URL section
                            st.subheader("➕ Add New URL")
                            new_url = st.text_input(
                                "Enter new URL:",
                                value=st.session_state.new_url_text,
                                key="new_url_input",
                                placeholder="https://www.sec.gov/..."
                            )
                            
                            if st.button("📥 Add URL to Sheet", key="add_url_button"):
                                if new_url and new_url.startswith(('http://', 'https://')):
                                    success = add_url_to_sheet(st.session_state.gc, spreadsheet_url, input_tab_name, new_url)
                                    if success:
                                        st.success(f"URL added to sheet!")
                                        st.session_state.new_url_text = ""
                                        st.session_state.sheet_updated = True
                                        # Clear cached URLs to force reload
                                        if 'urls' in st.session_state:
                                            del st.session_state.urls
                                        st.rerun()
                                    else:
                                        st.error("Failed to add URL to sheet")
                                else:
                                    st.error("Please enter a valid URL starting with http:// or https://")
                        
                        with col2:
                            st.subheader("❓ Questions (Editable)")
                            st.info("Edit questions below. Changes will be saved to the Google Sheet.")
                            
                            # Editable questions
                            for i in range(len(questions)):
                                edited_question = st.text_area(
                                    f"Question {i+1}",
                                    value=st.session_state.custom_questions[i] if i < len(st.session_state.custom_questions) else "",
                                    key=f"question_{i}",
                                    height=80
                                )
                                if i < len(st.session_state.custom_questions):
                                    st.session_state.custom_questions[i] = edited_question
                            
                            # Add new question section
                            st.subheader("➕ Add New Question")
                            new_question = st.text_area(
                                "Enter new question:",
                                value=st.session_state.new_question_text,
                                key="new_question_input",
                                height=80,
                                placeholder="What is the CEO's compensation?"
                            )
                            
                            if st.button("➕ Add Question", key="add_question_button"):
                                if new_question.strip():
                                    st.session_state.custom_questions.append(new_question.strip())
                                    st.session_state.new_question_text = ""
                                    st.success("Question added to list!")
                                    st.rerun()
                                else:
                                    st.error("Please enter a question")
                            
                            # Save changes button - updates Google Sheet
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("💾 Save to Google Sheet", key="save_sheet_button", type="primary"):
                                    # Save questions to Google Sheet
                                    if st.session_state.custom_questions:
                                        success = update_questions_in_sheet(
                                            st.session_state.gc, 
                                            spreadsheet_url, 
                                            input_tab_name, 
                                            st.session_state.custom_questions
                                        )
                                        if success:
                                            st.success("✅ Questions saved to Google Sheet!")
                                            st.session_state.sheet_updated = True
                                            st.session_state.original_questions = st.session_state.custom_questions.copy()
                                        else:
                                            st.error("Failed to save questions to Google Sheet")
                                    else:
                                        st.error("No questions to save")
                            
                            with col2:
                                if st.button("🔄 Reset to Sheet", key="reset_button"):
                                    st.session_state.custom_questions = st.session_state.original_questions.copy()
                                    st.success("Questions reset to original values from sheet!")
                                    st.rerun()
                        
                        # Display summary + context metrics
                        st.markdown("<div class='divider'>Context</div>", unsafe_allow_html=True)
                        summary_col1, summary_col2 = st.columns([2,1])
                        with summary_col1:
                            st.metric("Total URLs", len(urls))
                        with summary_col2:
                            st.metric("Total Questions", len(st.session_state.custom_questions))
                        # Right-side context panel
                        est_runtime_min = max(1, int((len(urls) / max(1, min(len(CLIENTS) if CLIENTS else 1, len(urls)))) * 2))
                        render_context_panel(
                            total_filings=len(urls),
                            model=MODEL,
                            concurrency=max(1, min(len(CLIENTS) if CLIENTS else 1, len(urls))),
                            est_runtime_min=est_runtime_min,
                            est_cost=None,
                        )
                        
                    else:
                        st.warning("No data found in the selected input tab")
                except Exception as e:
                    st.warning(f"Preview unavailable: {e}")
        
        # Process button
        # Compute prerequisites for processing
        prereq_ok = bool(st.session_state.gc) and bool(CLIENTS) and bool(spreadsheet_url) and bool(input_tab_name) and bool(output_tab_name)
        if not prereq_ok:
            st.markdown("<span class='hint'>Processing is disabled until credentials and sheet are configured.</span>", unsafe_allow_html=True)
            st.markdown("<div class='btn-primary disabled'>🚀 Process All Filings</div>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='hint'>This will process all rows in the input tab.</span>", unsafe_allow_html=True)
            if st.button("🚀 Process All Filings", type="primary", use_container_width=True, key="process"):
                st.session_state.processing = True
            # Do not start processing until the button is clicked
            if not st.session_state.get("processing"):
                st.info("Ready. Click 'Process All Filings' to start.")
                return
            if not spreadsheet_url:
                st.error("Please enter a Google Sheets URL")
                return
            if not input_tab_name:
                st.error("Please select an input tab")
                return
            if not output_tab_name:
                st.error("Please select an output tab")
                return
            
            # Use custom questions if available, otherwise fetch from sheet
            if st.session_state.custom_questions:
                questions = st.session_state.custom_questions
                st.info(f"Using {len(questions)} custom questions")
            else:
                with st.spinner("Reading questions from Google Sheet..."):
                    _, questions = get_inputs_from_sheet(st.session_state.gc, spreadsheet_url, input_tab_name)
            
            # Get URLs from session state or fetch from sheet
            if 'urls' in st.session_state and st.session_state.urls:
                urls = st.session_state.urls
            else:
                with st.spinner("Reading URLs from Google Sheet..."):
                    urls, _ = get_inputs_from_sheet(st.session_state.gc, spreadsheet_url, input_tab_name)
            
            if not urls:
                st.error("No URLs found. Please check your input tab.")
                return
            
            if not questions:
                st.error("No questions found. Please check your input tab or edit questions in preview.")
                return
            
            st.info(f"Processing all {len(urls)} URLs with {len(questions)} questions")
            
            # Setup output tab
            with st.spinner("Setting up output tab..."):
                worksheet = setup_output_tab(st.session_state.gc, spreadsheet_url, output_tab_name, len(questions))
            
            if not worksheet:
                st.error("Failed to setup output tab")
                return

            # Skip write probes to avoid modifying existing cells; errors will surface during append

            # Build URL→row index map from input tab and create revised input tab
            url_rows = get_input_url_rows(st.session_state.gc, spreadsheet_url, input_tab_name)
            url_row_map: Dict[str, int] = {}
            for u, idx in url_rows:
                if u not in url_row_map:
                    url_row_map[u] = idx
            # Deduplicate URLs while preserving order and keeping only those present in input
            url_order: List[str] = []
            for u in urls:
                if u in url_row_map and u not in url_order:
                    url_order.append(u)
            # Populate revised input tab with unique URLs
            setup_revised_output_tab(st.session_state.gc, spreadsheet_url, url_order)

            # Process filings concurrently, assigning one URL per API key
            progress_bar = st.progress(0.0)
            progress_html = st.empty()
            dash_placeholder = st.empty()
            results_container = st.container()
            # Live attempt counts display
            attempt_placeholder = st.empty()
            # Progress text display
            progress_text_placeholder = st.empty()
            # Live logs display
            st.markdown("<div class='divider'>Execution</div>", unsafe_allow_html=True)
            logs_expander = st.expander("📜 Execution Logs", expanded=False)
            logs_placeholder = logs_expander.empty()

            # Always write immediately on completion

            successful_count = 0
            failed_count = 0
            total_input_tokens = 0
            total_cached_tokens = 0
            total_output_tokens = 0

            # Preserve strict input order with de-duplication
            pending_urls = url_order.copy()
            results_map: Dict[str, Optional[Dict]] = {}

            def _process_with_client(cli, url):
                # Assign thread-local client for this worker
                thread_local.client = cli
                # Do not use Streamlit UI in worker thread
                result = process_single_filing(url, questions, progress_bar=None, status_text=None)
                return bool(result), result

            num_workers = max(1, min(len(CLIENTS) if CLIENTS else 1, len(pending_urls)))
            if num_workers == 0:
                num_workers = 1

            completed_files = 0
            inflight_urls = set()
            processed_urls = set()
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                running: Dict = {}
                # Seed initial tasks: one per client
                for i in range(num_workers):
                    cli = CLIENTS[i] if CLIENTS else client
                    if not pending_urls:
                        break
                    # Skip any URLs already inflight/processed (defensive)
                    while pending_urls and (pending_urls[0] in inflight_urls or pending_urls[0] in processed_urls):
                        pending_urls.pop(0)
                    if not pending_urls:
                        break
                    url = pending_urls.pop(0)
                    log(f"Queued URL for processing: {url}")
                    fut = executor.submit(_process_with_client, cli, url)
                    running[fut] = (cli, url)
                    inflight_urls.add(url)
                # Process until all URLs are handled
                while running:
                    done, _ = wait(list(running.keys()), return_when=FIRST_COMPLETED)
                    for fut in done:
                        cli, url = running.pop(fut)
                        try:
                            ok, result = fut.result()
                            if ok and result:
                                successful_count += 1
                                total_input_tokens += result['input_tokens']
                                total_cached_tokens += result['cached_tokens']
                                total_output_tokens += result['output_tokens']
                                # Write immediately to the specific row in output tab
                                out_row = url_row_map.get(result['url'])
                                write_result_to_sheet_row(worksheet, out_row if out_row else 2, result)
                                processed_urls.add(result['url'])
                                log(f"Completed: {result['url']} | input={result['input_tokens']} cached={result['cached_tokens']} output={result['output_tokens']}")
                            else:
                                failed_count += 1
                                # Write a failure placeholder at the specific row
                                fail_res = build_failure_result(url, len(questions))
                                out_row = url_row_map.get(url)
                                write_result_to_sheet_row(worksheet, out_row if out_row else 2, fail_res)
                                processed_urls.add(url)
                                log(f"Failed to process: {url}")
                        except Exception as e:
                            failed_count += 1
                            results_map[url] = None
                            processed_urls.add(url)
                            log(f"Processing error: {e}")
                        # This URL is no longer inflight
                        if url in inflight_urls:
                            inflight_urls.discard(url)
                        completed_files += 1
                        progress_bar.progress(min(1.0, completed_files / max(1, len(url_order))))
                        # Styled progress bar
                        pct = int(min(100, (completed_files / max(1, len(url_order))) * 100))
                        progress_html.markdown(f"<div class='panel'><div class='progress'><div class='bar' style='width:{pct}%;'></div></div></div>", unsafe_allow_html=True)
                        # Update live dashboard
                        pending_count = max(0, len(url_order) - completed_files)
                        inflight_count = len(inflight_urls)
                        dash_html = f"""
                        <div class='panel'>
                          <div class='grid'>
                            <div class='card'><h3>Completed</h3><div class='value success'>{successful_count + failed_count:,}</div></div>
                            <div class='card'><h3>Successful</h3><div class='value success'>{successful_count:,}</div></div>
                            <div class='card'><h3>Failed</h3><div class='value danger'>{failed_count:,}</div></div>
                            <div class='card'><h3>In-Flight</h3><div class='value accent'>{inflight_count:,}</div></div>
                            <div class='card'><h3>Pending</h3><div class='value'>{pending_count:,}</div></div>
                          </div>
                        </div>
                        """
                        dash_placeholder.markdown(dash_html, unsafe_allow_html=True)
                        # Progress text (current / total)
                        progress_text_placeholder.markdown(f"**Progress:** {completed_files}/{len(url_order)} filings processed")
                        # Update attempt counts display (top 10 by attempts)
                        if ATTEMPT_COUNTS:
                            top = sorted(ATTEMPT_COUNTS.items(), key=lambda x: x[1], reverse=True)[:10]
                            lines = [f"{u[:60]}... -> attempts: {c}" for u, c in top]
                            attempt_placeholder.markdown("**Retry attempts (live):**\n" + "\n".join(lines))
                        # Update live logs display (last 50 lines)
                        if LOG_BUFFER:
                            latest = LOG_BUFFER[-80:]
                            rendered = []
                            for ln in latest:
                                cls = "logline"
                                low = ln.lower()
                                if "completed:" in low:
                                    cls += " ok"
                                elif ("failed" in low) or ("error" in low):
                                    cls += " fail"
                                elif ("queued url" in low) or ("fetching:" in low):
                                    cls += " queue"
                                rendered.append(f"<div class='{cls}'>{ln}</div>")
                            logs_html = "<div class='logbox'>" + "".join(rendered) + "</div>"
                            logs_placeholder.markdown(logs_html, unsafe_allow_html=True)
                        # Assign next url to this client
                        if pending_urls:
                            # Skip duplicates or ones currently inflight/processed
                            while pending_urls and (pending_urls[0] in inflight_urls or pending_urls[0] in processed_urls):
                                pending_urls.pop(0)
                            if pending_urls:
                                next_url = pending_urls.pop(0)
                                log(f"Queued URL for processing: {next_url}")
                                nfut = executor.submit(_process_with_client, cli, next_url)
                                running[nfut] = (cli, next_url)
                                inflight_urls.add(next_url)

            # Always append rows immediately; skip any final batch write
            st.info("Rows are appended immediately upon completion.")
            
            # Final summary
            st.info("Processing complete!")
            progress_bar.progress(1.0)
            
            st.success("🎉 Processing Complete!")
            
            # Display summary (styled)
            st.subheader("Summary")
            total_cost = ((total_input_tokens / 1000) * 0.015) + ((total_output_tokens / 1000) * 0.060)
            render_stats_showcase(
                total_files=len(url_order),
                successful=successful_count,
                failed=failed_count,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cached_tokens=total_cached_tokens,
                total_cost=total_cost,
            )
            
            # Token Usage & Estimated Cost
            st.subheader("Token Usage")
            u1, u2 = st.columns(2)
            with u1:
                st.metric("Total Tokens Used", f"{(total_input_tokens + total_output_tokens):,}")
            with u2:
                st.metric("Cached Tokens", f"{total_cached_tokens:,}")

            # Post-run credits and usage
            # Estimated Cost (based on model pricing)
            st.subheader("Estimated Cost")
            input_cost = (total_input_tokens / 1000) * 0.015
            output_cost = (total_output_tokens / 1000) * 0.060
            total_cost = input_cost + output_cost
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Input Cost", f"${input_cost:.4f}")
            with c2:
                st.metric("Output Cost", f"${output_cost:.4f}")
            with c3:
                st.metric("Total Cost", f"${total_cost:.4f}")
            
            # Link to results
            st.markdown(f"**Results written to:** [{output_tab_name} tab]({spreadsheet_url})")
            # Footer
            render_footer(MODEL, spreadsheet_url)
            
            # Cleanup temp files
            try:
                os.unlink(service_account_path)
                if api_key_path:
                    os.unlink(api_key_path)
            except:
                pass
    
    else:
        # Show instructions when no files uploaded
        st.info("👈 Please upload credentials in the sidebar to get started")
        # Subtle motif illustration
        st.markdown(
                """
                <div class="panel" style="text-align:center;">
                    <svg width="200" height="120" viewBox="0 0 200 120" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                        <rect x="20" y="20" width="160" height="80" rx="8" fill="none" stroke="#18223a" stroke-width="2"/>
                        <rect x="30" y="32" width="70" height="12" rx="4" fill="#18223a"/>
                        <rect x="30" y="50" width="140" height="8" rx="4" fill="#18223a"/>
                        <rect x="30" y="64" width="120" height="8" rx="4" fill="#18223a"/>
                        <rect x="30" y="78" width="90" height="8" rx="4" fill="#18223a"/>
                    </svg>
                    <div class="mini">SEC filing processing pipeline</div>
                </div>
                """,
                unsafe_allow_html=True,
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Prerequisites")
            st.markdown("""
            1. **Google Cloud Service Account JSON**
               - Create a service account in Google Cloud Console
               - Enable Google Sheets API
               - Download the JSON credentials file
            
            2. **OpenAI API Key**
               - Get API key from [OpenAI Platform](https://platform.openai.com)
               - Save it in a text file
            
            3. **Google Sheet Setup**
               - Create a Google Sheet with:
                 - Row 1: Questions in columns B onward
                 - Column A: URLs starting from row 2
            """)
        
        with col2:
            st.subheader("🎯 How to Use")
            st.markdown("""
            1. **Upload Credentials**
               - Upload your Google Service Account JSON file
               - Upload your OpenAI API key text file
            
            2. **Configure Settings**
               - Enter your Google Sheets URL
               - Click "Fetch Sheet Tabs" to see available tabs
               - Select input and output tabs from dropdowns
            
            3. **Preview and Edit Data**
               - Expand "Preview & Edit Input Data" section
               - View URLs and questions
               - **Edit questions directly** in text areas
               - **Add new questions** using the "Add Question" section
               - **Add new URLs** using the "Add URL" section
               - Click "Save to Google Sheet" to update the actual spreadsheet
            
            4. **Process All Filings**
               - Click "Process All Filings" button
               - **Processes ALL URLs** by default
               - Monitor progress in real-time
               - View results in the output tab
            
            **Features:**
            - ✅ **Bidirectional sync** - edits save to Google Sheet
            - ✅ **Add new questions** to the list
            - ✅ **Add new URLs** directly to the sheet
            - ✅ Collapsible preview section
            - ✅ Processes all files by default
            - ✅ Full document processing
            - ✅ Prefix caching for cost savings
            - ✅ Real-time progress tracking
            - ✅ Cost estimation
            - ✅ Automatic 'none' for missing info
            """)

 
if __name__ == "__main__":
    main()

