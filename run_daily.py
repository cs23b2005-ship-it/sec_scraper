import os
import json
import re
import time
import random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None


def remove_front_zeros(s):
    return s.lstrip('0') if isinstance(s, str) else s


def id_correction(id_str):
    parts = str(id_str).split(":")
    first_part = parts[0].replace("-", "") if parts else ""
    second_part = parts[1] if len(parts) > 1 else ""
    return f"{first_part}/{second_part}"


def checked_symbol(entity_list):
    symbol = entity_list[1].strip(")").strip().replace(")", "") if len(entity_list) > 1 else ""
    if "CIK " in symbol:
        symbol = ""
    return symbol


def format_date_str(s):
    mon = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    dt = None
    if isinstance(s, str) and s:
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except Exception:
                pass
    if dt is None:
        dt = datetime.now()
    return f"{dt.day:02d}-{mon[dt.month-1]}-{dt.year}"


def format_bytes(n):
    try:
        n = int(n)
    except Exception:
        return None
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.0f} {u}" if u == "B" else f"{size:.1f} {u}"
        size /= 1024


def get_content_length(session, url, headers):
    try:
        r = session.head(url, headers=headers, timeout=15, allow_redirects=True)
        if 200 <= r.status_code < 300:
            cl = r.headers.get('Content-Length')
            if cl:
                return int(cl)
        r = session.get(url, headers=headers, timeout=20, stream=True)
        if 200 <= r.status_code < 300:
            cl = r.headers.get('Content-Length')
            if cl:
                return int(cl)
    except Exception as e:
        print(f"[size] error fetching size for {url}: {e}")
    return None


def extarct_data(q, date_range, category, startdt, enddt, forms, page, from_, entityName, size=100, sort=None, max_total_retries=2):
    """
    Deterministic SEC search-index fetch with:
    - explicit 'from' and 'size'
    - session-level retries for HTTP errors
    - additional manual retries with exponential backoff/jitter for transient 5xx/504/429 and JSON errors
    """
    url = "https://efts.sec.gov/LATEST/search-index"

    # normalize page/from
    try:
        page_int = int(page) if page is not None else 1
    except:
        page_int = 1
    try:
        from_int = int(from_) if from_ is not None else (page_int - 1) * int(size)
    except:
        from_int = (page_int - 1) * int(size)

    params = {
        "q": q,
        "dateRange": date_range,
        "category": category,
        "startdt": startdt,
        "enddt": enddt,
        "forms": forms,
        "entityName": entityName,
        "from": from_int,
        "size": int(size)
    }
    if sort:
        params["sort"] = sort

    params = {k: v for k, v in params.items() if v not in (None, "", [])}

    # Use a clear User-Agent (include contact) — replace contact token with your address if you have one
    headers = {
        "User-Agent": "sec-scraper/1.0 (contact: abhishek2005.siva@gmail.com)",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
        "Connection": "keep-alive"
    }

    # Session with urllib3 Retry to handle low-level network retries
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)

    MAX_ATTEMPTS = 6
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        try:
            resp = session.get(url, params=params, headers=headers, timeout=30)
        except Exception as e:
            print(f"[extarct_data] network exception attempt={attempt}: {e}")
            sleep_for = (2 ** (attempt - 1)) * 0.5 + random.uniform(0, 0.5)
            time.sleep(min(sleep_for, 10))
            continue

        status = resp.status_code
        if status in (429, 500, 502, 503, 504):
            snippet = ""
            try:
                snippet = resp.text[:200].replace("\n", " ")
            except:
                pass
            print(f"[extarct_data] transient status {status} attempt={attempt} body={snippet}")
            if attempt < MAX_ATTEMPTS:
                sleep_for = (2 ** (attempt - 1)) * 0.5 + random.uniform(0, 0.5)
                time.sleep(min(sleep_for, 10))
                continue
            else:
                print(f"[extarct_data] giving up after {attempt} attempts due to status {status}")
                return []

        if status != 200:
            snippet = ""
            try:
                snippet = resp.text[:200].replace("\n", " ")
            except:
                pass
            print(f"[extarct_data] non-retriable status {status} body={snippet}")
            return []

        try:
            data = resp.json()
        except Exception as e:
            print(f"[extarct_data] JSON parse error attempt={attempt}: {e}")
            if attempt < MAX_ATTEMPTS:
                sleep_for = (2 ** (attempt - 1)) * 0.5 + random.uniform(0, 0.5)
                time.sleep(min(sleep_for, 10))
                continue
            print("[extarct_data] JSON parse failed repeatedly, giving up")
            return []

        total_reported = data.get("hits", {}).get("total", {}).get("value", 0)
        print(f"[extarct_data] page={page_int} from={from_int} size={size} reported_total={total_reported} attempt={attempt}")

        if page_int == 1 and total_reported == 0 and attempt <= max_total_retries:
            time.sleep(0.5 * attempt)
            continue

        hits = data.get("hits", {}).get("hits", [])
        master_data = []
        for pre_targt in hits:
            targte_data = pre_targt.get("_source", {})
            form = targte_data.get("form", "")
            file_type = targte_data.get("file_type", "")
            file_description = targte_data.get("file_description", "")
            ciks = targte_data.get("ciks", [])
            corrected_ciks = remove_front_zeros(ciks[0]) if ciks else ""
            pathParts = pre_targt.get('_id', '').split(':')
            adsh_raw = pathParts[0] if len(pathParts) > 0 else ""
            adsh = adsh_raw.replace('-', '') if isinstance(adsh_raw, str) else ""
            filing_detail_url = f"https://www.sec.gov/Archives/edgar/data/{corrected_ciks}/{adsh}/{adsh_raw}-index.html"
            fileName = pathParts[1] if len(pathParts) > 1 else ""
            url_out = ""
            if pre_targt.get('_id','').lower().endswith(".xml"):
                xsl = targte_data.get("xsl")
                if xsl:
                    url_out = f"https://www.sec.gov/Archives/edgar/data/{corrected_ciks}/{adsh}/{xsl}/{fileName}"
            else:
                url_out = f"https://www.sec.gov/Archives/edgar/data/{corrected_ciks}/{id_correction(pre_targt.get('_id',''))}"
            file_date = targte_data.get("file_date", "")
            repotring_for = targte_data.get("period_ending", "")
            entity = targte_data.get("display_names", [""])[0] if targte_data.get("display_names") else ""
            entity_list = entity.split("(")
            company_name = entity_list[0].strip()
            symbol = checked_symbol(entity_list)
            cik = str(targte_data.get("ciks", [""])[0]) if targte_data.get("ciks") else ""

            size_human = None

            master_data.append({
                "Record ID": pre_targt.get("_id", ""),
                "Form": form,
                "File": file_type,
                "File description": file_description,
                "URL": url_out,
                "Filing Detail": filing_detail_url,
                "Filed": file_date,
                "Reporting For": repotring_for,
                "Filing entity": company_name,
                "Filing entity located": targte_data.get("biz_locations",[None])[0] if targte_data.get("biz_locations") else None,
                "Filing entity incorporated": None,
                "Filer name": targte_data.get("display_names", [""])[-1] if len(targte_data.get("display_names", [])) > 1 else None,
                "Filer located": targte_data.get("biz_locations", [None])[-1] if len(targte_data.get("biz_locations", [])) > 1 else None,
                "Filer incorporated": None,
                "Symbol": symbol,
                "CIK": cik
            })
        return master_data


def authorize_gspread_from_env():
    if gspread is None or Credentials is None:
        print("gspread/google-auth not installed; skipping Sheets write.")
        return None
    sa_json_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_json_str:
        print("GOOGLE_SERVICE_ACCOUNT_JSON not provided; skipping Sheets write.")
        return None
    try:
        creds_info = json.loads(sa_json_str)
        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        credentials = Credentials.from_service_account_info(creds_info, scopes=scopes)
        gc = gspread.authorize(credentials)
        return gc
    except Exception as e:
        print(f"Failed to authorize gspread: {e}")
        return None


def write_df_to_sheet(gc, spreadsheet_id, worksheet_title, df):
    sh = gc.open_by_key(spreadsheet_id)
    # Ensure worksheet exists; create if missing
    try:
        wks = sh.worksheet(worksheet_title)
    except Exception:
        try:
            wks = sh.add_worksheet(title=worksheet_title, rows=100, cols=26)
            print(f"Created missing worksheet '{worksheet_title}'.")
        except Exception as e:
            raise e

    # Helper: convert 1-based column index to A1 letter (supports beyond Z)
    def col_letter(n: int) -> str:
        s = ""
        while n > 0:
            n, r = divmod(n - 1, 26)
            s = chr(65 + r) + s
        return s

    headers = list(df.columns)
    rows = df.astype(object).where(pd.notnull(df), "").values.tolist()

    # Determine existing row count safely
    try:
        existing = wks.get_all_values()
    except Exception:
        existing = []
    # Write headers if the first row is empty (even if there are values below)
    try:
        first_row = wks.row_values(1)
    except Exception:
        first_row = []
    first_row_empty = (len(first_row) == 0) or all((v or "").strip() == "" for v in first_row)

    # Always write using explicit A1 ranges to start at column A
    try:
        if first_row_empty:
            end_col = col_letter(len(headers)) if headers else "A"
            wks.update(range_name=f"A1:{end_col}1", values=[headers], value_input_option='USER_ENTERED')
        # Only write data rows if there are any
        if rows:
            # If sheet was previously empty and we just wrote headers, start at row 2; else append after last non-empty row
            start_row = 2 if (first_row_empty and len(existing) == 0) else (len(existing) + 1)
            wks.update(range_name=f"A{start_row}", values=rows, value_input_option='USER_ENTERED')
    except Exception as e:
        # Fallback: write each row explicitly to avoid alignment issues
        print(f"[write] Primary write failed: {e}, trying fallback...")
        try:
            if first_row_empty:
                end_col = col_letter(len(headers)) if headers else "A"
                wks.update(range_name=f"A1:{end_col}1", values=[headers], value_input_option='USER_ENTERED')
        except Exception as ex:
            print(f"[write] Header fallback failed: {ex}")
        if rows:
            base_row = 2 if (first_row_empty and len(existing) == 0) else (len(existing) + 1)
            for i, r in enumerate(rows):
                try:
                    wks.update(range_name=f"A{base_row + i}", values=[r], value_input_option='USER_ENTERED')
                except Exception as ex:
                    print(f"[write] Row {base_row + i} failed: {ex}")


def main():
    # Defaults with robust env handling: ignore empty strings
    forms_env = os.getenv("FORMS", "").strip()
    forms_str = forms_env if forms_env else "8-K"
    doc_env = os.getenv("DOC_SEARCH", "")
    entity_env = os.getenv("ENTITY_SEARCH", "")
    doc_search = doc_env.strip() if doc_env else ""
    entity_search = entity_env.strip() if entity_env else ""

    # Date range: only yesterday (UTC)
    days_back = int(os.getenv("DAYS_BACK", "1"))  # default 1-day window (yesterday only)
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    from_date = yesterday - timedelta(days=days_back - 1)
    to_date = yesterday

    date_range_str = "custom"
    category = "custom"

    page = 1
    FILINGS_PER_PAGE = int(os.getenv("PAGE_SIZE", "100"))

    print(f"Filters: forms='{forms_str}', doc='{doc_search}', entity='{entity_search}'")
    print(f"Date window: {from_date} to {to_date} (days_back={days_back})")

    all_filings = []
    seen_ids = set()
    while True:
        from_ = (page - 1) * FILINGS_PER_PAGE
        page_str = str(page)
        data = extarct_data(
            doc_search,
            date_range_str,
            category,
            from_date.strftime("%Y-%m-%d"),
            to_date.strftime("%Y-%m-%d"),
            forms_str,
            page_str,
            from_,
            entity_search,
            size=FILINGS_PER_PAGE
        )
        if not data:
            break
        # Deduplicate across pages by Record ID (fallback to URL)
        new_rows = []
        for row in data:
            rid = row.get("Record ID") or row.get("URL")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                new_rows.append(row)
        if not new_rows:
            print("Reached repeated page; stopping pagination to avoid duplicates.")
            break
        all_filings.extend(new_rows)
        print(f"Fetched page {page}, page new records: {len(new_rows)}, total unique: {len(all_filings)}")
        if len(data) < FILINGS_PER_PAGE:
            break
        page += 1
        time.sleep(0.2)

    df = pd.DataFrame(all_filings)
    # Define canonical headers so we can write header-only when there are no filings
    expected_headers = [
        "Date", "Time", "Form", "File", "File description", "URL", "Filing Detail",
        "Filed", "Reporting For", "Filing entity", "Filing entity located", "Filing entity incorporated",
        "Filer name", "Filer located", "Filer incorporated", "Symbol", "CIK"
    ]
    
    # Diagnostic: check for duplicate Record IDs or forms distribution
    if not df.empty:
        print(f"\nDiagnostic info:")
        print(f"  Total records: {len(df)}")
        print(f"  Unique Record IDs: {df['Record ID'].nunique()}")
        print(f"  Forms distribution: {df['Form'].value_counts().to_dict()}")
    
    if not df.empty:
        # Allow override via env var for testing (formats: HH:MM or h:mm AM/PM)
        def _parse_time_override(s: str) -> str | None:
            if not s:
                return None
            s_norm = s.strip()
            try:
                return datetime.strptime(s_norm, "%H:%M").strftime("%H:%M")
            except Exception:
                pass
            try:
                return datetime.strptime(s_norm.upper(), "%I:%M %p").strftime("%H:%M")
            except Exception:
                return None

        override = os.getenv("TIME_OVERRIDE", "")
        parsed = _parse_time_override(override)
        ist = ZoneInfo("Asia/Kolkata")
        export_time = parsed if parsed else datetime.now(ist).strftime("%H:%M")
        date_col = df.get("Filed")
        formatted_dates = date_col.apply(format_date_str) if date_col is not None else pd.Series([format_date_str(None)] * len(df))
        df.insert(0, "Date", formatted_dates)
        df.insert(1, "Time", export_time)
    else:
        # Build an empty DataFrame with the expected headers for header-only write
        df = pd.DataFrame(columns=expected_headers)
    print(f"Found {len(df)} filings")
    if df.empty:
        print("No filings found; check filters and date range.")

    # Optional Google Sheets write (supports ID or full URL)
    def resolve_spreadsheet_id(value: str | None) -> str | None:
        if not value:
            return None
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", value)
        return m.group(1) if m else value.strip()

    spreadsheet_value = os.getenv("SPREADSHEET_ID") or os.getenv("SPREADSHEET_URL")
    spreadsheet_id = resolve_spreadsheet_id(spreadsheet_value)
    worksheet_name = os.getenv("WORKSHEET_NAME", "Sheet1")
    if spreadsheet_id:
        gc = authorize_gspread_from_env()
        if gc is not None:
            try:
                write_df_to_sheet(gc, spreadsheet_id, worksheet_name, df)
                if df.empty:
                    print("📝 Wrote headers only (no filings today).")
                else:
                    print("✅ Results appended to the spreadsheet.")
            except Exception as e:
                print(f"Failed to write to spreadsheet: {e}")
        else:
            print("Skipping spreadsheet write due to missing credentials or libraries.")


if __name__ == "__main__":
    main()
