import os
import json
import re
import time
import random
from datetime import datetime, timedelta, timezone

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
    url = "https://efts.sec.gov/LATEST/search-index"

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

    headers = {
        "User-Agent": "sec-scraper/1.0 (contact: you@example.com)",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
        "Connection": "keep-alive"
    }

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
            if url_out:
                size_bytes = get_content_length(session, url_out, headers)
                size_human = format_bytes(size_bytes) if size_bytes is not None else None

            master_data.append({
                "Form": form,
                "File": file_type,
                "File description": file_description,
                "URL": url_out,
                "Filing Detail": filing_detail_url,
                "size": size_human,
                "Filed": file_date,
                "Reporting For": repotring_for,
                "Filing entity": company_name,
                "Filing entity located": None,
                "Filing entity incorporated": None,
                "Filer name": None,
                "Filer located": None,
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

    headers = list(df.columns)
    rows = df.astype(object).where(pd.notnull(df), "").values.tolist()

    # Determine existing row count safely
    try:
        existing = wks.get_all_values()
    except Exception:
        existing = []
    need_headers = len(existing) == 0

    # Always write using explicit A1 ranges to start at column A
    try:
        if need_headers:
            wks.update("A1", [headers], value_input_option='USER_ENTERED')
            existing = [headers]  # reflect header written
        start_row = len(existing) + 1
        wks.update(f"A{start_row}", rows, value_input_option='USER_ENTERED')
    except Exception as e:
        # Fallback: write each row explicitly to avoid alignment issues
        try:
            if need_headers:
                wks.update("A1", [headers], value_input_option='USER_ENTERED')
        except Exception:
            pass
        base_row = 2 if need_headers else len(existing) + 1
        for i, r in enumerate(rows):
            try:
                wks.update(f"A{base_row + i}", [r], value_input_option='USER_ENTERED')
            except Exception:
                pass


def main():
    # Defaults with robust env handling: ignore empty strings
    forms_env = os.getenv("FORMS", "").strip()
    forms_str = forms_env if forms_env else "10-K,10-Q,8-K,DEF 14A"
    doc_env = os.getenv("DOC_SEARCH", "")
    entity_env = os.getenv("ENTITY_SEARCH", "")
    doc_search = doc_env.strip() if doc_env else ""
    entity_search = entity_env.strip() if entity_env else ""

    # Date range: strictly yesterday (UTC) only
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    from_date = yesterday
    to_date = yesterday

    date_range_str = "custom"
    category = "custom"

    page = 1
    FILINGS_PER_PAGE = int(os.getenv("PAGE_SIZE", "100"))

    print(f"Filters: forms='{forms_str}', doc='{doc_search}', entity='{entity_search}'")
    all_filings = []
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
        all_filings.extend(data)
        print(f"Fetched page {page}, total records: {len(all_filings)}")
        if len(data) < FILINGS_PER_PAGE:
            break
        page += 1
        time.sleep(0.2)

    df = pd.DataFrame(all_filings)
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
        export_time = parsed if parsed else datetime.now(timezone.utc).strftime("%H:%M")
        date_col = df.get("Filed")
        formatted_dates = date_col.apply(format_date_str) if date_col is not None else pd.Series([format_date_str(None)] * len(df))
        df.insert(0, "Date", formatted_dates)
        df.insert(1, "Time", export_time)
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
    if spreadsheet_id and not df.empty:
        gc = authorize_gspread_from_env()
        if gc is not None:
            try:
                write_df_to_sheet(gc, spreadsheet_id, worksheet_name, df)
                print("✅ Results appended to the spreadsheet.")
            except Exception as e:
                print(f"Failed to write to spreadsheet: {e}")
        else:
            print("Skipping spreadsheet write due to missing credentials or libraries.")


if __name__ == "__main__":
    main()
