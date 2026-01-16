import requests
import json
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
# Add retry/backoff imports
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import random
import re

# Optional deps for Google Sheets connection
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

all_locations ={'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'District of Columbia', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'X1': 'United States', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming', 'A0': 'Alberta, Canada', 'A1': 'British Columbia, Canada', 'Z4': 'Canada (Federal Level)', 'A2': 'Manitoba, Canada', 'A3': 'New Brunswick, Canada', 'A4': 'Newfoundland, Canada', 'A5': 'Nova Scotia, Canada', 'A6': 'Ontario, Canada', 'A7': 'Prince Edward Island, Canada', 'A8': 'Quebec, Canada', 'A9': 'Saskatchewan, Canada', 'B0': 'Yukon, Canada', 'B2': 'Afghanistan', 'Y6': 'Aland Islands', 'B3': 'Albania', 'B4': 'Algeria', 'B5': 'American Samoa', 'B6': 'Andorra', 'B7': 'Angola', '1A': 'Anguilla', 'B8': 'Antarctica', 'B9': 'Antigua and Barbuda', 'C1': 'Argentina', '1B': 'Armenia', '1C': 'Aruba', 'C3': 'Australia', 'C4': 'Austria', '1D': 'Azerbaijan', 'C5': 'Bahamas', 'C6': 'Bahrain', 'C7': 'Bangladesh', 'C8': 'Barbados', '1F': 'Belarus', 'C9': 'Belgium', 'D1': 'Belize', 'G6': 'Benin', 'D0': 'Bermuda', 'D2': 'Bhutan', 'D3': 'Bolivia', '1E': 'Bosnia and Herzegovina', 'B1': 'Botswana', 'D4': 'Bouvet Island', 'D5': 'Brazil', 'D6': 'British Indian Ocean Territory', 'D9': 'Brunei Darussalam', 'E0': 'Bulgaria', 'X2': 'Burkina Faso', 'E2': 'Burundi', 'E3': 'Cambodia', 'E4': 'Cameroon', 'E8': 'Cape Verde', 'E9': 'Cayman Islands', 'F0': 'Central African Republic', 'F2': 'Chad', 'F3': 'Chile', 'F4': 'China', 'F6': 'Christmas Island', 'F7': 'Cocos (Keeling) Islands', 'F8': 'Colombia', 'F9': 'Comoros', 'G0': 'Congo', 'Y3': 'Congo, the Democratic Republic of the', 'G1': 'Cook Islands', 'G2': 'Costa Rica', 'L7': "Cote D'ivoire ", '1M': 'Croatia', 'G3': 'Cuba', 'G4': 'Cyprus', '2N': 'Czech Republic', 'G7': 'Denmark', '1G': 'Djibouti', 'G9': 'Dominica', 'G8': 'Dominican Republic', 'H1': 'Ecuador', 'H2': 'Egypt', 'H3': 'El Salvador', 'H4': 'Equatorial Guinea', '1J': 'Eritrea', '1H': 'Estonia', 'H5': 'Ethiopia', 'H7': 'Falkland Islands (Malvinas)', 'H6': 'Faroe Islands', 'H8': 'Fiji', 'H9': 'Finland', 'I0': 'France', 'I3': 'French Guiana', 'I4': 'French Polynesia', '2C': 'French Southern Territories', 'I5': 'Gabon', 'I6': 'Gambia', '2Q': 'Georgia (country)', '2M': 'Germany', 'J0': 'Ghana', 'J1': 'Gibraltar', 'J3': 'Greece', 'J4': 'Greenland', 'J5': 'Grenada', 'J6': 'Guadeloupe', 'GU': 'Guam', 'J8': 'Guatemala', 'Y7': 'Guernsey', 'J9': 'Guinea', 'S0': 'Guinea-bissau', 'K0': 'Guyana', 'K1': 'Haiti', 'K4': 'Heard Island and Mcdonald Islands', 'X4': 'Holy See (Vatican City State)', 'K2': 'Honduras', 'K3': 'Hong Kong', 'K5': 'Hungary', 'K6': 'Iceland', 'K7': 'India', 'K8': 'Indonesia', 'K9': 'Iran, Islamic Republic of', 'L0': 'Iraq', 'L2': 'Ireland', 'Y8': 'Isle of Man', 'L3': 'Israel', 'L6': 'Italy', 'L8': 'Jamaica', 'M0': 'Japan', 'Y9': 'Jersey', 'M2': 'Jordan', '1P': 'Kazakhstan', 'M3': 'Kenya', 'J2': 'Kiribati', 'M4': "Korea, Democratic People's Republic of ", 'M5': 'Korea, Republic of', 'M6': 'Kuwait', '1N': 'Kyrgyzstan', 'M7': "Lao People's Democratic Republic ", '1R': 'Latvia', 'M8': 'Lebanon', 'M9': 'Lesotho', 'N0': 'Liberia', 'N1': 'Libyan Arab Jamahiriya', 'N2': 'Liechtenstein', '1Q': 'Lithuania', 'N4': 'Luxembourg', 'N5': 'Macau', '1U': 'Macedonia, the Former Yugoslav Republic of', 'N6': 'Madagascar', 'N7': 'Malawi', 'N8': 'Malaysia', 'N9': 'Maldives', 'O0': 'Mali', 'O1': 'Malta', '1T': 'Marshall Islands', 'O2': 'Martinique', 'O3': 'Mauritania', 'O4': 'Mauritius', '2P': 'Mayotte', 'O5': 'Mexico', '1K': 'Micronesia, Federated States of', '1S': 'Moldova, Republic of', 'O9': 'Monaco', 'P0': 'Mongolia', 'Z5': 'Montenegro', 'P1': 'Montserrat', 'P2': 'Morocco', 'P3': 'Mozambique', 'E1': 'Myanmar', 'T6': 'Namibia', 'P5': 'Nauru', 'P6': 'Nepal', 'P7': 'Netherlands', 'P8': 'Netherlands Antilles', '1W': 'New Caledonia', 'Q2': 'New Zealand', 'Q3': 'Nicaragua', 'Q4': 'Niger', 'Q5': 'Nigeria', 'Q6': 'Niue', 'Q7': 'Norfolk Island', '1V': 'Northern Mariana Islands', 'Q8': 'Norway', 'P4': 'Oman', 'R0': 'Pakistan', '1Y': 'Palau', '1X': 'Palestinian Territory, Occupied', 'R1': 'Panama', 'R2': 'Papua New Guinea', 'R4': 'Paraguay', 'R5': 'Peru', 'R6': 'Philippines', 'R8': 'Pitcairn', 'R9': 'Poland', 'S1': 'Portugal', 'PR': 'Puerto Rico', 'S3': 'Qatar', 'S4': 'Reunion', 'S5': 'Romania', '1Z': 'Russian Federation', 'S6': 'Rwanda', 'Z0': 'Saint Barthelemy', 'U8': 'Saint Helena', 'U7': 'Saint Kitts and Nevis', 'U9': 'Saint Lucia', 'Z1': 'Saint Martin', 'V0': 'Saint Pierre and Miquelon', 'V1': 'Saint Vincent and the Grenadines', 'Y0': 'Samoa', 'S8': 'San Marino', 'S9': 'Sao Tome and Principe', 'T0': 'Saudi Arabia', 'T1': 'Senegal', 'Z2': 'Serbia', 'T2': 'Seychelles', 'T8': 'Sierra Leone', 'U0': 'Singapore', '2B': 'Slovakia', '2A': 'Slovenia', 'D7': 'Solomon Islands', 'U1': 'Somalia', 'T3': 'South Africa', '1L': 'South Georgia and the South Sandwich Islands', 'U3': 'Spain', 'F1': 'Sri Lanka', 'V2': 'Sudan', 'V3': 'Suriname', 'L9': 'Svalbard and Jan Mayen', 'V6': 'Swaziland', 'V7': 'Sweden', 'V8': 'Switzerland', 'V9': 'Syrian Arab Republic', 'F5': 'Taiwan, Province of China', '2D': 'Tajikistan', 'W0': 'Tanzania, United Republic of', 'W1': 'Thailand', 'Z3': 'Timor-leste', 'W2': 'Togo', 'W3': 'Tokelau', 'W4': 'Tonga', 'W5': 'Trinidad and Tobago', 'W6': 'Tunisia', 'W8': 'Turkey', '2E': 'Turkmenistan', 'W7': 'Turks and Caicos Islands', '2G': 'Tuvalu', 'W9': 'Uganda', '2H': 'Ukraine', 'C0': 'United Arab Emirates', 'X0': 'United Kingdom', '2J': 'United States Minor Outlying Islands', 'X3': 'Uruguay', '2K': 'Uzbekistan', '2L': 'Vanuatu', 'X5': 'Venezuela', 'Q1': 'Viet Nam', 'D8': 'Virgin Islands, British', 'VI': 'Virgin Islands, U.s.', 'X8': 'Wallis and Futuna', 'U5': 'Western Sahara', 'T7': 'Yemen', 'Y4': 'Zambia', 'Y5': 'Zimbabwe', 'XX': 'Unknown'}

def checked_symbol(entity_list):
    symbol = entity_list[1].strip(")").strip().replace(")", "") if len(entity_list) > 1 else ""
    if "CIK " in symbol:
        symbol = ""
    return symbol


def remove_front_zeros(s):
    return s.lstrip('0')

def id_correction(id):
    id = id.split(":")
    first_part = id[0].replace("-", "")
    second_part = id[1]
    return f"{first_part}/{second_part}"

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
        "User-Agent": "sec-scraper/1.0 (contact: abhishek2005.siva@.com)",
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
            # exponential backoff with jitter
            sleep_for = (2 ** (attempt - 1)) * 0.5 + random.uniform(0, 0.5)
            time.sleep(min(sleep_for, 10))
            continue

        status = resp.status_code
        # Retry on server errors / rate limit
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

        # Non-200 but not retriable => bail
        if status != 200:
            snippet = ""
            try:
                snippet = resp.text[:200].replace("\n", " ")
            except:
                pass
            print(f"[extarct_data] non-retriable status {status} body={snippet}")
            return []

        # Try to parse JSON; if parse fails, retry a few times
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

        # Debug: log reported total for first page
        total_reported = data.get("hits", {}).get("total", {}).get("value", 0)
        print(f"[extarct_data] page={page_int} from={from_int} size={size} reported_total={total_reported} attempt={attempt}")

        # If first page reports total==0, do a short retry (transient)
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

            # Determine file size (human-readable)
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
                "Filing entity located": targte_data.get("biz_locations",[None])[0] if targte_data.get("biz_locations") else None,
                "Filing entity incorporated": all_locations.get(targte_data.get("inc_states",[None])[0]) if targte_data.get("inc_states") else None,
                "Filer name": targte_data.get("display_names", [""])[-1] if len(targte_data.get("display_names", [])) > 1 else None,
                "Filer located": targte_data.get("biz_locations", [None])[-1] if len(targte_data.get("biz_locations", [])) > 1 else None,
                "Filer incorporated": all_locations.get(targte_data.get("inc_states", [None])[-1]) if len(targte_data.get("inc_states", [])) > 1 else None,
                "Symbol": symbol,
                "CIK": cik
            })
        return master_data


def get_spreadsheet_id(url):
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url or "")
    return m.group(1) if m else None

def col_to_letter(n):
    # 1 -> A, 2 -> B, ..., 26 -> Z, 27 -> AA
    result = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result.append(chr(65 + rem))
    return ''.join(reversed(result))

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
        if r.status_code >= 200 and r.status_code < 300:
            cl = r.headers.get('Content-Length')
            if cl:
                return int(cl)
        # Fallback: try GET without downloading body fully
        r = session.get(url, headers=headers, timeout=20, stream=True)
        if r.status_code >= 200 and r.status_code < 300:
            cl = r.headers.get('Content-Length')
            if cl:
                return int(cl)
    except Exception as e:
        try:
            print(f"[size] error fetching size for {url}: {e}")
        except Exception:
            pass
    return None

def write_df_to_sheet(gc, spreadsheet_id, worksheet_title, df):
    sh = gc.open_by_key(spreadsheet_id)
    try:
        wks = sh.worksheet(worksheet_title)
    except Exception:
        # Auto-create missing worksheet
        try:
            sh.add_worksheet(title=worksheet_title, rows=1000, cols=max(len(df.columns), 26))
            wks = sh.worksheet(worksheet_title)
        except Exception:
            wks = sh.sheet1

    headers = list(df.columns)
    rows = df.astype(object).where(pd.notnull(df), "").values.tolist()

    try:
        existing = wks.get_all_values()
    except Exception:
        existing = []

    need_headers = len(existing) == 0
    try:
        # If empty sheet, write headers explicitly to A1:Z1... based on column count
        if need_headers:
            end_col = col_to_letter(len(headers))
            header_range = f"A1:{end_col}1"
            wks.update(header_range, [headers], value_input_option='USER_ENTERED')

        # Always start data at column A on the next available row
        if rows:
            start_row = 2 if need_headers else (len(existing) + 1)
            wks.update(f"A{start_row}", rows, value_input_option='USER_ENTERED')
    except Exception:
        # Fallback: attempt batch appends one-by-one starting at column A
        if need_headers:
            try:
                end_col = col_to_letter(len(headers))
                header_range = f"A1:{end_col}1"
                wks.update(header_range, [headers], value_input_option='USER_ENTERED')
            except Exception:
                pass
        if rows:
            for idx, r in enumerate(rows, start=(2 if need_headers else (len(existing) + 1))):
                try:
                    wks.update(f"A{idx}", [r], value_input_option='USER_ENTERED')
                except Exception:
                    pass

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


def main():
    st.set_page_config(page_title="SEC Filings Scraper", layout="wide", page_icon="📊")
    st.title("📊 SEC Filings Scraper (Fast API)")
    st.markdown("Use the official SEC API for fast, paginated keyword search.")

    with st.sidebar:
        st.header("🔍 Search Parameters")
        doc_search = st.text_input("Document word or phrase:", help="Search for specific text within filings")
        entity_search = st.text_input("Company name, ticker, or CIK:", help="Filter by specific company or individual")
        st.subheader("📅 Date Range")
        date_range = st.date_input("Select Date Range", [], key="date_range")
        from_date, to_date = (date_range if len(date_range) == 2 else (datetime.now().date() - timedelta(days=1), datetime.now().date()))
        st.subheader("📄 Filing Types & Groups")
        COMMON_FILING_TYPES = ['10-K', '10-Q', '8-K', 'DEF 14A', 'NT 10-K', 'NT 10-Q', '20-F']
        CUSTOM_FILING_GROUPS = {
            "Riskalert-General (custom)": ["10-K", "10-Q", "8-K", "DEF 14A", "NT 10-K", "NT 10-Q", "PRE 14A", "S-1", "SC 13D", "SCHEDULE 13D"],
            "Riskalert-DEF14 (custom)": ["DEF 14A", "PRE 14A"],
            "Non-mgt (custom)": ["DEFC14A", "DEFC14C", "DEFN14A", "DEFR14A", "DFAN14A", "DFRN14A", "PREC14A", "PREC14C", "PREN14A", "PRER14A", "PRRN14A"],
            "13D (custom)": ["SC 13D", "SCHEDULE 13D"],
            "13G (custom)": ["SC 13G", "SCHEDULE 13G"]
        }
        selected_groups = []
        for group in CUSTOM_FILING_GROUPS:
            if st.checkbox(group, key=f"group_{group}"):
                selected_groups.append(group)
        selected_types = []
        for ftype in COMMON_FILING_TYPES:
            if st.checkbox(ftype, key=f"type_{ftype}"):
                selected_types.append(ftype)
        # Combine all selected types
        filing_types_list = []
        for group in selected_groups:
            filing_types_list.extend(CUSTOM_FILING_GROUPS[group])
        filing_types_list.extend(selected_types)
        # Also allow manual entry
        custom_types = st.text_input("Custom filing type(s):", help="Example: '10-K, 10-Q, 8-K'")
        if custom_types:
            filing_types_list.extend([x.strip().upper() for x in custom_types.split(',') if x.strip()])

        # If nothing selected, fall back to common defaults so Fetch works
        if not filing_types_list:
            filing_types_list = ['10-K', '10-Q', '8-K']
            st.caption("No filing types selected; using defaults: 10-K, 10-Q, 8-K")

        st.subheader("📄 Spreadsheet Connection")
        service_json_file = st.file_uploader("Google service account JSON", type=["json"], help="Upload service account credentials JSON for Sheets access")
        sheet_url = st.text_input("Spreadsheet URL:", help="Paste Google Sheets URL (https://docs.google.com/spreadsheets/d/...) to connect")
        connect_btn = st.button("🔗 Test Spreadsheet Connection")

        if connect_btn:
            if gspread is None or Credentials is None:
                st.error("Missing dependencies: install 'gspread' and 'google-auth'.")
            elif not service_json_file or not sheet_url:
                st.error("Please provide both the JSON file and the spreadsheet URL.")
            else:
                try:
                    creds_info = json.load(service_json_file)
                    scopes = ['https://www.googleapis.com/auth/spreadsheets']
                    credentials = Credentials.from_service_account_info(creds_info, scopes=scopes)
                    gc = gspread.authorize(credentials)
                    ss_id = get_spreadsheet_id(sheet_url)
                    if not ss_id:
                        st.error("Invalid spreadsheet URL format.")
                    else:
                        sh = gc.open_by_key(ss_id)
                        st.success(f"Connected to spreadsheet: {sh.title}")
                        st.session_state['gspread_client'] = gc
                        st.session_state['spreadsheet_id'] = ss_id
                        st.session_state['spreadsheet_title'] = sh.title
                        st.session_state['worksheet_names'] = [ws.title for ws in sh.worksheets()]
                except Exception as e:
                    st.error(f"Connection failed: {e}")

        if 'worksheet_names' in st.session_state and st.session_state.get('worksheet_names'):
            st.selectbox("Worksheet:", st.session_state['worksheet_names'], key="selected_worksheet")
            st.caption("The selected worksheet will be used for future exports.")
        run_button = st.button("🔎 Fetch Filings", type="primary")

    if run_button:
        with st.spinner("Fetching SEC Filings..."):
            all_filings = []
            category = "custom"
            date_range_str = "custom"
            startdt = from_date.strftime("%Y-%m-%d")
            enddt = to_date.strftime("%Y-%m-%d")
            forms_str = ",".join(filing_types_list)
            entityName = entity_search
            page = 1
            FILINGS_PER_PAGE = 100  # use a consistent page size
            while True:
                from_ = (page - 1) * FILINGS_PER_PAGE
                page_str = str(page)
                data = extarct_data(doc_search, date_range_str, category, startdt, enddt, forms_str, page_str, from_, entityName, size=FILINGS_PER_PAGE)
                if not data:
                    break
                all_filings.extend(data)
                st.write(f"Fetched page {page}, total records: {len(all_filings)}")
                # If fewer results than requested page size, we've reached the end
                if len(data) < FILINGS_PER_PAGE:
                    break
                page += 1
                # small pause to reduce rate-limit errors
                time.sleep(0.2)
            df = pd.DataFrame(all_filings)
            if not df.empty:
                # Insert Date and Time as first two columns
                export_time = datetime.now().strftime("%H:%M")
                date_col = df.get("Filed")
                formatted_dates = date_col.apply(format_date_str) if date_col is not None else pd.Series([format_date_str(None)] * len(df))
                df.insert(0, "Date", formatted_dates)
                df.insert(1, "Time", export_time)
            if df.empty:
                st.warning("No filings found for the selected criteria.")
            else:
                st.success(f"✅ Found {len(df)} filings matching your search.")
                st.dataframe(df, use_container_width=True)
            if not df.empty:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"SEC_filings_{from_date}_to_{to_date}.csv",
                    mime='text/csv'
                )

            # Always write to Google Sheet when connected
            connected = (
                gspread is not None and Credentials is not None and
                st.session_state.get('gspread_client') and
                st.session_state.get('spreadsheet_id') and
                st.session_state.get('selected_worksheet')
            )
            if connected:
                try:
                    if df.empty:
                        # Header-only write to initialize a new tab
                        HEADERS = [
                            "Date","Time","Form","File","File description","URL","Filing Detail","size",
                            "Filed","Reporting For","Filing entity","Filing entity located","Filing entity incorporated",
                            "Filer name","Filer located","Filer incorporated","Symbol","CIK"
                        ]
                        empty_df = pd.DataFrame(columns=HEADERS)
                        write_df_to_sheet(
                            st.session_state['gspread_client'],
                            st.session_state['spreadsheet_id'],
                            st.session_state['selected_worksheet'],
                            empty_df
                        )
                        st.info("Headers initialized in the selected worksheet.")
                    else:
                        write_df_to_sheet(
                            st.session_state['gspread_client'],
                            st.session_state['spreadsheet_id'],
                            st.session_state['selected_worksheet'],
                            df
                        )
                        st.success("✅ Results written to the selected worksheet.")
                except Exception as e:
                    st.error(f"Failed to write to spreadsheet: {e}")

if __name__ == "__main__":
    main()
