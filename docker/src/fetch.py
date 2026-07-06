#!/usr/bin/env python3
"""
Download PMC full‑text XMLs via AWS S3, parse them on the fly with quality
checks, and store the structured articles in a SQLite database.

The script reads a YAML filter configuration (db_config.yml) to build a PubMed
query, interacts with the user to choose a download mode (all / subset), fetches
the PMCIDs using NCBI Entrez, streams the XMLs directly from the S3 open‑access
bucket, and passes each article through a strict extraction and validation
pipeline. Valid papers are upserted into /data/database/database.db.

Requires:
    config.py (ARTICLE_COLUMNS, DTD_DIR, …)
    schema.py (ARTICLE_COLUMNS)
    db_config.yml (filters and download preferences)
"""

print("Download starting...\n")
import os
import sys
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
from Bio import Entrez
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm
import pandas as pd
import lxml.etree as ET
from pathlib import Path
import sqlite3
from schema import ARTICLE_COLUMNS

# =============================================================================
# PATHS AND DEFAULTS FROM DOCKER/CONFIG
# =============================================================================
CONFIG_PATH = os.environ.get("DB_CONFIG_PATH", "/app/db_config.yml")
OUTPUT_DIR  = "/data/database"
DTD_DIR     = Path("/dtd")
OUTPUT_FILENAME = 'database'
NCBI_EMAIL  = os.environ.get("NCBI_EMAIL", None)
DB_PATH = Path(OUTPUT_DIR) / f"{OUTPUT_FILENAME}.db"
CSV_PATH        = Path(OUTPUT_DIR) / f"{OUTPUT_FILENAME}.csv"
ERROR_CSV_PATH  = Path(OUTPUT_DIR) / f"extraction_errors_{OUTPUT_FILENAME}.csv"

# =============================================================================
# DTD SETUP
# =============================================================================
if not DTD_DIR.exists():
    raise FileNotFoundError(f"DTD directory not found: {DTD_DIR}")
BASE_URL = DTD_DIR.as_uri() + "/"
_parser = ET.XMLParser(dtd_validation=False, load_dtd=True,
                       recover=True, no_network=True, huge_tree=True)

# =============================================================================
# TEXT CLEANING
# =============================================================================
RE_BRACKETS_SQ = re.compile(r'\[[\s,;\d-]*\]')
RE_BRACKETS_RD = re.compile(r'\([\s,;\d-]*\)')
RE_WHITESPACE  = re.compile(r'\s{2,}')

def clean_text_formatting(text: str) -> str:
    """Remove citation brackets and normalize whitespace."""
    if not text:
        return ""
    text = RE_BRACKETS_SQ.sub('', text)
    text = RE_BRACKETS_RD.sub('', text)
    text = RE_WHITESPACE.sub(' ', text)
    return text.strip()

# =============================================================================
# SECTION MAPPING
# =============================================================================
SECTION_MAP = {
    'introduction': ['introduction', 'background', 'context', 'rationale'],
    'methods':      ['method', 'material', 'implementation',
                     'experimental', 'procedure', 'methodology'],
    'results':      ['result', 'finding', 'observation', 'data analysis'],
    'discussion':   ['discussion', 'interpretation', 'comment'],
    'conclusion':   ['conclusion', 'summary', 'concluding', 'closing', 'outlook'],
}
NOISE_TITLES = {
    'conflict', 'funding', 'acknowledgment',
    'contribution', 'ethics', 'data availability'
}

def _extract_journal(root) -> str:
    paths = [
        '//journal-title-group/journal-title/text()',
        '//journal-title/text()',
        '//journal-id[@journal-id-type="nlm-ta"]/text()',
        '//journal-id[@journal-id-type="publisher-id"]/text()',
    ]
    for path in paths:
        hits = root.xpath(path)
        if hits and hits[0].strip():
            return hits[0].strip()
    return ""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def process_xml_worker(xml_content: bytes, filename: str):
    """
    Parse one XML file and return structured article data.

    Returns
    -------
    (paper_dict, None)   on success
    (None, error_dict)   on failure
    """
    try:
        root = ET.fromstring(xml_content, parser=_parser, base_url=BASE_URL)

        # Strip citations and MathML globally
        ET.strip_elements(root, 'xref', with_tail=True)
        ET.strip_elements(root, '{http://www.w3.org/1998/Math/MathML}*', with_tail=True)

        # Filter: research-article + English only
        article_type = root.get('article-type')
        lang_list    = root.xpath('//article/@xml:lang | //article-meta//@xml:lang')
        lang         = lang_list[0] if lang_list else "en"

        if article_type not in ["research-article", "review-article"] or not str(lang).startswith('en'):
            return None, {"filename": filename, "error": f"not research or review article: {article_type}"}

        # Load structure
        paper = {col: "" for col in ARTICLE_COLUMNS}
        paper['filename'] = filename.split('/')[-1]

        # Metadata
        # Support both the legacy "pmc" and the newer "pmcid" attribute values
        pmcid_raw = "".join(root.xpath('//article-id[@pub-id-type="pmcid" or @pub-id-type="pmc"]/text()'))
        paper['pmcid'] = pmcid_raw.replace("PMC", "").strip() if pmcid_raw else ""

        #paper['pmcid']   = "".join(root.xpath('//article-id[@pub-id-type="pmc"]/text()'))
        paper['doi']     = "".join(root.xpath('//article-id[@pub-id-type="doi"]/text()'))
        paper['journal'] = _extract_journal(root)
        paper['year'] = "".join(root.xpath('//pub-date//year/text()')[:1])

        auth_list = [
            f'{"".join(a.xpath("./surname/text()"))}'
            f' {"".join(a.xpath("./given-names/text()"))}'.strip()
            for a in root.xpath('//contrib[@contrib-type="author"]//name')
        ]
        paper['authors'] = "; ".join(auth_list)

        # Find title element
        title_elem = root.find('.//article-meta/title-group/article-title')
        if title_elem is None:
            title_elem = root.find('.//front/title-group/article-title')

        if title_elem is not None:
            # Allowed tags for formatting (will be interpreted as text later)
            allowed_tags = {'italic', 'bold', 'sub', 'sup', 'i', 'b'}

            # Recursive function: Removes all elements whose tag is not in allowed_tags
            def clean_element(elem):
                # Iterate through children (reverse order to ensure safe deletion)
                for child in list(elem):
                    if child.tag not in allowed_tags:
                        # Removes the element but keeps its text and tail
                        elem.remove(child)
                        # Text of the deleted child is inserted before the next siblings
                        # (lxml handles the tail automatically when the element is removed)
                    else:
                        # Clean allowed tags recursively
                        clean_element(child)

            clean_element(title_elem)
            # Now fetch the pure text from the cleaned element
            title_text = " ".join(title_elem.itertext())
        else:
            title_text = ""

        paper['title'] = clean_text_formatting(title_text)

        # Keywords original
        kwd_xpath = (
            '//kwd//text() | '
            '//compound-kwd-part//text() | '
            '//nested-kwd//text()'
        )
        raw_kwds = root.xpath(kwd_xpath)
        unique_keywords = set(k.strip() for k in raw_kwds if k.strip() and len(k.strip()) > 1)
        paper['keywords'] = "; ".join(sorted(unique_keywords))

        # Abstract
        abstract_paras = root.xpath('//abstract[not(@abstract-type)]//p')
        if not abstract_paras:
            abstract_paras = root.xpath('//abstract//p')
        paper['abstract'] = clean_text_formatting(
            ' '.join([' '.join(p.itertext()) for p in abstract_paras]))

        # Tables in Markdown
        tables = []
        for table_wrap in root.xpath('//table-wrap'):
            try:
                rows_data = []
                # Search all rows in table body or header area
                rows = table_wrap.xpath('.//tr | .//thead/tr | .//tbody/tr')
                for row in rows:
                    row_cells = []
                    # Find all th and td cells in exact order
                    cells = row.xpath('.//th | .//td')
                    for cell in cells:
                        # Get all text, even if nested inside tags
                        cell_text = ' '.join(cell.xpath('.//text()')).strip()
                        # Prevent line breaks or pipes from breaking the Markdown layout
                        cell_text = cell_text.replace('\n', ' ').replace('|', '\\|')
                        row_cells.append(clean_text_formatting(cell_text))
                    if row_cells:
                        rows_data.append(row_cells)

                if rows_data:
                    markdown_lines = []
                    for i, row in enumerate(rows_data):
                        markdown_row = '| ' + ' | '.join(row) + ' |'
                        markdown_lines.append(markdown_row)
                        # Insert Markdown separator after the first row (header)
                        if i == 0:
                            separator = '|' + '|'.join([' --- ' for _ in row]) + '|'
                            markdown_lines.append(separator)

                    table_markdown = '\n'.join(markdown_lines)
                    caption = table_wrap.xpath('.//caption//text()')
                    if caption:
                        caption_text = clean_text_formatting(' '.join(caption))
                        table_markdown = f"**{caption_text}**\n\n{table_markdown}"
                    tables.append(table_markdown)
            except Exception:
                continue
        paper['tables'] = tables

        # Sections
        for sec in root.xpath('//body/sec'):
            sec_title = " ".join(sec.xpath('./title/text()')).lower()
            sec_text  = clean_text_formatting(
                " ".join([" ".join(p.itertext()) for p in sec.xpath('.//p')]))

            if any(noise in sec_title for noise in NOISE_TITLES):
                paper['metadata_junk'] += f"\n[{sec_title}]: {sec_text}"
                continue

            assigned = False
            for target, keywords in SECTION_MAP.items():
                if any(k in sec_title for k in keywords):
                    paper[target] = (paper[target] + " " + sec_text).strip()
                    assigned = True
            if not assigned:
                paper['metadata_junk'] += f"\n[unmapped: {sec_title}]: {sec_text}"

        # Quality check for RAG
        if not paper['abstract'] or len(paper['abstract']) < 100:
            return None, {"filename": filename, "error": "abstract missing/too short"}

        # Check if any significant full-text content could be extracted
        total_text_length = sum(len(paper[sec]) for sec in ['introduction', 'methods', 'results', 'discussion', 'conclusion'])
        if total_text_length < 500:
            return None, {"filename": filename, "error": "fulltext content too short (<500 chars)"}

        return paper, None

    except Exception as e:
        # Catch the exact error at line level inside the worker
        import traceback
        error_stack = traceback.format_exc()
        return None, {
            "filename": filename,
            "error": f"WORKER_CRASHED: {str(e)} | Stack: {error_stack[:150]}"
        }


def build_boolean_subquery(criterion_dict, field_tag):
    """
    Build an arbitrarily flexible AND/OR/NOT group query for a given field.

    Supports 'must_contain' (required terms), 'or_groups' (at least one term
    from each group must match), and 'must_not_contain' (terms to exclude).
    """
    if not criterion_dict or not isinstance(criterion_dict, dict):
        return ""

    and_parts = []

    # 1. MUST (required terms)
    must_list = criterion_dict.get("must_contain", [])
    if must_list:
        must_str = " AND ".join([f'"{item}"{field_tag}' for item in must_list if item])
        if must_str:
            and_parts.append(f"({must_str})")

    # 2. Flexible OR groups (one from each group must match)
    or_groups = criterion_dict.get("or_groups", [])
    if or_groups:
        for group in or_groups:
            if isinstance(group, list) and group:
                group_str = " OR ".join([f'"{item}"{field_tag}' for item in group if item])
                if group_str:
                    and_parts.append(f"({group_str})")

    # Join positive conditions with AND
    subquery = " AND ".join(and_parts)
    if len(and_parts) > 1:
        subquery = f"({subquery})"

    # 3. MUST NOT (exclusion)
    not_list = criterion_dict.get("must_not_contain", [])
    if not_list:
        not_str = " OR ".join([f'"{item}"{field_tag}' for item in not_list if item])
        if not_str:
            if subquery:
                subquery = f"{subquery} NOT ({not_str})"
            else:
                subquery = f"NOT ({not_str})"

    return subquery


def prompt(prompt_str, choices):
    while True:
        val = input(prompt_str).strip().lower()
        if val in choices:
            return val
        print(f"Please enter one of: {', '.join(choices)}")


# =============================================================================
# PREPARE DOWNLOAD
# =============================================================================

# Ensure output directory exists (it’s a mounted volume, so files appear on host)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. LOAD FILTER CRITERIA FROM YAML
# =============================================================================
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

sel = config["data_selection"]

# =============================================================================
# 2. HANDLE NCBI EMAIL
# =============================================================================
if not NCBI_EMAIL:
    if sys.stdin.isatty():
        NCBI_EMAIL = input("Please enter your NCBI email address: ").strip()
    else:
        raise RuntimeError("NCBI_EMAIL environment variable not set and no TTY available.")
Entrez.email = NCBI_EMAIL



# =============================================================================
# 3. BUILD THE PubMed QUERY
# =============================================================================
print("Generating PubMed query from db_config.yml...")
query_parts = []

if "keywords" in sel:
    kw_subquery = build_boolean_subquery(sel["keywords"], "[TIAB]")
    if kw_subquery: query_parts.append(kw_subquery)
print(".")

if sel.get("time_period"):
    query_parts.append(f"{sel['time_period']}[DP]")
print(".")    

if "journals" in sel:
    journal_subquery = build_boolean_subquery(sel["journals"], "[TA]")
    if journal_subquery: query_parts.append(journal_subquery)
print(".")

if "mesh_subjects" in sel:
    mesh_subquery = build_boolean_subquery(sel["mesh_subjects"], "[MH]")
    if mesh_subquery: query_parts.append(mesh_subquery)
print(".")

# Fallback: if no filter is defined, query everything
if not query_parts:
    print("Warning: No filters defined. Querying the entire PMC …")
    final_query = "all[sb]"
else:
    final_query = " AND ".join(query_parts)
print(".")


# =============================================================================
# 4. QUICK PRE‑CHECK OF TOTAL HITS
# =============================================================================
handle = Entrez.esearch(db="pmc", term=final_query, retmax=0)
search_results = Entrez.read(handle)
handle.close()
print(".")
total_count = int(search_results["Count"])
print(f"...Matching papers: {total_count}\n")


#print(f"...found {total_count} papers in PMC matching the advanced criteria.")

# =============================================================================
# 5. INTERACTIVE SUBSET SELECTION
# =============================================================================
if sys.stdin.isatty():
    print("\nDepending on your PC, espacially GPU availability, download and processing of \nthe papers might take a while. For testing or slow computers define a subset.")
    print("\nDo you want to download:")
    mode = prompt("   - (a)ll papers\n   - (s)ubset of papers\n \nPress [a/s] accordingly: ", ["a", "s"])
    if mode == "s":
        print("------------------------------------------------------------------------------")
        num = int(input("\nEnter subset size: "))
        print("------------------------------------------------------------------------------")
        subset_type = prompt("\nWich papers do you want:\n   - (l)atest papers\n   - (r)andom papers\n \nPress [l/r] accordingly: ", ["l", "r"])
        print("------------------------------------------------------------------------------")
    else:
        num = total_count
        subset_type = None
        print("------------------------------------------------------------------------------")
else:
    mode = os.environ.get("FETCH_MODE", "a").lower()
    num = int(os.environ.get("FETCH_NUM", total_count))
    subset_type = os.environ.get("FETCH_SUBSET_TYPE", "l").lower()
    print(f"\nNon‑interactive mode: mode={mode}, num={num}, type={subset_type}")
    print("------------------------------------------------------------------------------")

# Cap the requested number to the total available
if num > total_count:
    num = total_count

# Prompt whether to include the 69 golden test papers
if sys.stdin.isatty():
    print("\nYou can evaluate chunk retrieval and answer generation of the RAG system by \nadding 69 evaluation papers for which defined questions exist.\n")
    ()
    include_golden = prompt("Do you want to add those papers to evaluate the database? (y/n): ", ["y", "n"])
else:
    include_golden = os.environ.get("INCLUDE_GOLDEN", "y").lower()
# Remember decision (for setup.sh)
os.makedirs(OUTPUT_DIR, exist_ok=True)
flag_path = Path(OUTPUT_DIR) / "include_golden.flag"
with open(flag_path, "w") as f:
    f.write("1" if include_golden == "y" else "0")
print("------------------------------------------------------------------------------")    



# =============================================================================
# 6. DYNAMIC ID RETRIEVAL (SMART SLICING FOR LARGE SETS)
# =============================================================================
all_ids = []

# Fast path: if the subset is below 10 000, we can fetch IDs directly
if num <= 9999 and mode == "s":
    # If 'latest' is desired, get IDs sorted by publication date
    sort_param = "pub_date" if subset_type == "l" else ""

    handle = Entrez.esearch(db="pmc", term=final_query, retmax=9999, sort=sort_param)
    slice_results = Entrez.read(handle)
    handle.close()

    all_ids = slice_results.get("IdList", [])

    if subset_type == "r":
        print("Shuffling local ID pool randomly …")
        random.shuffle(all_ids)
        print(".")
    pmcid_list = all_ids
    print(f"\n{num} of {len(pmcid_list)} papers will be downloaded...")


# Fallback: for large requests, use date‑based slicing to avoid the 9 999‑record limit
else:
    import datetime

    time_period = sel.get("time_period", "")
    if not time_period:
        start_year, end_year = 1970, datetime.datetime.now().year
    elif ":" in time_period:
        start_year, end_year = map(int, time_period.split(":"))
    else:
        start_year = end_year = int(time_period)

    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 12, 31)
    if end_date > datetime.date.today():
        end_date = datetime.date.today()

    current_start = start_date
    step_days = 30

    with tqdm(desc="Synchronizing global ID pool", unit=" IDs") as pbar:
        while current_start <= end_date:
            current_end = current_start + datetime.timedelta(days=step_days)
            if current_end > end_date: current_end = end_date

            date_fmt_start = current_start.strftime("%Y/%m/%d")
            date_fmt_end = current_end.strftime("%Y/%m/%d")
            slice_query = f"({final_query}) AND \"{date_fmt_start}\"[PDAT] : \"{date_fmt_end}\"[PDAT]"

            try:
                handle = Entrez.esearch(db="pmc", term=slice_query, retmax=0)
                search_results = Entrez.read(handle)
                handle.close()
                slice_count = int(search_results["Count"])

                if slice_count > 9999 and step_days > 3:
                    step_days = max(3, step_days // 4)
                    continue

                if slice_count > 0:
                    handle = Entrez.esearch(db="pmc", term=slice_query, retmax=9999)
                    slice_results = Entrez.read(handle)
                    handle.close()
                    all_ids.extend(slice_results.get("IdList", []))
                    pbar.update(len(slice_results.get("IdList", [])))

                current_start = current_end + datetime.timedelta(days=1)
                if slice_count < 2000 and step_days < 30:
                    step_days = min(30, step_days + 5)
                time.sleep(0.15)

            except Exception as slice_error:
                tqdm.write(f"\nAPI error in block {date_fmt_start} to {date_fmt_end}: {slice_error}")
                time.sleep(3)
                continue

    # Deduplicate and optionally shuffle/order
    all_ids = list(set(all_ids))
    if mode == "s" and subset_type == "r":
        random.shuffle(all_ids)

    pmcid_list = all_ids
    print(f"\n{num} of {len(pmcid_list)} papers will be downloaded...")

# =============================================================================
# 7. CONFIGURE S3 CLIENT
# =============================================================================
s3_config = Config(
    signature_version=UNSIGNED,
    connect_timeout=5,
    read_timeout=5,
    retries={"max_attempts": 2},
    max_pool_connections=128
)
s3_client = boto3.client("s3", region_name="eu-central-1", config=s3_config)
bucket_name = "pmc-oa-opendata"

def process_s3_paper_live(pmcid):
    highest_version = 0
    consecutive_misses = 0
    MAX_CONSECUTIVE_MISSES = 2
    MAX_VERSION = 10
    MAX_RETRIES = 5               # per version
    BASE_BACKOFF = 0.5             # seconds

    # 1. Find optimal version (optimistic, with backoff on throttling)
    for v in range(1, MAX_VERSION + 1):
        s3_key = f"PMC{pmcid}.{v}/PMC{pmcid}.{v}.xml"
        retries = MAX_RETRIES
        backoff = BASE_BACKOFF
        success = False

        while retries > 0:
            try:
                s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                highest_version = v
                consecutive_misses = 0
                success = True
                break                     # version found
            except ClientError as ce:
                code = ce.response['Error']['Code']
                if code == '404':
                    consecutive_misses += 1
                    if consecutive_misses >= MAX_CONSECUTIVE_MISSES:
                        # No further versions expected
                        break
                    # otherwise just try next version (no retry needed)
                    break
                elif code in ('503', '429', 'InternalError', 'SlowDown'):
                    # Temporary overload → wait and retry
                    retries -= 1
                    if retries == 0:
                        return None, {"filename": f"PMC{pmcid}", "error": f"Throttling limit exceeded: {ce}"}
                    sleep_time = backoff + random.uniform(0, 0.5)
                    time.sleep(sleep_time)
                    backoff *= 2.0
                else:
                    # Permanent error (403 etc.) → abort immediately
                    return None, {"filename": f"PMC{pmcid}", "error": f"S3 head error: {ce}"}
            except Exception as e:
                retries -= 1
                if retries == 0:
                    return None, {"filename": f"PMC{pmcid}", "error": f"Network timeout limit exceeded: {e}"}
                time.sleep(backoff + random.uniform(0, 0.3))
                backoff *= 2.0

        # If the inner loop was broken by a 404, break outer loop as well
        if consecutive_misses >= MAX_CONSECUTIVE_MISSES:
            break

    if highest_version == 0:
        return None, {"filename": f"PMC{pmcid}", "error": "No version found (all 1‑10 returned 404)"}

    # 2. Download XML of the highest version with the same robustness
    final_key = f"PMC{pmcid}.{highest_version}/PMC{pmcid}.{highest_version}.xml"
    retries = MAX_RETRIES
    backoff = BASE_BACKOFF
    while retries > 0:
        try:
            s3_object = s3_client.get_object(Bucket=bucket_name, Key=final_key)
            xml_content = s3_object['Body'].read()
            break
        except ClientError as ce:
            code = ce.response['Error']['Code']
            if code in ('503', '429', 'InternalError', 'SlowDown'):
                retries -= 1
                if retries == 0:
                    return None, {"filename": f"PMC{pmcid}", "error": f"Download throttling limit exceeded: {ce}"}
                time.sleep(backoff + random.uniform(0, 0.5))
                backoff *= 2.0
            else:
                return None, {"filename": f"PMC{pmcid}", "error": f"Download error: {ce}"}
        except Exception as e:
            retries -= 1
            if retries == 0:
                return None, {"filename": f"PMC{pmcid}", "error": f"Download network error limit exceeded: {e}"}
            time.sleep(backoff + random.uniform(0, 0.3))
            backoff *= 2.0

    res, err = process_xml_worker(xml_content, f"PMC{pmcid}.xml")
    return res, err

# =============================================================================
# 8. PARALLEL PROCESSING WITH LIVE SUBMISSION CONTROL
# =============================================================================
articles = []
errors = []

# Launch the thread pool
with ThreadPoolExecutor(max_workers=128) as executor:
    future_map = {}
    id_iterator = iter(pmcid_list)

    # Submit an initial batch of 128 tasks to saturate the pipeline
    for _ in range(min(128, len(pmcid_list))):
        try:
            pmcid = next(id_iterator)
            future = executor.submit(process_s3_paper_live, pmcid)
            future_map[future] = pmcid
        except StopIteration:
            break

    # Progress bar tracks actual successes (total=num)
    with tqdm(total=num, desc="Progress", ncols=70) as pbar:
        # Process results asynchronously as they arrive
        while future_map:
            # If the limit is reached, stop the loop
            if len(articles) >= num:
                #print(f"\n[LIMIT REACHED] Successfully secured {len(articles)} articles. Shutting down pool instantly.")
                tqdm.write(f"...{len(articles)} papers have been downloaded. Shutting down pool instantly.")
                # Cancel pending tasks to stop the pool immediately
                for f in future_map.keys():
                    f.cancel()
                break

            # Wait for the next completed task
            done = next(as_completed(future_map.keys()))
            current_pmcid = future_map.pop(done)
            pbar.update(0)  # don't blindly advance the bar

            try:
                result = done.result()
                if result is not None:
                    res, err = result
                    if err:
                        errors.append(err)
                    if res:
                        articles.append(res)
                        pbar.update(1)  # only count real successes
            except Exception as thread_exception:
                errors.append({"filename": f"PMC{current_pmcid}", "error": str(thread_exception)})

            # As long as the limit is not reached and IDs are still available,
            # immediately submit a new task to keep 128 threads active
            if len(articles) < num:
                try:
                    next_pmcid = next(id_iterator)
                    new_future = executor.submit(process_s3_paper_live, next_pmcid)
                    future_map[new_future] = next_pmcid
                except StopIteration:
                    pass  # ID pool exhausted

    # When the thread pool closes, check why it closed
    if len(articles) >= num:
        pass
        #tqdm.write(f"\nTarget subset of {num} articles successfully reached!")
    else:
        tqdm.write(f"\n[Pool exhausted]. Only {len(articles)} of {num} articles passed the quality check.")

# =============================================================================
# 9. SAVE RESULTS – SQLITE UPSERT (NO CSV EXPORT)
# =============================================================================

print("------------------------------------------------------------------------------\n")
print("DOWNLOAD SUMMARY:")
print(f"Total valid articles in this run : {len(articles)}")
print(f"Total processing errors/404      : {len(errors)}")
print("")

if articles:
    new_df = pd.DataFrame(articles)

    # Enforce column order
    new_df = new_df[ARTICLE_COLUMNS]

    # Convert tables list to string
    if 'tables' in new_df.columns:
        new_df['tables'] = new_df['tables'].apply(
            lambda x: " \n\n---NEW_TABLE---\n\n ".join(x) if isinstance(x, list) else str(x)
        )

    conn = sqlite3.connect(str(DB_PATH))
    col_defs = ", ".join([f'"{col}" TEXT' for col in ARTICLE_COLUMNS])
    conn.execute(f"CREATE TABLE IF NOT EXISTS articles ({col_defs}, PRIMARY KEY (pmcid))")

    # Get old total before upsert
    old_len = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    placeholders = ", ".join(["?"] * len(ARTICLE_COLUMNS))
    for row in new_df.itertuples(index=False, name=None):
        conn.execute(f"INSERT OR REPLACE INTO articles VALUES ({placeholders})", row)

    conn.commit()
    new_total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    conn.close()

    added = new_total - old_len
    updated = len(new_df) - added   # the rest were updated (replaced)

    tqdm.write("SQLite database update:")
    tqdm.write(f"   Articles before this run : {old_len}")
    tqdm.write(f"   Parsed in this run       : {len(new_df)}")
    tqdm.write(f"   ── newly added           : {added}")
    tqdm.write(f"   ── updated               : {updated}")
    tqdm.write(f"   Total in DB now          : {new_total}")
    tqdm.write(f"   Container path           : {DB_PATH}")
    host_dir = os.environ.get("HOST_OUTPUT_DIR", "Unknown host path")
    tqdm.write(f"   Host path                : {host_dir}/database.db")
else:
    tqdm.write("\nNo new articles met the quality criteria. Database unchanged.")

if errors:
    err_df = pd.DataFrame(errors)
    err_df.to_csv(ERROR_CSV_PATH, index=False)

print("")
print("")
