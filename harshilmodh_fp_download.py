# Python 3.11+
# Stevens Username: harshilmodh
#
# Phase 2a — Download SEC EDGAR filings for AAPL, MSFT, GOOGL.
#
# Downloads 1 × 10-K and 3 × 10-Q per company, covering FY2024 filings.
# Files are saved to data/filings/sec-edgar-filings/<TICKER>/<FORM>/<accession>/
#
# Idempotent: already-downloaded companies/form-types are skipped.
# SEC ToS requires a User-Agent with company name + email — config provides both.
#
# Run once before harshilmodh_fp_build_kb.py:
#   python harshilmodh_fp_download.py

import time
from pathlib import Path

from sec_edgar_downloader import Downloader
from tqdm import tqdm

from harshilmodh_fp_config import (
    COMPANIES,
    DATA_DIR,
    FILING_AFTER,
    FILING_BEFORE,
    FILINGS_PER_TYPE,
    FILING_TYPES,
    SEC_COMPANY_NAME,
    SEC_EMAIL,
)

FILINGS_ROOT = DATA_DIR / "sec-edgar-filings"


def _already_downloaded(ticker: str, form_type: str) -> bool:
    target_dir = FILINGS_ROOT / ticker / form_type
    if not target_dir.exists():
        return False
    existing = [d for d in target_dir.iterdir() if d.is_dir()]
    return len(existing) >= FILINGS_PER_TYPE[form_type]


def download_all() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dl = Downloader(SEC_COMPANY_NAME, SEC_EMAIL, str(DATA_DIR))

    tasks = [
        (ticker, form_type)
        for ticker in COMPANIES
        for form_type in FILING_TYPES
    ]

    print(f"\nFinancial Earnings Call Analyzer — SEC EDGAR Downloader")
    print(f"Corpus: {list(COMPANIES.keys())}  |  Forms: {FILING_TYPES}")
    print(f"Date range: {FILING_AFTER} → {FILING_BEFORE}\n")

    for ticker, form_type in tqdm(tasks, desc="Downloading", unit="filing-set"):
        limit = FILINGS_PER_TYPE[form_type]
        company_name = COMPANIES[ticker]

        if _already_downloaded(ticker, form_type):
            tqdm.write(f"  SKIP  {ticker} {form_type:5s} — already have {limit} filing(s)")
            continue

        tqdm.write(f"  GET   {ticker} {form_type:5s} ({company_name}, limit={limit})")
        try:
            dl.get(
                form_type,
                ticker,
                limit=limit,
                after=FILING_AFTER,
                before=FILING_BEFORE,
            )
        except Exception as exc:
            tqdm.write(f"  ERROR {ticker} {form_type}: {exc}")

        time.sleep(0.5)   # respect SEC EDGAR rate limits

    print("\nDownload complete. Verifying...")
    _report(FILINGS_ROOT)


def _report(root: Path) -> None:
    total = 0
    for ticker in COMPANIES:
        for form_type in FILING_TYPES:
            d = root / ticker / form_type
            count = len([x for x in d.iterdir() if x.is_dir()]) if d.exists() else 0
            status = "OK" if count >= FILINGS_PER_TYPE[form_type] else "MISSING"
            print(f"  [{status}] {ticker} {form_type:5s}: {count} accession(s)")
            total += count
    print(f"\nTotal accession directories found: {total}")


if __name__ == "__main__":
    download_all()
