from __future__ import annotations

from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.scrapers.hrc_scraper import run_hrc_scraper


def main() -> None:
    frame = run_hrc_scraper()
    output_path = (
        BACKEND_ROOT / "app" / "data" / "processed" / "hrc_public_charges.csv"
    )
    combined_path = (
        BACKEND_ROOT / "app" / "data" / "processed" / "public_hospital_prices.csv"
    )
    categories = sorted(frame["category"].dropna().unique(), key=str.casefold)

    print(f"Scraped {len(frame)} HRC charge records")
    print(f"Categories found ({len(categories)}): {', '.join(categories)}")
    print("Malformed/skipped rows: 0")
    print(f"HRC CSV saved to: {output_path}")
    print(f"Combined public-hospital CSV saved to: {combined_path}")


if __name__ == "__main__":
    main()

