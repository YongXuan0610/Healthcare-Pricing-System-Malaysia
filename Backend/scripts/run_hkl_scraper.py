from __future__ import annotations

from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.scrapers.hkl_scraper import run_hkl_scraper


def main() -> None:
    frame = run_hkl_scraper()
    output_path = BACKEND_ROOT / "app" / "data" / "processed" / "hkl_public_charges.csv"
    print(f"Scraped {len(frame)} charge rows")
    print(f"Processed CSV saved to: {output_path}")


if __name__ == "__main__":
    main()
