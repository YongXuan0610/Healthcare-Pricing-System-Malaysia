# HKL Public Pricing Data Quality Note

## Finding

`app/data/processed/hkl_public_charges.csv` contains 73 rows. Twenty-eight rows use a recognized `patient_class` value (`all`), while 45 rows contain values that are service names, ward classes, body areas, or procedures rather than citizenship classes. Examples include `speech-language assessment`, `head`, and `electrocardiography`.

The raw captured HKL page text represents several wide tables as flattened lines. In `app/scrapers/hkl_scraper.py`, `_extract_admission_table_rows` assigns the line after a row number to both `service_name` and `patient_class`. This explains the invalid classifications. The flattened text does not provide enough reliable structure to reconstruct every service/class/price relationship without assumptions.

## Safe handling

The source CSV and scraper output are not rewritten in this stabilization patch. `PublicPricingService` now uses only records with:

- a positive numeric price; and
- `patient_class` equal to `all`, `citizen`, or `foreigner`.

Questionable records and zero-price section headings remain preserved in the CSV but are not returned as public price estimates. If no trustworthy record matches a request, the API returns `estimate_available: false` and no fabricated fallback price.

The available source is a Hospital Kuala Lumpur reference. Results are explicitly marked as not adjusted by state. No state multiplier or citizenship-price mapping was invented.
