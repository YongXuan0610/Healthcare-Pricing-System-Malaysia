# Public Pricing Data Audit

Audit date: 2026-08-25

## Dataset inspected

- Combined file: `Backend/app/data/processed/public_hospital_prices.csv`
- Manifest: `Backend/app/data/processed/public_pricing_manifest.json`
- Normalized records: 1,090
- Normalized categories: 21
- Source types: `hospital`, `national_reference`

| Source | Records | Categories | Refresh status |
|---|---:|---:|---|
| Hospital Kuala Lumpur (HKL) | 15 | 1 | Migrated from the existing validated HKL dataset |
| Hospital Rehabilitasi Cheras (HRC) | 30 | 9 | Live official page retrieved |
| Hospital Cyberjaya (HCJ) | 35 | 4 | Live official page retrieved |
| Hospital Sungai Buloh (HSB) | 61 | 8 | Six linked official pricing pages retrieved |
| Hospital Pakar Sultanah Fatimah (HPSF) | 152 | 10 | Preserved official-source snapshot used after live HTTP 500 |
| MOH Caj Wad | 18 | 1 | Official page snapshot used after automated HTTP 403 |
| MOH Caj Rawatan | 779 | 9 | Official page snapshot used after automated HTTP 403 |

## Validation results

- All rows have a stable source-scoped ID, numeric non-negative `price_rm`, original price text, and an HTTPS official source URL.
- No duplicate IDs or cross-source ID collisions remain.
- Free/`Percuma` values are retained as RM0 only when the original source explicitly describes the charge as free.
- Hospital and MOH records are separated by `source_type`; MOH records have no hospital code/name and use state `National`.
- Exact duplicates removed during source parsing: HCJ 2; MOH Caj Rawatan 7. Other new-source parsers removed 0.
- HCJ skipped 2 non-price/malformed cells. HRC, HSB, HPSF, and the MOH snapshot parsers skipped 0 malformed price cells.

## Legacy HKL quality constraint

The existing flattened HKL file contains 73 rows. Fifteen are trustworthy positive-price rows with patient class `all`; 58 are excluded from the normalized combined dataset because they are non-price section labels, zero placeholders without explicit free wording, or service names incorrectly flattened into the patient-class field. The HKL prices and runtime behaviour for the 15 validated ward records are preserved.

## Source-specific findings

### HRC

Rowspan/colspan expansion is required for foreign ward prices, free Class 3 prices, specialist visits, and the multi-service clinical table. The current official page reports `Kemaskini: 25/08/2026`.

### HCJ

The official `www` hostname currently presents a certificate for the non-`www` JKN Selangor hostname. The scraper uses the canonical non-`www` URL. A separate older official HCJ ward page publishes values that differ from the requested current consolidated charge page; the normalized dataset uses only the requested consolidated page and does not merge the older schedule.

### HSB

The charge hub contains no prices itself. The scraper follows only six linked official charge pages: specialist clinic, deposits, ward/treatment, delivery, medical reports, and emergency. Published medical-report ranges are represented as explicit lower/upper records rather than collapsed into a fabricated single price.

### HPSF

The live WordPress/Elementor page returned HTTP 500 during refresh. Its official WordPress metadata remained accessible and reported modification on 2024-04-17. The scraper attempts the live page first, then uses the committed structured snapshot recovered from the same official government page. The snapshot limitation is explicit in the manifest and refresh output.

### MOH

The MOH pages returned Akamai HTTP 403 to normal automated requests. Their official rendered tables were captured through a browser and committed as compact HTML snapshots. The scraper still attempts direct HTTP first. MOH data is citizen-oriented and is not substituted into a Non-Malaysian query.

## Statistical interpretation

Statistics are calculated only within an individual hospital's matching records. Values are never pooled across hospitals into a market median because hospitals may reproduce the same national fee schedule. MOH schedules are presented separately as national references and are not included in hospital statistics.

## Remaining limitations

- HKL remains limited to the 15 trustworthy ward rows from its legacy flattened dataset.
- HPSF and MOH refreshes depend on preserved official-source snapshots while their live pages remain unavailable to normal HTTP clients.
- Similar labels across sources are categorized consistently but are not automatically declared equivalent at service level.
- Published charges can change and are reference values, not guaranteed final bills.
