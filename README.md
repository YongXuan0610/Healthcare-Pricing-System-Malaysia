# Healthcare Pricing System for Public and Private Hospitals in Malaysia

A Final Year Project that brings several clearly separated healthcare-cost resources into one web application: Malaysian public-hospital charges, Malaysian private-hospital packages and ward rates, published LIAM private-hospital bill references, a deterministic healthcare-service assistant, and research machine-learning models.

The application is an educational decision-support system. It does not provide a quotation, diagnosis, clinical advice, insurance advice, or a guaranteed final bill.

## 1. Project Overview

The project combines a React single-page application with a FastAPI backend and Supabase authentication/data services. Malaysian pricing modules return source-linked reference values in MYR. The primary ML research module is deliberately isolated from those modules: it uses India's NSS 80th Round healthcare survey and predicts hospitalisation-related medical expenditure in INR.

## 2. Main Features

- Public-hospital charge search and comparison across selected Malaysian hospitals and Ministry of Health references
- Private-hospital package and ward-rate lookup backed by Supabase
- Published LIAM P25, typical, and P75 bill references
- Healthcare Service Assistant for deterministic service navigation and emergency-message handling
- Practical 15-input NSS expenditure research prediction with a calibrated range
- Secondary US medical-charge benchmark, kept separate from Malaysian and NSS outputs
- Supabase user authentication, profiles, prediction history, role checks, and administration screens
- Scraper, data-normalisation, model-training, evaluation, and automated-test source code

## 3. System Architecture

```text
React + Vite + TypeScript
        |
        | HTTP / Supabase client
        v
FastAPI application --------------------> Supabase Auth + PostgreSQL/Data API
        |
        +--> Malaysian public pricing CSVs and official-source snapshots (MYR)
        +--> Malaysian private pricing tables in Supabase (MYR)
        +--> LIAM published reference table (MYR)
        +--> NSS Practical 15 model and 80% nominal range artifact (INR)
        +--> US benchmark artifacts (USD)
```

The three country/data contexts are not merged or currency-relabeled. The API and user interface identify the role, currency, and limitations of each component.

## 4. Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React 18, Vite, TypeScript, Tailwind CSS, shadcn/Radix UI, TanStack Query, Vitest |
| Backend | FastAPI, Python, Uvicorn, Pydantic, pandas, NumPy |
| Data/Auth | Supabase Auth and PostgREST with Row Level Security |
| ML | scikit-learn, joblib, HistGradientBoostingRegressor |
| Testing | pytest, Vitest, Testing Library, ESLint |

## 5. Malaysian Public Hospital Pricing

The backend serves a normalized, source-linked dataset for selected Malaysian public hospitals and national Ministry of Health references. Runtime data is under `Backend/app/data/processed/`; compact official-source snapshots under `Backend/app/data/raw/` support provenance and reproducible scraper parsing when a live government page is unavailable or blocks automation.

The data is reference material only. Published schedules can change, similarly named services may not be clinically equivalent, and the final amount may depend on patient class, treatment details, supplies, and hospital policy.

See `Backend/reports/PUBLIC_PRICING_DATA_AUDIT.md` for source and normalization notes.

## 6. Private Hospital Pricing

Private package and ward data is read from the Supabase `private_packages` and `private_ward_rates` tables. The API preserves published prices and rate bases; it does not infer a guaranteed total bill. The admin interface can manage these records subject to the project's Supabase policies and role controls.

## 7. LIAM Pricing Reference

The LIAM component is a lookup over a published Malaysian reference table. It returns the published P25, typical, and P75 amounts in MYR when credible data is available and reports unavailable when the source marks a segment as having insufficient credible data. LIAM records are not used to train either ML model.

## 8. Healthcare Service Assistant

The assistant is a deterministic navigation helper, not a generative medical chatbot. It maps supported wording to available public-pricing categories, detects selected emergency phrases, and avoids making diagnoses or treatment recommendations.

## 9. NSS 80 Machine Learning Research Component

The NSS model is based on the India NSS 80th Round Survey on Household Social Consumption: Health, Schedule 25.0 (January-December 2025). It predicts hospitalisation-related total medical expenditure (Block 7, item 12) in INR per inpatient case.

The deployed model is the Practical 15 leakage-safe `HistGradientBoostingRegressor`. It uses 15 user-facing inputs, keeps preprocessing inside the serialized estimator, excludes known expenditure leakage fields, and uses a person-disjoint locked test split.

Locked-test metrics:

| Metric | Result |
|---|---:|
| R² | 0.39770 |
| MAE | INR 19,614.18 |
| RMSE | INR 57,639.14 |
| Median Absolute Error | INR 5,436.30 |

R² is a coefficient of determination, not an accuracy percentage. The user-facing module presents a nominal 80% expenditure range with 88.47% empirical locked-test episode coverage. This is marginal empirical coverage across comparable records, not an individual guarantee.

This result is an India NSS research estimate in INR. It is **not** a Malaysian hospital quotation and is not converted to MYR.

### NSS data availability

Raw NSS public-use CSVs and the generated inpatient modelling table are intentionally excluded from this repository. Obtain the data from the [official India Microdata Library catalog](https://microdata.gov.in/NADA/index.php/catalog/290), review its current terms, and place the extracted files as:

```text
Backend/datasets/nss80/raw/hhscsL1.csv
Backend/datasets/nss80/raw/hhscsL2.csv
...
Backend/datasets/nss80/raw/hhscsL7.csv
```

The expected provenance and hashes are documented in `Backend/datasets/nss80/source_manifest.json`. The running API does not require the raw survey files; they are needed only for retraining and full research reproduction.

## 10. Installation

Prerequisites:

- Python 3.11 or a compatible Python version
- Node.js 20+ and npm
- A Supabase project configured with appropriate tables and Row Level Security policies

Clone the repository, then configure the backend and frontend separately.

## 11. Backend Setup

From PowerShell:

```powershell
cd Backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `Backend/.env` with your own Supabase project settings. Never place a Supabase service-role key in this application repository.

The backend expects these Supabase tables to exist: `profiles`, `prediction_history`, `system_settings`, `public_charges`, `private_packages`, and `private_ward_rates`. Database schema migrations are not included in this repository, so reproduce the table structure and RLS policies in your own Supabase project before using all account and admin features.

## 12. Frontend Setup

```powershell
cd Frontend
npm ci
Copy-Item .env.example .env.local
```

Edit `Frontend/.env.local` with your own project values. Then run `npm run dev`. The Vite development server uses `http://localhost:8080`.

## 13. Environment Variables

Backend (`Backend/.env`):

| Variable | Purpose |
|---|---|
| `SUPABASE_URL` | Supabase project URL used by the FastAPI Data/Auth requests |
| `SUPABASE_PUBLISHABLE_KEY` | Publishable key used with Supabase RLS; do not use a service-role secret |
| `FRONTEND_ORIGINS` | Comma-separated CORS origins allowed by FastAPI |

Frontend (`Frontend/.env.local`):

| Variable | Purpose |
|---|---|
| `VITE_API_BASE_URL` | FastAPI base URL, normally `http://localhost:8000` |
| `VITE_SUPABASE_URL` | Supabase project URL |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | Browser-safe publishable key governed by RLS |

Only variable names and placeholders are committed. Create local environment files from the examples; `.env` and `.env.local` are ignored by Git.

## 14. Running the System

Start the backend from `Backend`:

```powershell
.\.venv\Scripts\uvicorn.exe scripts.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal from `Frontend`:

```powershell
npm run dev
```

Open `http://localhost:8080`. FastAPI documentation is available at `http://localhost:8000/docs` while the backend is running.

## 15. Machine Learning Model Information

The clean repository contains only the NSS artifacts required by the production API:

- `Backend/models/nss80/primary_model.joblib`
- `Backend/models/nss80/metadata.json`
- `Backend/models/nss80/practical_15_prediction_range/practical_15_interval_80.joblib`

The US benchmark remains isolated under `Backend/models/us_kaggle/` and returns USD. It is retained for comparison only; it is not a Malaysian estimator and is not combined with NSS records.

Historical model binaries, backup production versions, candidate quantile models, row-level predictions, and raw research datasets are excluded. Relevant training/evaluation source code and compact final reports remain available for academic review.

## 16. Testing

Backend checks from `Backend`:

```powershell
.\.venv\Scripts\python.exe -m compileall app model_training scripts tests
.\.venv\Scripts\python.exe -m pytest -q
```

Frontend checks from `Frontend`:

```powershell
npm test
npm run build
npm run lint
```

Tests that explicitly reproduce historical NSS experiments require the separately downloaded raw survey files and excluded research artifacts. Runtime, pricing, service-assistant, and UI tests can be run from the clean repository.

## 17. Project Limitations

- Malaysian prices are references and may become outdated or omit case-specific fees.
- The repository does not contain Supabase SQL migrations; a compatible schema and secure RLS policies must be provisioned separately.
- Live Supabase/authentication tests require a configured project and network access.
- The Practical 15 model explains only part of individual expenditure variation, especially for unusual and high-cost cases.
- The NSS range is not an individual coverage guarantee, and the locked test is finite historical survey data.
- The India NSS, Malaysia pricing, LIAM, and US benchmark components use different populations, meanings, and currencies and must not be directly conflated.

## 18. Disclaimer

This software is for education and research. It is not medical, financial, insurance, or legal advice. Results must not be used as a diagnosis, treatment decision, hospital quotation, or guarantee of expenditure. Confirm current charges and eligibility directly with the relevant hospital, insurer, or authority.

## 19. Author / Final Year Project Information

**Name:** Loh Yong Xuan  
**Programme:** Bachelor of Computer Science (Honours)  
**Institution:** Universiti Tunku Abdul Rahman (UTAR), Kampar Campus  
**Project:** Healthcare Pricing System for Public and Private Hospitals in Malaysia  
**Academic Year:** 2026