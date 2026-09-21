from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag


PUBLIC_PRICE_COLUMNS = [
    "id",
    "source_type",
    "source_code",
    "hospital_code",
    "hospital_name",
    "state",
    "category",
    "service_name",
    "original_service_name",
    "charge_type",
    "citizenship",
    "patient_class",
    "patient_type",
    "ward_class",
    "room_type",
    "price_rm",
    "price_unit",
    "original_price_text",
    "notes",
    "source_name",
    "source_url",
    "source_updated_at",
    "scraped_at",
]

FREE_PATTERN = re.compile(r"\b(?:free|percuma|tiada\s+caj)\b", re.IGNORECASE)
RM_PATTERN = re.compile(r"RM\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(?<![\w])([\d,]+(?:\.\d+)?)(?![\w])")


@dataclass(frozen=True)
class PublicPricingSource:
    source_type: str
    source_code: str
    source_name: str
    source_url: str
    state: str
    hospital_code: str = ""
    hospital_name: str = ""
    source_updated_at: str = ""


@dataclass
class ScrapeSummary:
    source_code: str
    records: int
    categories: list[str]
    duplicates_removed: int
    skipped_malformed: int
    output_path: Path
    used_snapshot_fallback: bool = False


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return " ".join(str(value).replace("\xa0", " ").split()).strip()


def patient_class_from_label(value: str) -> str:
    normalized = re.sub(r"[^a-z]", "", value.casefold())
    if normalized in {
        "warganegara",
        "warganegaramalaysia",
        "malaysian",
        "citizen",
        "citizens",
    }:
        return "citizen"
    if normalized in {
        "bukanwarganegara",
        "warganegaraasing",
        "wargaasing",
        "noncitizen",
        "noncitizens",
        "foreigner",
        "foreigners",
    }:
        return "foreigner"
    if normalized == "unhcr":
        return "unhcr"
    if "awam" in normalized or "pemastautintetap" in normalized:
        return "resident"
    if normalized in {"all", "semua"}:
        return "all"
    return ""


def citizenship_from_patient_class(patient_class: str) -> str:
    return {
        "citizen": "Malaysian",
        "foreigner": "Non-Malaysian",
        "unhcr": "Non-Malaysian (UNHCR)",
        "resident": "Non-Malaysian (eligible public category)",
        "all": "All",
    }.get(patient_class, "")


def parse_price(value: str, *, allow_bare: bool = True) -> float | None:
    text = clean_text(value)
    if not text or text in {"-", "–", "—"}:
        return None
    if FREE_PATTERN.search(text):
        return 0.0
    match = RM_PATTERN.search(text)
    if match is None and allow_bare:
        match = NUMBER_PATTERN.search(text)
    if match is None:
        return None
    return float(match.group(1).replace(",", ""))


def parse_all_prices(value: str, *, allow_bare: bool = False) -> list[float]:
    text = clean_text(value)
    prices = [
        float(match.replace(",", ""))
        for match in RM_PATTERN.findall(text)
    ]
    if not prices and allow_bare:
        prices = [
            float(match.replace(",", ""))
            for match in NUMBER_PATTERN.findall(text)
        ]
    if FREE_PATTERN.search(text):
        prices.insert(0, 0.0)
    return prices


def infer_price_unit(*values: str, default: str = "") -> str:
    text = " ".join(clean_text(value) for value in values).casefold()
    for pattern, unit in (
        (r"(?:sehari|per\s*day)", "per day"),
        (r"(?:setiap\s+lawatan|per\s*visit)", "per visit"),
        (r"(?:setiap\s+sesi|per\s*session)", "per session"),
        (r"(?:setiap\s+rawatan|per\s*treatment)", "per treatment"),
        (r"(?:setiap\s+kemasukan|per\s*admission)", "per admission"),
        (r"(?:sebulan|per\s*month)", "per month"),
        (r"(?:setiap\s+pecahan|per\s*fraction)", "per fraction"),
        (r"(?:setiap\s+ulangan|follow.?up|subsequent)", "per follow-up"),
    ):
        if re.search(pattern, text, re.IGNORECASE):
            return unit
    return default


def ward_class_from_text(value: str) -> str:
    match = re.search(r"(?:kelas|class)\s*(satu|dua|tiga|1|2|3)", value, re.I)
    if match is None:
        return ""
    number = {"satu": "1", "dua": "2", "tiga": "3"}.get(
        match.group(1).casefold(), match.group(1)
    )
    return f"Class {number}"


def table_grid(table: Tag) -> list[list[str]]:
    """Expand HTML rowspans and colspans into a rectangular text grid."""

    rows: list[list[str]] = []
    pending: dict[int, tuple[int, str]] = {}

    for html_row in table.find_all("tr"):
        row: list[str] = []
        column_index = 0

        def fill_pending() -> None:
            nonlocal column_index
            while column_index in pending:
                rows_left, value = pending[column_index]
                row.append(value)
                if rows_left <= 1:
                    del pending[column_index]
                else:
                    pending[column_index] = (rows_left - 1, value)
                column_index += 1

        fill_pending()
        for cell in html_row.find_all(["th", "td"], recursive=False):
            fill_pending()
            value = clean_text(cell.get_text(" ", strip=True))
            colspan = max(int(cell.get("colspan", 1)), 1)
            rowspan = max(int(cell.get("rowspan", 1)), 1)
            for offset in range(colspan):
                row.append(value)
                if rowspan > 1:
                    pending[column_index + offset] = (rowspan - 1, value)
            column_index += colspan
        fill_pending()
        if any(row):
            rows.append(row)

    width = max((len(row) for row in rows), default=0)
    return [row + [""] * (width - len(row)) for row in rows]


def fetch_official_html(
    source: PublicPricingSource,
    *,
    fallback_path: Path | None = None,
    verify_tls: bool = True,
    timeout: int = 40,
) -> tuple[str, bool]:
    try:
        response = requests.get(
            source.source_url,
            timeout=timeout,
            verify=verify_tls,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36 MyCareCost/1.0"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "ms-MY,ms;q=0.9,en;q=0.8",
            },
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding
        return response.text, False
    except requests.RequestException as exc:
        if fallback_path is not None and fallback_path.exists():
            return fallback_path.read_text(encoding="utf-8"), True
        raise RuntimeError(
            f"Failed to fetch official pricing page {source.source_url}: {exc}"
        ) from exc


def extract_source_updated_at(html: str) -> str:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    numeric = re.search(
        r"(?:kemas kini|kemaskini|tarikh kemaskini)[^\d]{0,20}"
        r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})",
        text,
        re.IGNORECASE,
    )
    if numeric:
        return f"{numeric.group(3)}-{int(numeric.group(2)):02d}-{int(numeric.group(1)):02d}"

    months = {
        "januari": 1,
        "februari": 2,
        "mac": 3,
        "april": 4,
        "mei": 5,
        "jun": 6,
        "julai": 7,
        "ogos": 8,
        "september": 9,
        "oktober": 10,
        "november": 11,
        "disember": 12,
    }
    written = re.search(
        r"(?:kemas kini|kemaskini|tarikh kemaskini)[^\d]{0,20}"
        r"(\d{1,2})\s+(" + "|".join(months) + r")\s+(\d{4})",
        text,
        re.IGNORECASE,
    )
    if written:
        return (
            f"{written.group(3)}-{months[written.group(2).casefold()]:02d}-"
            f"{int(written.group(1)):02d}"
        )
    return ""


def make_record(
    source: PublicPricingSource,
    *,
    category: str,
    service_name: str,
    charge_type: str,
    price_rm: float,
    original_price_text: str,
    patient_class: str = "all",
    patient_type: str = "",
    ward_class: str = "",
    room_type: str = "",
    price_unit: str = "",
    notes: str = "",
    original_service_name: str | None = None,
    scraped_at: str,
    source_updated_at: str | None = None,
) -> dict[str, Any]:
    original_name = clean_text(original_service_name or service_name)
    normalized_service = clean_text(service_name)
    semantic_key = "|".join(
        [
            source.source_code,
            category,
            normalized_service,
            charge_type,
            patient_class,
            patient_type,
            ward_class,
            room_type,
            str(float(price_rm)),
            price_unit,
        ]
    )
    record_id = hashlib.sha1(semantic_key.encode("utf-8")).hexdigest()[:16]
    return {
        "id": f"{source.source_code.lower()}-{record_id}",
        "source_type": source.source_type,
        "source_code": source.source_code,
        "hospital_code": source.hospital_code,
        "hospital_name": source.hospital_name,
        "state": source.state,
        "category": clean_text(category),
        "service_name": normalized_service,
        "original_service_name": original_name,
        "charge_type": clean_text(charge_type),
        "citizenship": citizenship_from_patient_class(patient_class),
        "patient_class": patient_class,
        "patient_type": clean_text(patient_type),
        "ward_class": clean_text(ward_class),
        "room_type": clean_text(room_type),
        "price_rm": float(price_rm),
        "price_unit": clean_text(price_unit),
        "original_price_text": clean_text(original_price_text),
        "notes": clean_text(notes),
        "source_name": source.source_name,
        "source_url": source.source_url,
        "source_updated_at": source_updated_at or source.source_updated_at,
        "scraped_at": scraped_at,
    }


def finalize_records(
    records: list[dict[str, Any]],
    *,
    source: PublicPricingSource,
    output_path: Path,
    skipped_malformed: int,
    used_snapshot_fallback: bool,
) -> tuple[pd.DataFrame, ScrapeSummary]:
    frame = pd.DataFrame(records)
    for column in PUBLIC_PRICE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[PUBLIC_PRICE_COLUMNS].copy()
    frame["price_rm"] = pd.to_numeric(frame["price_rm"], errors="coerce")
    before_validation = len(frame)
    frame = frame[frame["price_rm"].notna() & frame["price_rm"].ge(0)].copy()
    skipped_malformed += before_validation - len(frame)
    before_deduplication = len(frame)
    frame = frame.drop_duplicates(
        subset=[
            "source_code",
            "category",
            "service_name",
            "charge_type",
            "patient_class",
            "patient_type",
            "ward_class",
            "room_type",
            "price_rm",
            "price_unit",
        ]
    ).reset_index(drop=True)
    duplicates_removed = before_deduplication - len(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    categories = sorted(frame["category"].unique().tolist(), key=str.casefold)
    return frame, ScrapeSummary(
        source_code=source.source_code,
        records=len(frame),
        categories=categories,
        duplicates_removed=duplicates_removed,
        skipped_malformed=skipped_malformed,
        output_path=output_path,
        used_snapshot_fallback=used_snapshot_fallback,
    )


def utc_scraped_at() -> str:
    return datetime.now(timezone.utc).isoformat()

