"""Official Schedule 25.0 names and category decoders for NSS 80 retraining.

The mappings in this module are transcribed from the official 15-page NSS
Schedule 25.0 questionnaire stored at
``datasets/nss80/documentation/nss80_health_schedule.pdf``.  Auxiliary
public-use survey-design columns that are not defined by that questionnaire
are kept, but are explicitly labelled as unverified and are never used as
predictors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .codebook import AILMENT as EXISTING_AILMENT
from .codebook import STATE as EXISTING_STATE


OFFICIAL_SOURCE = "NSS 80 Schedule 25.0 (January-December 2025)"

HOUSEHOLD_KEY = [
    "rnd",
    "sch",
    "fsu",
    "samp",
    "sec",
    "st",
    "nssreg",
    "dist",
    "strm",
    "sstrm",
    "subrnd",
    "sro",
    "sd",
    "sss",
    "hhd",
]

# ``suno`` is retained as a readable source identifier but is not a reliable
# cross-level join field in the supplied public-use files.  L1 has missing
# values for some otherwise uniquely identified households, and a small number
# of L2/L3 rows disagree with L4.  The remaining fields above uniquely identify
# L1 households and L2/L3 people in this release.


@dataclass(frozen=True)
class ColumnDefinition:
    official_meaning: str
    clean_name: str
    data_type: str
    notes: str = ""


def _definition(
    meaning: str, clean_name: str, data_type: str, notes: str = ""
) -> ColumnDefinition:
    return ColumnDefinition(meaning, clean_name, data_type, notes)


COMMON_COLUMN_DEFINITIONS = {
    "rnd": _definition("Round number", "round_number", "identifier"),
    "sch": _definition("Schedule number", "schedule_number", "identifier"),
    "fsu": _definition(
        "Serial number of sample first-stage unit", "fsu_serial_no", "identifier"
    ),
    "samp": _definition("Type of sample", "sample_type", "categorical"),
    "sec": _definition("Sector", "sector", "categorical"),
    "st": _definition("State or Union Territory", "state", "categorical"),
    "nssreg": _definition(
        "NSS region code",
        "nss_region",
        "categorical",
        "The schedule does not provide region-name labels; values remain explicit code labels.",
    ),
    "dist": _definition(
        "District code",
        "district",
        "categorical",
        "The schedule names the district field but does not provide district-name labels; values are paired with state.",
    ),
    "strm": _definition(
        "Survey stratum",
        "survey_stratum",
        "identifier",
        "Meaning supported by the official catalog sampling description; code labels are not in Schedule 25.0.",
    ),
    "sstrm": _definition(
        "Survey sub-stratum",
        "survey_sub_stratum",
        "identifier",
        "Meaning supported by the official catalog sampling description; code labels are not in Schedule 25.0.",
    ),
    "subrnd": _definition(
        "Survey sub-round", "survey_sub_round", "identifier"
    ),
    "sro": _definition(
        "Auxiliary public-use survey-design field (sro)",
        "unverified_survey_design_field_sro",
        "unverified",
        "Not defined in the available Schedule 25.0 PDF; excluded from modelling.",
    ),
    "suno": _definition(
        "Sample sub-unit number",
        "sample_sub_unit_no",
        "identifier",
        "Retained for audit/reference; excluded from cross-level joins because the public-use files contain missing or inconsistent values across levels.",
    ),
    "sd": _definition(
        "Sample sub-division number", "sample_sub_division_no", "identifier"
    ),
    "sss": _definition(
        "Second-stage stratum number", "second_stage_stratum_no", "identifier"
    ),
    "hhd": _definition(
        "Sample household number", "sample_household_no", "identifier"
    ),
    "level": _definition(
        "Public-use record level", "source_record_level", "identifier"
    ),
    "mult": _definition(
        "Survey multiplier",
        "survey_multiplier",
        "numeric",
        "Used as an optional normalized training weight; not a predictive feature.",
    ),
    "nst": _definition(
        "Auxiliary public-use survey-design field (nst)",
        "unverified_survey_design_field_nst",
        "unverified",
        "Not defined in the available Schedule 25.0 PDF; excluded from modelling.",
    ),
    "nstj": _definition(
        "Auxiliary public-use survey-design field (nstj)",
        "unverified_survey_design_field_nstj",
        "unverified",
        "Not defined in the available Schedule 25.0 PDF; excluded from modelling.",
    ),
    "subdvsn": _definition(
        "Auxiliary public-use survey-design field (subdvsn)",
        "unverified_survey_design_field_subdvsn",
        "unverified",
        "Not defined in the available Schedule 25.0 PDF; excluded from modelling.",
    ),
    "caph": _definition(
        "Auxiliary public-use survey-design field (caph)",
        "unverified_survey_design_field_caph",
        "unverified",
        "Not defined in the available Schedule 25.0 PDF; excluded from modelling.",
    ),
    "smah": _definition(
        "Auxiliary public-use survey-design field (smah)",
        "unverified_survey_design_field_smah",
        "unverified",
        "Not defined in the available Schedule 25.0 PDF; excluded from modelling.",
    ),
}


BLOCK_COLUMN_DEFINITIONS = {
    # Level 1: identification, fieldwork and Block 5 household characteristics.
    "svc": _definition("Survey code", "survey_code", "categorical"),
    "b1i16": _definition(
        "Reason for substitution or casualty of the original household",
        "reason_for_household_substitution_or_casualty",
        "categorical",
    ),
    "infcode": _definition(
        "Serial number of informant household member", "informant_member_serial_no", "identifier"
    ),
    "b2i9": _definition(
        "Response code of informant assessed by enumerator",
        "informant_response_code",
        "categorical",
    ),
    "date": _definition(
        "Auxiliary public-use field (date)",
        "unverified_fieldwork_date",
        "unverified",
        "The public-use column is not tied to one of the four Block 2 dates in the available layout.",
    ),
    "time": _definition(
        "Auxiliary public-use field (time)",
        "unverified_fieldwork_time",
        "unverified",
        "The public-use column is not unambiguously defined in the available layout.",
    ),
    "hhsz": _definition("Household size", "household_size", "numeric"),
    "b5i2": _definition("Religion", "religion", "categorical"),
    "b5i3": _definition("Social group", "social_group", "categorical"),
    "b5i4": _definition("Household type", "household_type", "categorical"),
    "b5i5": _definition(
        "Sudden community outbreak of communicable disease during last 365 days",
        "community_communicable_disease_outbreak",
        "binary",
    ),
    "b5i6": _definition(
        "Medical insurance premium paid during last 365 days (Rs.)",
        "medical_insurance_premium_rs",
        "numeric",
    ),
    "b5i7": _definition(
        "Usual monthly purchased household goods and services expenditure (Rs.)",
        "monthly_purchased_household_goods_services_rs",
        "numeric",
    ),
    "b5i8": _definition(
        "Usual monthly imputed consumption from home-grown stock (Rs.)",
        "monthly_home_grown_consumption_imputed_rs",
        "numeric",
    ),
    "b5i9": _definition(
        "Usual monthly imputed consumption from wages in kind, gifts or free collection (Rs.)",
        "monthly_in_kind_gifts_consumption_imputed_rs",
        "numeric",
    ),
    "b5i10": _definition(
        "Annual clothing and footwear purchase expenditure (Rs.)",
        "annual_clothing_footwear_expenditure_rs",
        "numeric",
    ),
    "b5i11": _definition(
        "Annual household durables purchase expenditure (Rs.)",
        "annual_household_durables_expenditure_rs",
        "numeric",
    ),
    "umce": _definition(
        "Usual monthly consumer expenditure (Rs.)",
        "household_usual_consumer_expenditure_rs",
        "numeric",
    ),
    # Level 2: Block 3 household members.
    "b3c1": _definition("Household member serial number", "person_serial_no", "identifier"),
    "b3c3": _definition("Relation to household head", "relation_to_household_head", "categorical"),
    "b3c4": _definition("Gender", "gender", "categorical"),
    "b3c5": _definition("Age (years)", "age_years", "numeric"),
    "b3c6": _definition("Marital status", "marital_status", "categorical"),
    "b3c7": _definition("Highest educational level attained", "highest_education_level", "categorical"),
    "b3c8": _definition("Received any vaccine during last 365 days", "received_vaccine", "binary"),
    "b3c9": _definition("Hospitalised during last 365 days", "hospitalised", "binary"),
    "b3c10": _definition("Number of hospitalisations during last 365 days", "number_of_hospitalisations", "numeric"),
    "b3c11": _definition("Pregnant during last 365 days", "pregnant", "binary"),
    "b3c12": _definition("Household paid major share of childbirth expenses", "household_paid_major_childbirth_share", "categorical"),
    "b3c13": _definition("Communicable disease during last 365 days", "communicable_disease", "categorical"),
    "b3c14": _definition("Suffering from any chronic ailment", "chronic_ailment", "binary"),
    "b3c15": _definition("Other ailment during last 15 days", "other_ailment_last_15_days", "binary"),
    "b3c16": _definition("Other ailment on day before survey", "other_ailment_previous_day", "binary"),
    "b3c17": _definition("Health financing scheme or insurance coverage", "health_financing_or_insurance_coverage", "categorical"),
    # Level 3: Block 4 deceased former members.
    "b4c1": _definition("Deceased former member serial number", "deceased_member_serial_no", "identifier"),
    "b4c3": _definition("Gender", "gender", "categorical"),
    "b4c4": _definition("Age at death (years)", "age_at_death_years", "numeric"),
    "b4c5": _definition("Medical attention received before death", "medical_attention_before_death", "binary"),
    "b4c6": _definition("Hospitalised during last 365 days", "hospitalised", "binary"),
    "b4c7": _definition("Number of hospitalisations during last 365 days", "number_of_hospitalisations", "numeric"),
    "b4c8": _definition("Reason for non-hospitalisation just before death", "reason_for_non_hospitalisation_before_death", "categorical"),
    "b4c9": _definition("Suffered from chronic ailment", "chronic_ailment", "categorical"),
    "b4c10": _definition("Other ailment during last 15 days", "other_ailment_last_15_days", "categorical"),
    "b4c11": _definition("Pregnant during last 365 days", "pregnant", "binary"),
    "b4c12": _definition("Time of death in relation to pregnancy", "pregnancy_related_time_of_death", "categorical"),
    # Level 4: Blocks 6 and 7 inpatient episodes and expenditure.
    "b6i1": _definition("Hospitalisation case serial number", "hospitalisation_case_serial_no", "identifier"),
    "b6i2": _definition("Serial number of hospitalised member", "person_serial_no", "identifier"),
    "b6i3": _definition("Age (years)", "age_years", "numeric"),
    "b6i4": _definition("Age (days) for member under one year", "age_days_if_under_one_year", "numeric"),
    "b6i5": _definition("Nature of ailment", "ailment_nature", "categorical"),
    "b6i6": _definition("Nature of hospitalisation treatment", "hospitalisation_treatment_nature", "categorical"),
    "b6i7": _definition("Type of medical institution", "medical_institution_type", "categorical"),
    "b6i8": _definition("Reason for not availing government or public hospital", "reason_not_using_government_public_hospital", "categorical"),
    "b6i9": _definition("Type of ward", "ward_type", "categorical"),
    "b6i10": _definition("When admitted", "admission_timing", "categorical"),
    "b6i11": _definition("When discharged", "discharge_timing", "categorical"),
    "b6i12": _definition("Duration of stay in hospital (days)", "length_of_stay_days", "numeric"),
    "b6i13": _definition("Surgery service received", "surgery", "categorical"),
    "b6i14": _definition("Medicine service received", "medicine", "categorical"),
    "b6i15": _definition("X-ray, ECG, EEG or scan service received", "xray_ecg_eeg_scan", "categorical"),
    "b6i16": _definition("Other diagnostic tests received", "other_diagnostic_tests", "categorical"),
    "b6i17": _definition("Treated on medical advice before hospitalisation", "treated_on_medical_advice_before_hospitalisation", "binary"),
    "b6i18": _definition("Pre-hospitalisation nature of treatment", "pre_hospitalisation_treatment_nature", "categorical"),
    "b6i19": _definition("Pre-hospitalisation level of care", "pre_hospitalisation_level_of_care", "categorical"),
    "b6i20": _definition("Pre-hospitalisation treatment duration (days)", "pre_hospitalisation_treatment_duration_days", "numeric"),
    "b6i21": _definition("Medical-advice treatment continued after discharge", "treatment_continued_after_discharge", "binary"),
    "b6i22": _definition("Post-hospitalisation nature of treatment", "post_hospitalisation_treatment_nature", "categorical"),
    "b6i23": _definition("Post-hospitalisation level of care", "post_hospitalisation_level_of_care", "categorical"),
    "b6i24": _definition("Post-hospitalisation treatment duration (days)", "post_hospitalisation_treatment_duration_days", "numeric"),
    "b7i5": _definition("Any medical service provided free, fully or partly", "medical_service_free_fully_or_partly", "categorical"),
    "b7i6": _definition("Package component (Rs.)", "package_component_rs", "numeric"),
    "b7i7": _definition("Doctor or surgeon fee (Rs.)", "doctor_surgeon_fee_rs", "numeric"),
    "b7i8": _definition("Medicines expenditure (Rs.)", "medicines_rs", "numeric"),
    "b7i9": _definition("Diagnostic tests expenditure (Rs.)", "diagnostic_tests_rs", "numeric"),
    "b7i10": _definition("Bed charges (Rs.)", "bed_charges_rs", "numeric"),
    "b7i11": _definition("Other medical expenses (Rs.)", "other_medical_expenses_rs", "numeric"),
    "b7i12": _definition("Total medical expenditure, items 6-11 (Rs.)", "total_medical_expenditure_rs", "numeric"),
    "b7i13": _definition("Patient transport expenditure (Rs.)", "patient_transport_rs", "numeric"),
    "b7i14": _definition("Other non-medical household expenses (Rs.)", "other_non_medical_household_expenses_rs", "numeric"),
    "b7i15": _definition("Total expenditure, items 12-14 (Rs.)", "total_expenditure_rs", "numeric"),
    "b7i16": _definition("Insurance or employer reimbursement (Rs.)", "insurance_or_employer_reimbursement_rs", "numeric"),
    "b7i17": _definition("Major source of finance for expenses", "major_source_of_finance", "categorical"),
    "b7i18": _definition("Place of hospitalisation", "place_of_hospitalisation", "categorical"),
    "b7i19": _definition("State code when hospitalised in another state", "treatment_state_code", "categorical"),
    "b7i20": _definition("Household income loss due to hospitalisation (Rs.)", "household_income_loss_due_to_hospitalisation_rs", "numeric"),
    # Level 5: Blocks 8 and 9 ailment spells and non-inpatient expenditure.
    "b8i1": _definition("Ailment spell serial number", "ailment_spell_serial_no", "identifier"),
    "b8i2": _definition("Serial number of member reporting ailment", "person_serial_no", "identifier"),
    "b8i3": _definition("Age (years)", "age_years", "numeric"),
    "b8i4": _definition("Age (days) for member under one year", "age_days_if_under_one_year", "numeric"),
    "b8i5": _definition("Nature of ailment", "ailment_nature", "categorical"),
    "b8i6": _definition("Whether ailment is chronic", "chronic_ailment", "binary"),
    "b8i7": _definition("Status of ailment", "ailment_status", "categorical"),
    "b8i8": _definition("Total duration of ailment (days)", "ailment_duration_days", "numeric"),
    "b8i9": _definition("Nature of treatment", "treatment_nature", "categorical"),
    "b8i10": _definition("Whether hospitalised", "hospitalised", "binary"),
    "b8i11": _definition("Treatment taken on medical advice", "treatment_on_medical_advice", "binary"),
    "b8i12": _definition("Level of care", "level_of_care", "categorical"),
    "b8i13": _definition("Reason for not availing government source", "reason_not_using_government_source", "categorical"),
    "b8i14": _definition("Reason for not seeking medical advice", "reason_not_seeking_medical_advice", "categorical"),
    "b8i15": _definition("Person consulted without medical advice", "person_consulted_without_medical_advice", "categorical"),
    "b9i5": _definition("Any medical service provided free, fully or partly", "medical_service_free_fully_or_partly", "categorical"),
    "b9i6": _definition("Surgery service received", "surgery", "categorical"),
    "b9i7": _definition("Ayush medicine service received", "ayush_medicine", "categorical"),
    "b9i8": _definition("Non-Ayush medicine service received", "non_ayush_medicine", "categorical"),
    "b9i9": _definition("X-ray, ECG, EEG or scan service received", "xray_ecg_eeg_scan", "categorical"),
    "b9i10": _definition("Other diagnostic tests received", "other_diagnostic_tests", "categorical"),
    "b9i11": _definition("Doctor or surgeon fee (Rs.)", "doctor_surgeon_fee_rs", "numeric"),
    "b9i12": _definition("Ayush medicines expenditure (Rs.)", "ayush_medicines_rs", "numeric"),
    "b9i13": _definition("Non-Ayush medicines expenditure (Rs.)", "non_ayush_medicines_rs", "numeric"),
    "b9i14": _definition("Diagnostic tests expenditure (Rs.)", "diagnostic_tests_rs", "numeric"),
    "b9i15": _definition("Other medical expenses (Rs.)", "other_medical_expenses_rs", "numeric"),
    "b9i16": _definition("Total medical expenditure, items 11-15 (Rs.)", "total_medical_expenditure_rs", "numeric"),
    "b9i17": _definition("Patient transport expenditure (Rs.)", "patient_transport_rs", "numeric"),
    "b9i18": _definition("Other household expenses (Rs.)", "other_household_expenses_rs", "numeric"),
    "b9i19": _definition("Total expenditure, items 16-18 (Rs.)", "total_expenditure_rs", "numeric"),
    "b9i20": _definition("Insurance or employer reimbursement (Rs.)", "insurance_or_employer_reimbursement_rs", "numeric"),
    "b9i21": _definition("Major source of finance for expenses", "major_source_of_finance", "categorical"),
    "b9i22": _definition("Place of treatment", "place_of_treatment", "categorical"),
    "b9i23": _definition("State code when treated in another state", "treatment_state_code", "categorical"),
    "b9i24": _definition("Household income loss due to treatment (Rs.)", "household_income_loss_due_to_treatment_rs", "numeric"),
    # Level 6: Block 10 vaccination.
    "b10i1": _definition("Vaccine serial number", "vaccine_serial_no", "identifier"),
    "b10i2": _definition("Household member serial number", "person_serial_no", "identifier"),
    "b10i3": _definition("Age (years)", "age_years", "numeric"),
    "b10i4": _definition("Type of vaccine", "vaccine_type", "categorical"),
    "b10i5": _definition("Source of vaccination", "vaccination_source", "categorical"),
    "b10i6": _definition("Whether vaccination expenditure was incurred", "vaccination_expenditure_incurred", "binary"),
    "b10i7": _definition("Vaccination expenditure (Rs.)", "vaccination_expenditure_rs", "numeric"),
    # Level 7: Block 11 maternity care.
    "b11c1": _definition("Household member serial number", "person_serial_no", "identifier"),
    "b11c2": _definition("Age (years)", "age_years", "numeric"),
    "b11c3": _definition("Pregnancy serial number", "pregnancy_serial_no", "identifier"),
    "b11c4": _definition("Major source of ante-natal care", "ante_natal_care_source", "categorical"),
    "b11c5": _definition("Nature of ante-natal care", "ante_natal_care_nature", "categorical"),
    "b11c6": _definition("Ante-natal care expenditure (Rs.)", "ante_natal_care_expenditure_rs", "numeric"),
    "b11c7": _definition("Outcome of pregnancy", "pregnancy_outcome", "categorical"),
    "b11c8": _definition("Place of delivery or abortion", "delivery_or_abortion_place", "categorical"),
    "b11c9": _definition("Delivery attendant", "delivery_attendant", "categorical"),
    "b11c10": _definition("Expenditure on delivery at home (Rs.)", "home_delivery_expenditure_rs", "numeric"),
    "b11c11": _definition("Major source of post-natal care", "post_natal_care_source", "categorical"),
    "b11c12": _definition("Nature of post-natal care", "post_natal_care_nature", "categorical"),
    "b11c13": _definition("Post-natal care expenditure (Rs.)", "post_natal_care_expenditure_rs", "numeric"),
}

COLUMN_DEFINITIONS = {**COMMON_COLUMN_DEFINITIONS, **BLOCK_COLUMN_DEFINITIONS}


def _title_mapping(mapping: dict[str, str]) -> dict[str, str]:
    return {str(code): label.replace("_", " ").capitalize() for code, label in mapping.items()}


STATE = {str(code): label for code, label in EXISTING_STATE.items() if code != "99"}
STATE["99"] = "Unverified state code 99"
AILMENT = {str(code): label for code, label in EXISTING_AILMENT.items()}

YES_NO = {"1": "Yes", "2": "No"}
GENDER = {"1": "Male", "2": "Female", "3": "Transgender"}
RELATION = {
    "1": "Self",
    "2": "Spouse of household head",
    "3": "Married child",
    "4": "Spouse of married child",
    "5": "Unmarried child",
    "6": "Grandchild",
    "7": "Parent or parent-in-law",
    "8": "Other relative",
    "9": "Employee, servant or other non-relative",
}
MARITAL_STATUS = {
    "1": "Never married",
    "2": "Currently married or living together",
    "3": "Widowed",
    "4": "Divorced or separated",
}
EDUCATION = {
    "1": "Not literate",
    "2": "Literate with non-formal education",
    "3": "Formal education below primary",
    "4": "Primary",
    "5": "Upper primary or middle",
    "6": "Secondary",
    "7": "Higher secondary",
    "8": "Diploma or certificate up to secondary",
    "10": "Diploma or certificate at higher-secondary level",
    "11": "Diploma or certificate at graduation level or above",
    "12": "Graduate",
    "13": "Postgraduate or above",
}
COMMUNICABLE_DISEASE = {
    "1": "Malaria",
    "2": "Viral hepatitis with jaundice",
    "3": "Viral hepatitis without jaundice",
    "4": "Acute diarrhoeal disease or dysentery",
    "5": "Dengue fever",
    "6": "Chikungunya",
    "7": "Measles",
    "8": "Acute encephalitis syndrome",
    "9": "Other specified communicable disease",
    "10": "HIV or AIDS",
    "11": "Sexually or reproductive-tract transmitted infection",
    "12": "Leprosy",
    "19": "Not suffered",
}
INSURANCE_COVERAGE = {
    "1": "AB-PMJAY",
    "2": "State health insurance scheme",
    "3": "ESIS or ESIC",
    "4": "CGHS, ECHS or other central-government health scheme",
    "5": "State-government medical reimbursement for employees",
    "6": "Public-sector undertaking as employer",
    "7": "Other employer health insurance or reimbursement",
    "10": "Privately purchased commercial insurance only",
    "19": "Not covered",
}
TREATMENT_NATURE = {
    "1": "Allopathy",
    "2": "Ayurveda, Yoga, Naturopathy, Unani, Siddha, Sowa-Rigpa or Homoeopathy",
    "3": "Allopathy and Ayush or traditional treatment",
    "9": "Other treatment",
}
MEDICAL_INSTITUTION = {
    "1": "Government or public hospital",
    "2": "Charitable, trust or NGO-run hospital",
    "3": "Private hospital",
}
WARD_TYPE = {"1": "Free ward", "2": "Paying general ward", "3": "Paying special ward"}
SERVICE_RECEIPT = {
    "1": "Not received",
    "2": "Received free",
    "3": "Received partly free",
    "4": "Received on payment",
}
REASON_NOT_GOVERNMENT = {
    "1": "Required specific services not available",
    "2": "Available but quality unsatisfactory or doctor unavailable",
    "3": "Quality satisfactory but facility too far",
    "4": "Quality satisfactory but long waiting time",
    "5": "Financial constraint",
    "6": "Preference for trusted doctor or hospital",
    "9": "Other reason",
}
LEVEL_OF_CARE = {
    "1": "Government or public hospital",
    "2": "Charitable, trust or NGO-run hospital",
    "3": "Private hospital",
    "4": "Private doctor or clinic",
    "5": "Informal healthcare provider",
}
FREE_MEDICAL_SERVICE = {
    "1": "Yes, government or public provider",
    "2": "Yes, private, charitable, NGO or trust provider",
    "3": "Yes, both provider types",
    "4": "No",
}
FINANCE_SOURCE = {
    "1": "Household income or savings",
    "2": "Borrowings",
    "3": "Sale of physical assets",
    "4": "Contributions from friends and relatives",
    "9": "Other source",
}
PLACE = {
    "1": "Same district, rural area",
    "2": "Same district, urban area",
    "3": "Different district in same state, rural area",
    "4": "Different district in same state, urban area",
    "5": "Other state",
}


CATEGORY_MAPPINGS: dict[str, dict[str, str]] = {
    "samp": {"1": "Central sample", "2": "State sample"},
    "sec": {"1": "Rural", "2": "Urban"},
    "st": STATE,
    "svc": {"1": "Original household", "2": "Substitute household", "3": "Casualty"},
    "b1i16": {"1": "Informant busy", "2": "Members away from home", "3": "Informant non-cooperative", "9": "Other reason"},
    "b2i9": {"1": "Co-operative and capable", "2": "Co-operative but not capable", "3": "Busy", "4": "Reluctant", "9": "Other response"},
    "b5i2": {"1": "Hinduism", "2": "Islam", "3": "Christianity", "4": "Sikhism", "5": "Jainism", "6": "Buddhism", "7": "Zoroastrianism", "9": "Other religion"},
    "b5i3": {"1": "Scheduled tribe", "2": "Scheduled caste", "3": "Other backward class", "9": "Other social group"},
    "b5i5": YES_NO,
    "b3c3": RELATION,
    "b3c4": GENDER,
    "b3c6": MARITAL_STATUS,
    "b3c7": EDUCATION,
    "b3c8": YES_NO,
    "b3c9": YES_NO,
    "b3c11": YES_NO,
    "b3c12": {"1": "Yes", "2": "No", "3": "Pregnancy continuing"},
    "b3c13": COMMUNICABLE_DISEASE,
    "b3c14": YES_NO,
    "b3c15": YES_NO,
    "b3c16": YES_NO,
    "b3c17": INSURANCE_COVERAGE,
    "b4c3": GENDER,
    "b4c5": YES_NO,
    "b4c6": YES_NO,
    "b4c8": {"1": "Hospital care not considered satisfactory", "2": "Doctor or medical attendant unavailable", "3": "Ailment not considered serious", "4": "Financial constraints", "5": "Transportation problem", "6": "Patient did not want hospitalisation", "7": "Patient died before hospital transfer", "9": "Other reason"},
    "b4c9": {"1": "Yes", "2": "No", "9": "Not applicable"},
    "b4c10": {"1": "Yes", "2": "No", "9": "Not applicable"},
    "b4c11": YES_NO,
    "b4c12": {"1": "During pregnancy", "2": "During delivery", "3": "During abortion", "4": "Within six weeks of delivery or abortion", "9": "Other cause"},
    "b6i5": AILMENT,
    "b6i6": TREATMENT_NATURE,
    "b6i7": MEDICAL_INSTITUTION,
    "b6i8": REASON_NOT_GOVERNMENT,
    "b6i9": WARD_TYPE,
    "b6i10": {"1": "During last 15 days", "2": "16 to 365 days ago", "3": "More than 365 days ago"},
    "b6i11": {"1": "Not yet discharged", "2": "During last 15 days", "3": "16 to 365 days ago"},
    "b6i13": SERVICE_RECEIPT,
    "b6i14": SERVICE_RECEIPT,
    "b6i15": SERVICE_RECEIPT,
    "b6i16": SERVICE_RECEIPT,
    "b6i17": YES_NO,
    "b6i18": TREATMENT_NATURE,
    "b6i19": LEVEL_OF_CARE,
    "b6i21": YES_NO,
    "b6i22": TREATMENT_NATURE,
    "b6i23": LEVEL_OF_CARE,
    "b7i5": FREE_MEDICAL_SERVICE,
    "b7i17": FINANCE_SOURCE,
    "b7i18": PLACE,
    "b7i19": STATE,
    "b8i5": AILMENT,
    "b8i6": YES_NO,
    "b8i7": {"1": "Started over 15 days ago and continuing", "2": "Started over 15 days ago and ended", "3": "Started within 15 days and continuing", "4": "Started within 15 days and ended"},
    "b8i9": {**TREATMENT_NATURE, "5": "No treatment"},
    "b8i10": YES_NO,
    "b8i11": YES_NO,
    "b8i12": LEVEL_OF_CARE,
    "b8i13": REASON_NOT_GOVERNMENT,
    "b8i14": {"1": "No medical facility in neighbourhood", "2": "Facility too expensive", "3": "Could not wait because of domestic or economic engagement", "4": "Ailment not considered serious", "5": "Familial or religious belief", "9": "Other reason"},
    "b8i15": {"1": "Self, household member or friend", "2": "Medicine shop", "9": "Other person"},
    "b9i5": FREE_MEDICAL_SERVICE,
    "b9i6": SERVICE_RECEIPT,
    "b9i7": SERVICE_RECEIPT,
    "b9i8": SERVICE_RECEIPT,
    "b9i9": SERVICE_RECEIPT,
    "b9i10": SERVICE_RECEIPT,
    "b9i21": FINANCE_SOURCE,
    "b9i22": PLACE,
    "b9i23": STATE,
    "b10i4": {
        "01": "BCG, OPV-0, hepatitis-B birth dose, OPV-1, pentavalent-1, RVV-1 or fIPV-1",
        "02": "Pneumococcal conjugate vaccine 1",
        "03": "OPV-2, pentavalent-2 or RVV-2",
        "04": "OPV-3, pentavalent-3, fIPV-2, RVV-3 or PCV-2",
        "05": "MR-1, JE-1, PCV booster or fIPV-3",
        "06": "MR-2, JE-2, DPT booster-1 or OPV booster",
        "07": "DPT booster-2",
        "08": "Tetanus and adult diphtheria",
        "09": "Td1, Td2 or Td booster during pregnancy",
        "10": "Hepatitis A",
        "11": "DTwP or DTaP",
        "12": "Inactivated polio vaccine",
        "13": "MMR third dose",
        "14": "Pneumococcal conjugate vaccine for sickle-cell risk",
        "15": "Tdap or Td",
        "16": "Varicella vaccine",
        "17": "HPV two doses",
        "18": "HPV three doses",
        "19": "Pneumococcal polysaccharide vaccine",
        "20": "Herpes zoster vaccine",
        "21": "Hepatitis-B booster",
        "22": "Oral cholera vaccine",
        "23": "Yellow fever vaccine",
        "24": "Rabies vaccine",
        "25": "COVID-19 booster vaccination",
        "26": "Meningococcal vaccine",
        "27": "Annual influenza vaccine",
    },
    "b10i5": {"1": "Government or public hospital", "2": "Charitable, trust or NGO-run hospital", "3": "Private doctor or clinic", "4": "Private hospital"},
    "b10i6": YES_NO,
    "b11c4": {**LEVEL_OF_CARE, "8": "No ante-natal care received"},
    "b11c5": {"1": "Ayush", "2": "Non-Ayush", "3": "Both Ayush and non-Ayush"},
    "b11c7": {"1": "Pregnancy continuing", "2": "Mother alive and live birth", "3": "Mother alive and stillbirth", "4": "Mother alive and abortion", "5": "Mother died and live birth", "6": "Mother died and stillbirth", "7": "Mother died and abortion", "9": "Other outcome"},
    "b11c8": {"1": "Government or public hospital", "2": "Charitable, trust or NGO-run hospital", "3": "Private hospital or private clinic", "4": "Home"},
    "b11c9": {"1": "Doctor or nurse", "2": "Auxiliary nurse midwife", "3": "Traditional birth attendant", "9": "Other attendant"},
    "b11c11": {**LEVEL_OF_CARE, "8": "No post-natal care received"},
    "b11c12": {"1": "Ayush", "2": "Non-Ayush", "3": "Both Ayush and non-Ayush"},
}

RURAL_HOUSEHOLD_TYPE = {
    "1": "Self-employed in agriculture",
    "2": "Self-employed in non-agriculture",
    "3": "Regular wage or salary earning in agriculture",
    "4": "Regular wage or salary earning in non-agriculture",
    "5": "Casual labour in agriculture",
    "6": "Casual labour in non-agriculture",
    "9": "Other rural household type",
}
URBAN_HOUSEHOLD_TYPE = {
    "1": "Self-employed",
    "2": "Regular wage or salary earning",
    "3": "Casual labour",
    "9": "Other urban household type",
}


def normalize_code(value: object) -> str | None:
    """Normalize integer-looking public-use codes without changing leading zeros."""

    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        text = text[:-2]
    return text


def decode_series(
    source_code: str,
    series: pd.Series,
    *,
    sector: pd.Series | None = None,
    strict: bool = True,
) -> pd.Series:
    """Decode one official categorical field, preserving explicit missingness."""

    normalized = series.map(normalize_code)
    if source_code in {"st", "b7i19", "b9i23"}:
        normalized = normalized.map(
            lambda value: value.zfill(2)
            if isinstance(value, str) and value.isdigit()
            else value
        )
    if source_code == "nssreg":
        return normalized.map(lambda value: None if value is None else f"NSS region code {value}")
    if source_code == "dist":
        if sector is None:
            return normalized.map(lambda value: None if value is None else f"District code {value}")
        return normalized.map(lambda value: None if value is None else f"District code {value}")
    if source_code == "b5i4":
        if sector is None:
            raise ValueError("Sector is required to decode Block 5 household type")
        sector_codes = sector.map(normalize_code)
        decoded = pd.Series(index=series.index, dtype="object")
        rural = sector_codes.eq("1")
        urban = sector_codes.eq("2")
        decoded.loc[rural] = normalized.loc[rural].map(RURAL_HOUSEHOLD_TYPE)
        decoded.loc[urban] = normalized.loc[urban].map(URBAN_HOUSEHOLD_TYPE)
        unknown_mask = normalized.notna() & decoded.isna()
        if strict and unknown_mask.any():
            raise ValueError(
                "Unknown NSS code(s) for b5i4: "
                f"{sorted(normalized.loc[unknown_mask].unique().tolist())}"
            )
        return decoded
    mapping = CATEGORY_MAPPINGS[source_code]
    decoded = normalized.map(mapping)
    unknown_mask = normalized.notna() & decoded.isna()
    if strict and unknown_mask.any():
        raise ValueError(
            f"Unknown NSS code(s) for {source_code}: "
            f"{sorted(normalized.loc[unknown_mask].unique().tolist())}"
        )
    return decoded


def decode_yes_no_binary(series: pd.Series, field: str) -> pd.Series:
    normalized = series.map(normalize_code)
    unknown = normalized.dropna()[~normalized.dropna().isin(["1", "2"])]
    if not unknown.empty:
        raise ValueError(f"Unknown yes/no code(s) for {field}: {sorted(unknown.unique())}")
    return normalized.map({"1": 1, "2": 0}).astype("Int64")


def decode_service_received_binary(series: pd.Series, field: str) -> pd.Series:
    normalized = series.map(normalize_code)
    valid = {"1", "2", "3", "4"}
    unknown = normalized.dropna()[~normalized.dropna().isin(valid)]
    if not unknown.empty:
        raise ValueError(f"Unknown service-receipt code(s) for {field}: {sorted(unknown.unique())}")
    return normalized.map({"1": 0, "2": 1, "3": 1, "4": 1}).astype("Int64")


def decode_coverage_binary(series: pd.Series) -> pd.Series:
    normalized = series.map(normalize_code)
    valid = set(INSURANCE_COVERAGE)
    unknown = normalized.dropna()[~normalized.dropna().isin(valid)]
    if not unknown.empty:
        raise ValueError(f"Unknown insurance code(s): {sorted(unknown.unique())}")
    return normalized.map({code: int(code != "19") for code in valid}).astype("Int64")


def decode_communicable_binary(series: pd.Series) -> pd.Series:
    normalized = series.map(normalize_code)
    valid = set(COMMUNICABLE_DISEASE)
    unknown = normalized.dropna()[~normalized.dropna().isin(valid)]
    if not unknown.empty:
        raise ValueError(f"Unknown communicable-disease code(s): {sorted(unknown.unique())}")
    return normalized.map({code: int(code != "19") for code in valid}).astype("Int64")


def column_mapping_rows(level_columns: dict[str, Iterable[str]]) -> list[dict[str, str]]:
    """Build one documented row per raw code, consolidating source levels."""

    levels_by_code: dict[str, list[str]] = {}
    for level, columns in level_columns.items():
        for column in columns:
            levels_by_code.setdefault(column, []).append(level)
    rows = []
    for column in sorted(levels_by_code):
        definition = COLUMN_DEFINITIONS.get(
            column,
            _definition(
                f"Unverified public-use field ({column})",
                f"unverified_public_use_field_{column}",
                "unverified",
                "Not defined in the available Schedule 25.0 PDF; excluded from modelling.",
            ),
        )
        rows.append(
            {
                "original_code": column,
                "official_meaning": definition.official_meaning,
                "clean_name": definition.clean_name,
                "source_level": ";".join(levels_by_code[column]),
                "data_type": definition.data_type,
                "notes": definition.notes,
            }
        )
    return rows


def value_mapping_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    mappings = dict(CATEGORY_MAPPINGS)
    mappings["b5i4_rural"] = RURAL_HOUSEHOLD_TYPE
    mappings["b5i4_urban"] = URBAN_HOUSEHOLD_TYPE
    for source_code, mapping in sorted(mappings.items()):
        base_code = "b5i4" if source_code.startswith("b5i4_") else source_code
        definition = COLUMN_DEFINITIONS.get(base_code)
        for raw_value, decoded_value in mapping.items():
            binary_value: int | str = ""
            if base_code in {"b5i5", "b3c8", "b3c9", "b3c11", "b3c14", "b3c15", "b3c16", "b4c5", "b4c6", "b4c11", "b6i17", "b6i21", "b8i6", "b8i10", "b8i11", "b10i6"}:
                binary_value = 1 if raw_value == "1" else 0
            elif base_code in {"b6i13", "b6i14", "b6i15", "b6i16", "b9i6", "b9i7", "b9i8", "b9i9", "b9i10"}:
                binary_value = 0 if raw_value == "1" else 1
            elif base_code == "b3c17":
                binary_value = 0 if raw_value == "19" else 1
            elif base_code == "b3c13":
                binary_value = 0 if raw_value == "19" else 1
            rows.append(
                {
                    "feature": definition.clean_name if definition else base_code,
                    "source_code": base_code,
                    "raw_value": raw_value,
                    "decoded_value": decoded_value,
                    "model_binary_value": binary_value,
                    "applies_when": (
                        "sector=Rural"
                        if source_code == "b5i4_rural"
                        else "sector=Urban"
                        if source_code == "b5i4_urban"
                        else ""
                    ),
                    "official_source": (
                        "Public-use data; absent from available Schedule 25.0 state table"
                        if base_code in {"st", "b7i19", "b9i23"}
                        and raw_value == "99"
                        else OFFICIAL_SOURCE
                    ),
                    "notes": (
                        "Meaning not verified; retained as an explicit nominal code label and never treated as numeric."
                        if base_code in {"st", "b7i19", "b9i23"}
                        and raw_value == "99"
                        else "Blank source values remain missing or not applicable."
                    ),
                }
            )
    return rows
