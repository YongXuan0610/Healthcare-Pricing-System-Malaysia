"""Human-readable mappings from the official NSS Schedule 25.0 questionnaire."""

from __future__ import annotations


SECTOR = {
    "1": "rural",
    "2": "urban",
}

GENDER = {
    "1": "male",
    "2": "female",
    "3": "transgender",
}

STATE = {
    "01": "Jammu and Kashmir",
    "02": "Himachal Pradesh",
    "03": "Punjab",
    "04": "Chandigarh",
    "05": "Uttarakhand",
    "06": "Haryana",
    "07": "Delhi",
    "08": "Rajasthan",
    "09": "Uttar Pradesh",
    "10": "Bihar",
    "11": "Sikkim",
    "12": "Arunachal Pradesh",
    "13": "Nagaland",
    "14": "Manipur",
    "15": "Mizoram",
    "16": "Tripura",
    "17": "Meghalaya",
    "18": "Assam",
    "19": "West Bengal",
    "20": "Jharkhand",
    "21": "Odisha",
    "22": "Chhattisgarh",
    "23": "Madhya Pradesh",
    "24": "Gujarat",
    "25": "Dadra and Nagar Haveli and Daman and Diu",
    "27": "Maharashtra",
    "28": "Andhra Pradesh",
    "29": "Karnataka",
    "30": "Goa",
    "31": "Lakshadweep",
    "32": "Kerala",
    "33": "Tamil Nadu",
    "34": "Puducherry",
    "35": "Andaman and Nicobar Islands",
    "36": "Telangana",
    "37": "Ladakh",
    "99": "Other or unspecified state",
}

AILMENT = {
    "01": "Fever with loss of consciousness or altered consciousness",
    "02": "Malaria",
    "03": "Fever due to diphtheria or whooping cough",
    "04": "All other fevers",
    "05": "Tuberculosis",
    "06": "Filariasis",
    "07": "Tetanus",
    "08": "HIV or AIDS",
    "09": "Other sexually transmitted diseases",
    "10": "Jaundice",
    "11": "Diarrhoea or dysentery",
    "12": "Worm infestation",
    "13": "Cancer",
    "14": "Anaemia",
    "15": "Bleeding disorders",
    "16": "Sickle cell anaemia, thalassemia, or related condition",
    "17": "Diabetes",
    "18": "Under-nutrition",
    "19": "Goitre or other thyroid disease",
    "20": "Other endocrine, metabolic, or nutritional condition",
    "21": "Intellectual disability",
    "22": "Mental disorders",
    "23": "Headache",
    "24": "Seizures or epilepsy",
    "25": "Limb-muscle weakness or difficulty in movement",
    "26": "Stroke, hemiplegia, or sudden speech or movement loss",
    "27": "Other neurological condition",
    "28": "Eye discomfort, pain, redness, swelling, or boils",
    "29": "Cataract",
    "30": "Glaucoma",
    "31": "Chronic decreased vision not corrected by glasses",
    "32": "Other eye condition",
    "33": "Earache with discharge, bleeding, or infection",
    "34": "Decreased hearing or hearing loss",
    "35": "Hypertension",
    "36": "Heart disease, chest pain, or breathlessness",
    "37": "Acute upper respiratory infection",
    "38": "Cough with sputum, with or without fever, not diagnosed as TB",
    "39": "Bronchial asthma or recurrent wheezing and breathlessness",
    "40": "Disease of mouth, teeth, or gums",
    "41": "Abdominal pain, gastric or peptic ulcer, acid reflux, or acute abdomen",
    "42": "Lump or fluid in abdomen or scrotum",
    "43": "Gastrointestinal bleeding",
    "44": "Skin infection or other skin disease",
    "45": "Joint or bone disease, pain, or swelling",
    "46": "Back or body aches",
    "47": "Difficulty or abnormality in urination",
    "48": "Pelvic or reproductive-tract pain or infection",
    "49": "Menstrual, gynaecological, andrological, or infertility condition",
    "50": "Pregnancy complication before or during labour",
    "51": "Maternal complication after childbirth",
    "52": "Illness in newborn or infant",
    "53": "Accidental injury, road traffic accident, or fall",
    "54": "Accidental drowning or submersion",
    "55": "Burns or corrosions",
    "56": "Poisoning",
    "57": "Intentional self-harm",
    "58": "Assault",
    "59": "Contact with venomous or harm-causing animals or plants",
    "60": "Kidney failure",
    "61": "Symptom not fitting another category",
    "62": "Main symptom could not be stated",
    "87": "Normal delivery",
    "88": "Caesarean delivery",
    "89": "Other type of delivery",
}

TREATMENT_SYSTEM = {
    "1": "allopathy",
    "2": "ayush_or_traditional",
    "3": "allopathy_and_ayush",
    "9": "other",
}

MEDICAL_INSTITUTION = {
    "1": "government_or_public_hospital",
    "2": "charitable_trust_or_ngo_hospital",
    "3": "private_hospital",
}

WARD_TYPE = {
    "1": "free",
    "2": "paying_general",
    "3": "paying_special",
}

SERVICE_RECEIPT = {
    "1": "not_received",
    "2": "received_free",
    "3": "received_partly_free",
    "4": "received_on_payment",
}

FREE_MEDICAL_SERVICE = {
    "1": "yes_public_provider",
    "2": "yes_private_charitable_or_ngo_provider",
    "3": "yes_both_provider_types",
    "4": "no",
}

CODEBOOKS = {
    "sector": SECTOR,
    "gender": GENDER,
    "state": STATE,
    "ailment": AILMENT,
    "treatment_system": TREATMENT_SYSTEM,
    "medical_institution": MEDICAL_INSTITUTION,
    "ward_type": WARD_TYPE,
    "surgery": SERVICE_RECEIPT,
    "medicine": SERVICE_RECEIPT,
    "imaging": SERVICE_RECEIPT,
    "other_diagnostics": SERVICE_RECEIPT,
    "free_medical_service": FREE_MEDICAL_SERVICE,
}


def allowed_values() -> dict[str, list[str]]:
    """Return stable human-readable categories accepted by the prediction API."""

    return {
        feature: sorted(set(mapping.values()))
        for feature, mapping in CODEBOOKS.items()
    }
