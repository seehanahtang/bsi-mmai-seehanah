"""Utilities for truncating clinical notes to risk-factor-relevant content."""

import re
from typing import List

# =============================================================================
# Configuration: relevance keywords and filtering patterns
# =============================================================================

# Risk factors vocabulary (copied from text_extractor for local use)
RISK_FACTORS: List[str] = [
    "IMMUNOCOMPROMISED STATE",
    "POST TRANSPLANT STATUS",
    "HIV",
    "ACTIVE CHEMOTHERAPY",
    "NEUTROPENIA",
    "LONG TERM STEROID OR IMMUNOMODULATOR USE",
    "CHRONIC HEMODIALYSIS",
    "LIVER CIRRHOSIS OR FAILURE",
    "DIABETES MELLITUS",
    "CHRONIC KIDNEY DISEASE",
    "CHRONIC LUNG DISEASE",
    "MALNUTRITION",
    "AGE OVER 65",
    "CENTRAL LINE OR PICC",
    "FOLEY CATHETER",
    "SURGICAL OR PERCUTANEOUS DRAINS",
    "IMPLANTED HARDWARE",
    "RECENT SURGERY",
    "HEART VALVE DISEASE OR PROSTHETIC VALVE",
    "RECENT DENTAL PROCEDURE OR DENTAL INFECTION",
    "SKIN BREAKDOWN OR OPEN WOUND",
    "MUCOSAL BREAKDOWN",
    "GASTROINTESTINAL ISCHEMIA OR INFECTION",
    "GALLBLADDER OR KIDNEY STONES WITH OBSTRUCTION",
    "FEVER",
    "HYPOTENSION",
    "TACHYCARDIA",
    "ELEVATED LACTATE",
    "ALTERED MENTAL STATUS",
    "INTRAVENOUS DRUG USE",
    "TOTAL PARENTERAL NUTRITION",
    "RECENT HEALTHCARE EXPOSURE",
    "RECENT TRAVEL",
    "INFECTIOUS EXPOSURE RISK",
    "PREVIOUS INFECTION",
]

# Build keyword set for relevance filtering
RELEVANCE_KEYWORDS = set()
for factor in RISK_FACTORS:
    RELEVANCE_KEYWORDS.add(factor.lower())

RELEVANCE_KEYWORDS.update(
    {
        # Conditions
        "immunocompromised",
        "transplant",
        "hiv",
        "aids",
        "chemotherapy",
        "chemo",
        "neutropenic",
        "neutrophil",
        "steroid",
        "prednisone",
        "immunosuppressant",
        "dialysis",
        "hemodialysis",
        "cirrhosis",
        "hepatic",
        "liver",
        "diabetes",
        "diabetic",
        "a1c",
        "glucose",
        "kidney",
        "renal",
        "ckd",
        "esrd",
        "copd",
        "pulmonary",
        "malnutrition",
        "albumin",
        "frail",
        "elderly",
        # Lines/devices
        "central line",
        "picc",
        "catheter",
        "foley",
        "drain",
        "hardware",
        "prosthetic",
        "implant",
        "pacemaker",
        "valve",
        "endocarditis",
        "dental",
        "abscess",
        # Barriers
        "wound",
        "ulcer",
        "decubitus",
        "pressure sore",
        "cellulitis",
        "skin",
        "mucositis",
        "oral",
        "gi bleed",
        "colitis",
        "ischemia",
        "bowel",
        "cholecystitis",
        "cholelithiasis",
        "nephrolithiasis",
        "obstruction",
        # Vitals/labs
        "fever",
        "febrile",
        "temperature",
        "hypothermia",
        "hypotension",
        "sepsis",
        "septic",
        "shock",
        "tachycardia",
        "heart rate",
        "lactate",
        "lactic",
        "altered",
        "mental status",
        "confusion",
        "encephalopathy",
        "ams",
        # Exposures
        "ivdu",
        "iv drug",
        "injection drug",
        "tpn",
        "parenteral nutrition",
        "nursing home",
        "snf",
        "travel",
        "exposure",
        # Blood culture specific / clinical suspicion
        "blood culture",
        "bacteremia",
        "positive culture",
        "gram stain",
        "antibiotic",
        "infection",
        "infectious",
        "concern for infection",
        "suspected infection",
        "possible infection",
        "rule out sepsis",
        "rule out bacteremia",
        "source of infection",
        "infectious workup",
        "blood cultures drawn",
        "cultures pending",
    }
)

# Patterns for filtering out irrelevant sentences
EMPTY_PATTERNS = [
    "not on file",
    "not documented",
    "no history on file",
    "not recorded",
    "none documented",
    "none reported",
    "none known",
    "no known",
    "not available",
    "not assessed",
    "not applicable",
    "not obtained",
    "unable to obtain",
    "unable to assess",
]

FIRST_PERSON_PATTERNS = [
    r"\bI\s+reviewed",
    r"\bI\s+personally",
    r"\bI\s+spoke",
    r"\bI\s+discussed",
    r"\bI\s+have\s+reviewed",
    r"\bI\s+spent",
    r"\bI\s+supervised",
    r"\bI\s+provided",
    r"\bI\s+saw",
    r"\bI\s+examined",
    r"\bI\s+evaluated",
    r"\bI\s+performed",
    r"\bI\s+agree",
    r"\bI\s+attest",
    r"\bI\s+confirm",
    r"\bMy\s+note",
    r"\bMy\s+findings",
    r"^I\s+",
]

BOILERPLATE_SECTIONS = [
    "socioeconomic history",
    "occupational history",
    "tobacco use",
    "smokeless tobacco",
    "smoking status",
    "substance and sexual activity",
    "substance use topics",
    "vaping use",
    "other topics concern",
    "social drivers of health",
    "financial resource strain",
    "food insecurity",
    "transportation needs",
    "housing stability",
    "social connections",
    "physical activity:",
    "stress:",
    "sexual activity:",
    "marital status:",
    "spouse name:",
    "number of children:",
    "years of education:",
    "highest education level:",
    "alcohol use:",
    "drug use:",
    "social history narrative",
]

ATTESTATION_PATTERNS = [
    "attending attestation",
    "resident attestation",
    "personally saw the patient",
    "substantive portion",
    "take responsibility",
    "reviewed the resident",
    "approved the management",
]


def _should_skip_sentence(sentence: str, sentence_lower: str) -> bool:
    """Return True if a sentence should be filtered out as boilerplate/irrelevant."""
    for pattern in EMPTY_PATTERNS:
        if pattern in sentence_lower:
            return True

    for pattern in FIRST_PERSON_PATTERNS:
        if re.search(pattern, sentence, re.IGNORECASE):
            return True

    for section in BOILERPLATE_SECTIONS:
        if section in sentence_lower:
            return True

    for pattern in ATTESTATION_PATTERNS:
        if pattern in sentence_lower:
            return True

    return False


def truncate_notes_by_relevance(notes: str, max_sentences: int = 50) -> str:
    """Truncate clinical notes to keep only sentences relevant to risk factors."""
    if not notes or not notes.strip():
        return ""

    sentences = re.split(
        r"(?<=[.!?\n])\s+|(?=\n[A-Z]{2,}:)|(?=\n\d+\.)",
        notes,
    )
    relevant: List[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_lower = sentence.lower()
        if _should_skip_sentence(sentence, sentence_lower):
            continue

        if any(kw in sentence_lower for kw in RELEVANCE_KEYWORDS):
            relevant.append(sentence)
            if len(relevant) >= max_sentences:
                break

    return " ".join(relevant)

