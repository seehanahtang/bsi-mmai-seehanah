#!/usr/bin/env python3
"""
LLM-based BSI risk factor extraction from tabular + clinical notes data.

For each patient encounter, MedGemma-27B-it separately identifies BSI risk
factors from:
  1. Structured tabular data (vitals, labs, ICD diagnoses, chief complaints,
     medications)
  2. Free-text clinical notes (CombinedNotes column)

Supports base_HH_2025.csv and base_HH_before_2025.csv (same schema).
Multiple --data paths are accepted; each dataset is tagged with a
dataset_source column in the output.

Usage:
    python extract_risk_factors.py --model medgemma-27b --hf_token $HF_TOKEN
    python extract_risk_factors.py --model medgemma-27b \\
        --data /path/to/base_HH_2025.csv /path/to/base_HH_before_2025.csv
    python extract_risk_factors.py --model medgemma-27b --dry_run --n_patients 5
"""

import argparse
import fcntl
import gc
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

# ── Import shared utilities from notes_summary ────────────────────────────────

NOTES_SUMMARY_DIR = Path(__file__).parent.parent / "notes_summary"
sys.path.insert(0, str(NOTES_SUMMARY_DIR))

from summarize_notes import (
    MODEL_REGISTRY,
    login_huggingface,
    load_model,
    truncate_notes_to_fit,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = Path(__file__).parent
DEFAULT_DATA  = [str(SCRIPT_DIR / ".." / "data" / "base_HH_2025.csv")]
DEFAULT_OUT   = SCRIPT_DIR / "data"

STOP_TOKEN = "<<<END>>>"

# ── Column definitions (mirrors predict_bsi.py) ───────────────────────────────

VITALS_COLS = ["Temp", "SpO2", "Pulse", "Resp",
               "HighFever", "LowFever", "HighPulse", "LowOxygen"]

LAB_COLS = [
    "lab_white blood cell",
    "lab_lactic acid",
    "lab_creatinine",
    "lab_abs neutrophils",
    "lab_INR",
    "lab_platelet",
    "lab_albumin",
    "lab_bilirubin",
    "lab_Alanine Aminotrans",
    "lab_Aspartate Aminotrans",
    "lab_Blood Urea Nitrogen",
    "lab_abs lymphocyte",
    "lab_P/F ratio",
]

# ── Patient text builder (same as predict_bsi.py) ─────────────────────────────

def _format_binary_cols(row: pd.Series, prefix: str) -> str:
    parts = []
    for col in row.index:
        if col.startswith(prefix):
            try:
                val = row[col]
                if pd.notna(val) and float(val) >= 1:
                    name = col[len(prefix):].replace("_", " ")
                    name = re.sub(r"\(?\s*ICD-10-CM.*", "", name).strip()
                    parts.append(name.lower())
            except (ValueError, TypeError):
                pass
    return ", ".join(parts)


def _format_vitals(row: pd.Series, cols: List[str]) -> str:
    parts = []
    labels = {
        "Temp":      "Temperature",
        "SpO2":      "SpO2",
        "Pulse":     "Heart rate",
        "Resp":      "Resp rate",
        "HighFever": "High fever flag",
        "LowFever":  "Low fever flag",
        "HighPulse": "High pulse flag",
        "LowOxygen": "Low oxygen flag",
    }
    for col in cols:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                parts.append(f"{labels.get(col, col)}: {val}")
    return "; ".join(parts) if parts else "Not available"


def _format_labs(row: pd.Series, lab_cols: List[str]) -> str:
    parts = []
    label_map = {
        "lab_white blood cell":    "WBC",
        "lab_lactic acid":         "Lactate",
        "lab_creatinine":          "Creatinine",
        "lab_abs neutrophils":     "ANC",
        "lab_INR":                 "INR",
        "lab_platelet":            "Platelets",
        "lab_albumin":             "Albumin",
        "lab_bilirubin":           "Bilirubin",
        "lab_Alanine Aminotrans":  "ALT",
        "lab_Aspartate Aminotrans":"AST",
        "lab_Blood Urea Nitrogen": "BUN",
        "lab_abs lymphocyte":      "ALC",
        "lab_P/F ratio":           "P/F ratio",
    }
    for col in lab_cols:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and val != 0:
                parts.append(f"{label_map.get(col, col.replace('lab_', ''))}: {val}")
    return "; ".join(parts) if parts else "Not available"


def build_patient_text(row: pd.Series, include_notes: bool = True) -> str:
    """
    Build structured patient summary.  The prompt separates the tabular
    section from the clinical notes section so the model can attribute each
    risk factor to its source.
    """
    parts = []

    # Structured / tabular data
    parts.append("=== STRUCTURED DATA ===")
    parts.append(f"Vitals: {_format_vitals(row, VITALS_COLS)}.")
    parts.append(f"Laboratory values: {_format_labs(row, LAB_COLS)}.")

    icd = _format_binary_cols(row, "icd_")
    if icd:
        parts.append(f"Past/active diagnoses (ICD-coded): {icd}.")

    complaints = _format_binary_cols(row, "cmp_")
    if complaints:
        parts.append(f"Chief complaints: {complaints}.")

    meds = _format_binary_cols(row, "rx_")
    if meds:
        parts.append(f"Medications: {meds}.")

    # Free-text clinical notes
    if include_notes and "CombinedNotes" in row.index:
        notes = str(row["CombinedNotes"]) if pd.notna(row["CombinedNotes"]) else ""
        if notes and notes.lower() != "nan":
            parts.append("\n=== CLINICAL NOTES ===")
            parts.append(notes)

    return "\n".join(parts)


# ── Prompt ────────────────────────────────────────────────────────────────────

_FEW_SHOT_EXAMPLE = """\
EXAMPLE PATIENT:
=== STRUCTURED DATA ===
Vitals: Temperature: 38.9; Heart rate: 118; SpO2: 94; Resp rate: 22; High fever flag: 1.0; High pulse flag: 1.0.
Laboratory values: WBC: 18.4; Lactate: 3.2; Creatinine: 1.8; ANC: 16.2; Platelets: 88.
Past/active diagnoses (ICD-coded): end stage renal disease, type 2 diabetes mellitus.
Chief complaints: fever, altered mental status.
Medications: insulin, erythropoietin, phosphate binders.

=== CLINICAL NOTES ===
68 yo M on chronic HD presenting with fever to 38.9 and altered mental status. Has tunneled HD catheter in R IJ placed 3 months ago. Blood cultures drawn — two sets sent. History of prior MRSA bacteremia 1 year ago.

EXAMPLE OUTPUT:
{
  "risk_factors_tabular": [
    "fever (temperature 38.9°C, high fever flag positive)",
    "tachycardia (heart rate 118, high pulse flag positive)",
    "mild hypoxemia (SpO2 94%)",
    "leukocytosis (WBC 18.4)",
    "elevated lactate (3.2 — tissue hypoperfusion)",
    "thrombocytopenia (platelets 88)",
    "elevated creatinine (1.8 — renal insufficiency)",
    "end-stage renal disease (ICD diagnosis — immunocompromise, vascular access dependence)",
    "type 2 diabetes mellitus (ICD diagnosis — impaired host defences)",
    "altered mental status (chief complaint — possible sepsis encephalopathy)"
  ],
  "risk_factors_notes": [
    "tunneled HD catheter right internal jugular placed 3 months ago — intravascular device, portal of entry",
    "prior MRSA bacteremia 1 year ago — elevated recurrence risk",
    "blood cultures drawn — clinician concern for BSI documented"
  ],
  "risk_factors_all": [
    "fever and tachycardia (systemic inflammatory response)",
    "leukocytosis with elevated lactate (early sepsis markers)",
    "thrombocytopenia (possible consumptive process)",
    "ESRD and diabetes (underlying immunocompromise)",
    "tunneled hemodialysis catheter (intravascular device — portal of entry)",
    "prior MRSA bacteremia (recurrence risk)",
    "altered mental status (possible sepsis encephalopathy)"
  ],
  "risk_summary": "This patient has multiple convergent BSI risk factors. Structured data reveals a systemic inflammatory response (fever, tachycardia, leukocytosis) with early sepsis markers (elevated lactate, thrombocytopenia) and immunocompromising comorbidities (ESRD, T2DM). Clinical notes add a critical device-related risk: a tunneled HD catheter as a likely portal of entry, compounded by a documented prior MRSA bacteremia that significantly elevates recurrence probability. Overall risk profile is high."
}
<<<END>>>"""


def build_extraction_prompt(patient_text: str) -> str:
    """
    Chain-of-thought prompt instructing the model to separately enumerate
    risk factors from structured tabular data and from clinical notes.
    """
    return f"""You are an expert infectious disease clinician. Your task is to extract all clinical risk factors for blood stream infection (BSI/bacteremia) from a patient's record.

The patient record is divided into two clearly labelled sections:
  1. STRUCTURED DATA — vitals, lab values, coded ICD diagnoses, chief complaints, medications.
  2. CLINICAL NOTES — free-text nursing/physician documentation.

You must extract risk factors from EACH section separately, then produce a combined list and a brief narrative summary.

{_FEW_SHOT_EXAMPLE}

--------------------------------------------------

NOW EVALUATE THIS PATIENT:

{patient_text}

--------------------------------------------------

INSTRUCTIONS — follow these steps in order:
Step 1: List up to 10 BSI risk factors found ONLY in the STRUCTURED DATA section (abnormal vitals, lab derangements, immunocompromising diagnoses, high-risk medications, relevant chief complaints). Be specific — include the actual value or ICD term.
Step 2: List up to 10 BSI risk factors found ONLY in the CLINICAL NOTES section (intravascular devices, procedures, microbiological history, clinical context not captured in structured fields).
Step 3: Produce a deduplicated combined list of the most important risk factors (up to 10 items).
Step 4: Write 2–4 sentences summarising the patient's overall BSI risk profile, noting which source (structured data vs notes) contributed the most informative risk factors.

OUTPUT FORMAT — return ONLY valid JSON followed immediately by {STOP_TOKEN}. No text before or after.

{{
  "risk_factors_tabular": ["finding1 (value/context)", "finding2"],
  "risk_factors_notes": ["finding1", "finding2"],
  "risk_factors_all": ["top finding1", "top finding2"],
  "risk_summary": "Narrative summary here."
}}
{STOP_TOKEN}

OUTPUT:
"""


# ── Generation config ─────────────────────────────────────────────────────────

@dataclass
class ExtractionGenerationConfig:
    """Generation config for structured risk factor extraction with CoT."""
    max_new_tokens: int = 700
    min_new_tokens: int = 10
    temperature: float = 0.3
    top_p: float = 0.9
    stream: bool = True
    verbose: bool = False
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.0

    stop_sequences: List[str] = field(default_factory=lambda: [
        STOP_TOKEN,
        "<<<END>>>",
        "NOW EVALUATE THIS PATIENT:",
    ])


# ── JSON parsing ──────────────────────────────────────────────────────────────

def parse_extraction_json(raw: str) -> Dict:
    """
    Parse LLM output into a validated risk-factor extraction dict.

    Expected keys: risk_factors_tabular (list), risk_factors_notes (list),
                   risk_factors_all (list), risk_summary (str).

    Falls back to partial extraction on parse failure.
    """
    cleaned = raw.strip().replace(STOP_TOKEN, "").strip()

    for text in [cleaned, _fix_common_json_issues(cleaned)]:
        try:
            obj = json.loads(text)
            return _validate_extraction(obj)
        except (json.JSONDecodeError, ValueError):
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                obj = json.loads(match.group())
                return _validate_extraction(obj)
            except (json.JSONDecodeError, ValueError):
                pass

    partial = _recover_extraction_fields(cleaned)
    if partial is not None:
        return partial

    raise ValueError(f"Could not parse extraction JSON: {raw[:300]}")


def _fix_common_json_issues(text: str) -> str:
    fixed = re.sub(r",\s*}", "}", text)
    fixed = re.sub(r",\s*]", "]", fixed)
    return fixed


def _validate_extraction(obj: dict) -> Dict:
    if not isinstance(obj, dict):
        raise ValueError("Not a dict")
    return {
        "risk_factors_tabular": _coerce_list(obj.get("risk_factors_tabular")),
        "risk_factors_notes":   _coerce_list(obj.get("risk_factors_notes")),
        "risk_factors_all":     _coerce_list(obj.get("risk_factors_all")),
        "risk_summary":         str(obj.get("risk_summary", "")),
    }


def _coerce_list(val) -> List[str]:
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        return [val] if val else []
    return []


def _recover_extraction_fields(text: str) -> Optional[Dict]:
    """Last-resort: pull out whatever lists and summary text we can find."""
    def extract_list(key: str) -> List[str]:
        m = re.search(rf'"{key}"\s*:\s*\[([^\]]*)\]', text, re.DOTALL)
        if not m:
            return []
        items = re.findall(r'"([^"]+)"', m.group(1))
        return items

    tabular = extract_list("risk_factors_tabular")
    notes   = extract_list("risk_factors_notes")
    all_rf  = extract_list("risk_factors_all")
    summary_m = re.search(r'"risk_summary"\s*:\s*"([^"]*)"', text)
    summary = summary_m.group(1) if summary_m else ""

    if tabular or notes or all_rf:
        return {
            "risk_factors_tabular": tabular,
            "risk_factors_notes":   notes,
            "risk_factors_all":     all_rf,
            "risk_summary":         summary,
        }
    return None


# ── Text generation ───────────────────────────────────────────────────────────

def generate_extraction(
    model,
    tokenizer,
    patient_text: str,
    gen_config: ExtractionGenerationConfig,
    model_type: str,
    context_length: int,
) -> str:
    """Generate structured risk factor extraction for a single patient."""
    from transformers import TextIteratorStreamer

    # Truncate notes to fit context window.
    # Prompt overhead (instructions + example) is ~1100 tokens.
    truncated_text = truncate_notes_to_fit(
        patient_text, tokenizer, context_length,
        prompt_overhead_tokens=1100,
        max_new_tokens=gen_config.max_new_tokens,
    )

    raw_prompt = build_extraction_prompt(truncated_text)

    if model_type == "gemma":
        messages = [{"role": "user", "content": raw_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    elif model_type == "qwen":
        messages = [{"role": "user", "content": raw_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        prompt = raw_prompt

    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        **inputs,
        "max_new_tokens":     gen_config.max_new_tokens,
        "min_new_tokens":     gen_config.min_new_tokens,
        "do_sample":          gen_config.do_sample,
        "num_beams":          gen_config.num_beams,
        "repetition_penalty": gen_config.repetition_penalty,
        "pad_token_id":       tokenizer.eos_token_id or 0,
    }
    if gen_config.do_sample:
        gen_kwargs["temperature"] = gen_config.temperature
        gen_kwargs["top_p"]       = gen_config.top_p

    if gen_config.stream:
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        gen_kwargs["streamer"] = streamer

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        generated = ""
        for chunk in streamer:
            generated += chunk
            if gen_config.verbose:
                print(chunk, end="", flush=True)
            if STOP_TOKEN in generated:
                break
        thread.join()
        if gen_config.verbose:
            print()
    else:
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        generated = tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True,
        )

    return generated


# ── Progress tracking & resume ────────────────────────────────────────────────

OUTPUT_COLUMNS = [
    "primarymrn", "EncounterKey", "Positive", "dataset_source",
    "risk_factors_tabular", "risk_factors_notes", "risk_factors_all",
    "risk_summary", "parse_success", "notes_length", "model",
]


def load_existing_results(output_file: str) -> pd.DataFrame:
    if os.path.exists(output_file):
        return pd.read_csv(output_file)
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def append_results(output_file: str, new_rows: List[dict]):
    if not new_rows:
        return
    lock_file = output_file + ".lock"
    with open(lock_file, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            new_df = pd.DataFrame(new_rows)
            if os.path.exists(output_file):
                existing = pd.read_csv(output_file)
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df
            combined.to_csv(output_file, index=False)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# ── Main inference loop ───────────────────────────────────────────────────────

def run_inference(
    df: pd.DataFrame,
    model,
    tokenizer,
    gen_config: ExtractionGenerationConfig,
    model_name: str,
    model_type: str,
    context_length: int,
    output_file: str,
    save_every: int = 10,
    dry_run: bool = False,
    reverse: bool = False,
    chunk: Optional[int] = None,
    n_chunks: Optional[int] = None,
):
    existing = load_existing_results(output_file)
    done_encounters = set(existing["EncounterKey"].astype(str).values)
    logger.info(f"  {len(done_encounters):,} encounters already processed, skipping.")

    remaining = df[~df["EncounterKey"].astype(str).isin(done_encounters)]
    if reverse:
        remaining = remaining.iloc[::-1]
        logger.info("  Processing encounters in reverse order (--reverse).")
    if chunk is not None and n_chunks is not None:
        remaining = remaining.iloc[chunk::n_chunks]
        logger.info(f"  Chunk {chunk}/{n_chunks}: {len(remaining):,} encounters to process.")
    if dry_run:
        logger.info("DRY RUN — no model calls, testing text builder only.")

    buffer = []
    parse_failures = 0
    pbar = tqdm(remaining.iterrows(), total=len(remaining), desc="Extracting risk factors")

    for _, row in pbar:
        encounter_key  = str(row["EncounterKey"])
        mrn            = str(row.get("primarymrn", ""))
        label          = row.get("Positive", None)
        dataset_source = str(row.get("_dataset_source", "unknown"))
        notes          = str(row["CombinedNotes"]) if "CombinedNotes" in row.index and pd.notna(row["CombinedNotes"]) else ""
        notes_length   = len(notes)

        pbar.set_postfix(ek=encounter_key, notes_len=notes_length, fails=parse_failures)

        patient_text = build_patient_text(row)

        if dry_run:
            buffer.append({
                "primarymrn":          mrn,
                "EncounterKey":        encounter_key,
                "Positive":            label,
                "dataset_source":      dataset_source,
                "risk_factors_tabular": "[]",
                "risk_factors_notes":  "[]",
                "risk_factors_all":    "[]",
                "risk_summary":        "[DRY RUN]",
                "parse_success":       True,
                "notes_length":        notes_length,
                "model":               model_name,
            })
            done_encounters.add(encounter_key)
            if len(buffer) >= save_every:
                append_results(output_file, buffer)
                buffer = []
            continue

        try:
            raw_output = generate_extraction(
                model, tokenizer, patient_text, gen_config, model_type, context_length,
            )

            try:
                result = parse_extraction_json(raw_output)
                parse_success = True
            except ValueError as e:
                logger.warning(f"Parse failure for {encounter_key}: {e}")
                result = {
                    "risk_factors_tabular": [],
                    "risk_factors_notes":   [],
                    "risk_factors_all":     [],
                    "risk_summary":         raw_output.replace(STOP_TOKEN, "").strip(),
                }
                parse_success = False
                parse_failures += 1

            buffer.append({
                "primarymrn":          mrn,
                "EncounterKey":        encounter_key,
                "Positive":            label,
                "dataset_source":      dataset_source,
                "risk_factors_tabular": json.dumps(result["risk_factors_tabular"]),
                "risk_factors_notes":  json.dumps(result["risk_factors_notes"]),
                "risk_factors_all":    json.dumps(result["risk_factors_all"]),
                "risk_summary":        result["risk_summary"],
                "parse_success":       parse_success,
                "notes_length":        notes_length,
                "model":               model_name,
            })

        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM for encounter {encounter_key} (notes_len={notes_length}). Skipping.")
            torch.cuda.empty_cache()
            buffer.append({
                "primarymrn":          mrn,
                "EncounterKey":        encounter_key,
                "Positive":            label,
                "dataset_source":      dataset_source,
                "risk_factors_tabular": "[]",
                "risk_factors_notes":  "[]",
                "risk_factors_all":    "[]",
                "risk_summary":        "ERROR: OOM",
                "parse_success":       False,
                "notes_length":        notes_length,
                "model":               model_name,
            })
        except Exception as e:
            logger.error(f"Error for encounter {encounter_key}: {e}")
            buffer.append({
                "primarymrn":          mrn,
                "EncounterKey":        encounter_key,
                "Positive":            label,
                "dataset_source":      dataset_source,
                "risk_factors_tabular": "[]",
                "risk_factors_notes":  "[]",
                "risk_factors_all":    "[]",
                "risk_summary":        f"ERROR: {e}",
                "parse_success":       False,
                "notes_length":        notes_length,
                "model":               model_name,
            })

        done_encounters.add(encounter_key)

        if len(buffer) >= save_every:
            append_results(output_file, buffer)
            buffer = []

    if buffer:
        append_results(output_file, buffer)

    logger.info(f"Inference complete. Results → {output_file}")
    logger.info(f"Parse failures: {parse_failures}/{len(remaining)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-based BSI risk factor extraction from tabular + notes data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()), required=True,
        help="Model to use for extraction",
    )
    parser.add_argument(
        "--data", type=str, nargs="+", default=DEFAULT_DATA,
        help=(
            "One or more input CSV paths (same schema as base_HH_2025.csv). "
            "Each file is tagged with its filename in dataset_source column. "
            "Example: --data base_HH_2025.csv base_HH_before_2025.csv"
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUT),
        help="Directory for extraction output CSV",
    )
    parser.add_argument(
        "--output_name", type=str, default=None,
        help=(
            "Output CSV filename (without directory). "
            "Defaults to bsi_risk_factors_<model>.csv"
        ),
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token for gated models",
    )
    parser.add_argument(
        "--quantize_4bit", action="store_true", default=False,
        help="Load model in 4-bit NF4 quantization (~14 GB VRAM vs ~54 GB bfloat16)",
    )
    parser.add_argument(
        "--save_every", type=int, default=10,
        help="Checkpoint every N patients",
    )
    parser.add_argument(
        "--n_patients", type=int, default=None,
        help="Limit to first N patients per dataset (for testing)",
    )
    parser.add_argument(
        "--dry_run", action="store_true", default=False,
        help="Test text builder without loading the model",
    )
    parser.add_argument(
        "--retry_failures", action="store_true", default=False,
        help="Re-run only encounters where parse_success=False",
    )
    parser.add_argument(
        "--reverse", action="store_true", default=False,
        help="Process encounters in descending order (for a second parallel job)",
    )
    parser.add_argument(
        "--chunk", type=int, default=None,
        help="0-indexed chunk to process (use with --n_chunks / SLURM array jobs).",
    )
    parser.add_argument(
        "--n_chunks", type=int, default=None,
        help="Total number of parallel chunks.",
    )

    gen = parser.add_argument_group("Generation parameters")
    gen.add_argument("--max_new_tokens", type=int, default=700)
    gen.add_argument("--min_new_tokens", type=int, default=10)
    gen.add_argument("--temperature", type=float, default=0.3)
    gen.add_argument("--top_p", type=float, default=0.9)
    gen.add_argument("--no_stream", action="store_true", default=False)
    gen.add_argument("--verbose", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load and tag all datasets ─────────────────────────────────────────────
    dfs = []
    for data_path in args.data:
        if not os.path.exists(data_path):
            logger.warning(f"Dataset not found, skipping: {data_path}")
            continue
        logger.info(f"Loading dataset: {data_path}")
        df_part = pd.read_csv(data_path, low_memory=False)
        df_part["EncounterKey"] = df_part["EncounterKey"].astype(str)
        df_part["_dataset_source"] = Path(data_path).name
        if args.n_patients is not None:
            df_part = df_part.head(args.n_patients)
            logger.info(f"  Limited to {len(df_part)} patients (--n_patients)")
        else:
            logger.info(f"  {len(df_part):,} encounters loaded")
        dfs.append(df_part)

    if not dfs:
        logger.error("No valid dataset files found. Exiting.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total encounters across all datasets: {len(df):,}")

    # ── Output file ───────────────────────────────────────────────────────────
    out_name = args.output_name or f"bsi_risk_factors_{args.model}.csv"
    output_file = os.path.join(args.output_dir, out_name)

    # ── Retry-failure mode ────────────────────────────────────────────────────
    if args.retry_failures and os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        n_before = len(existing)
        kept = existing[existing["parse_success"] == True]
        n_dropped = n_before - len(kept)
        if n_dropped > 0:
            kept.to_csv(output_file, index=False)
            logger.info(f"Retry mode: dropped {n_dropped} failed rows from {output_file}")
        else:
            logger.info("Retry mode: no failed rows found, running normally.")

    # ── Generation config ─────────────────────────────────────────────────────
    gen_config = ExtractionGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=not args.no_stream,
        verbose=args.verbose,
        do_sample=True,
    )

    # ── Model load ────────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("DRY RUN mode — skipping model load")
        model, tokenizer = None, None
        model_type = "gemma"
        context_length = MODEL_REGISTRY["medgemma-27b"]["context_length"]
    else:
        if args.hf_token:
            login_huggingface(args.hf_token)
        model_cfg = MODEL_REGISTRY[args.model]
        model, tokenizer = load_model(args.model, args.hf_token, quantize_4bit=args.quantize_4bit)
        model_type     = model_cfg["model_type"]
        context_length = model_cfg["context_length"]

    # ── Inference ─────────────────────────────────────────────────────────────
    run_inference(
        df=df,
        model=model,
        tokenizer=tokenizer,
        gen_config=gen_config,
        model_name=args.model,
        model_type=model_type,
        context_length=context_length,
        output_file=output_file,
        save_every=args.save_every,
        dry_run=args.dry_run,
        reverse=args.reverse,
        chunk=args.chunk,
        n_chunks=args.n_chunks,
    )

    if model is not None:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()
