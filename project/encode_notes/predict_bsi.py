#!/usr/bin/env python3
"""
End-to-end LLM prediction of positive BSI using tabular + clinical notes data.

Uses MedGemma-27B-it with a chain-of-thought prompt:
  1. Patient context built from vitals, labs, ICD diagnoses, chief complaints,
     medications, and free-text clinical notes.
  2. Model enumerates evidence for/against BSI, reasons through it, then
     outputs a binary prediction and confidence score in JSON.

Reuses model loading infrastructure from haim/notes_summary/summarize_notes.py.

Usage:
    python predict_bsi.py --model medgemma-27b --hf_token $HF_TOKEN
    python predict_bsi.py --model medgemma-27b --quantize_4bit --hf_token $HF_TOKEN
    python predict_bsi.py --model medgemma-27b --dry_run --n_patients 5
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
from typing import Dict, List, Optional, Tuple

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

SCRIPT_DIR   = Path(__file__).parent
DEFAULT_DATA = SCRIPT_DIR / ".." / "data" / "base_HH_2025.csv"
DEFAULT_OUT  = SCRIPT_DIR / "data"

STOP_TOKEN = "<<<END>>>"

# ── Column definitions ────────────────────────────────────────────────────────

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

# ── Patient text builder ──────────────────────────────────────────────────────

def _format_binary_cols(row: pd.Series, prefix: str) -> str:
    """Return comma-separated names of binary indicator columns with value >= 1."""
    parts = []
    for col in row.index:
        if col.startswith(prefix):
            try:
                val = row[col]
                if pd.notna(val) and float(val) >= 1:
                    name = col[len(prefix):].replace("_", " ")
                    # Strip leftover ICD-10-CM parenthetical
                    name = re.sub(r"\(?\s*ICD-10-CM.*", "", name).strip()
                    parts.append(name.lower())
            except (ValueError, TypeError):
                pass
    return ", ".join(parts)


def _format_vitals(row: pd.Series, cols: List[str]) -> str:
    """Return a readable vitals string, skipping NaN values."""
    parts = []
    labels = {
        "Temp":       "Temperature",
        "SpO2":       "SpO2",
        "Pulse":      "Heart rate",
        "Resp":       "Resp rate",
        "HighFever":  "High fever flag",
        "LowFever":   "Low fever flag",
        "HighPulse":  "High pulse flag",
        "LowOxygen":  "Low oxygen flag",
    }
    for col in cols:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                label = labels.get(col, col)
                parts.append(f"{label}: {val}")
    return "; ".join(parts) if parts else "Not available"


def _format_labs(row: pd.Series, lab_cols: List[str]) -> str:
    """Return a readable lab values string, skipping NaN and zero values."""
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
                label = label_map.get(col, col.replace("lab_", ""))
                parts.append(f"{label}: {val}")
    return "; ".join(parts) if parts else "Not available"


def build_patient_text(row: pd.Series, include_notes: bool = True) -> str:
    """
    Build a structured patient summary from tabular + notes data.

    Sections: Vitals → Labs → ICD diagnoses → Chief complaints →
              Medications → Clinical notes
    """
    parts = []

    # Vitals
    vitals_str = _format_vitals(row, VITALS_COLS)
    parts.append(f"Vitals: {vitals_str}.")

    # Labs
    labs_str = _format_labs(row, LAB_COLS)
    parts.append(f"Laboratory values: {labs_str}.")

    # ICD diagnoses
    icd = _format_binary_cols(row, "icd_")
    if icd:
        parts.append(f"Past/active diagnoses (ICD-coded): {icd}.")

    # Chief complaints
    complaints = _format_binary_cols(row, "cmp_")
    if complaints:
        parts.append(f"Chief complaints: {complaints}.")

    # Medications
    meds = _format_binary_cols(row, "rx_")
    if meds:
        parts.append(f"Medications: {meds}.")

    # Notes
    if include_notes and "CombinedNotes" in row.index:
        notes = str(row["CombinedNotes"]) if pd.notna(row["CombinedNotes"]) else ""
        if notes and notes.lower() != "nan":
            parts.append(f"\nClinical notes:\n{notes}")

    return "\n".join(parts)


# ── Prompt ────────────────────────────────────────────────────────────────────

_FEW_SHOT_EXAMPLE = """\
EXAMPLE PATIENT:
Vitals: Temperature: 38.9; Heart rate: 118; SpO2: 94; Resp rate: 22; High fever flag: 1.0; High pulse flag: 1.0.
Laboratory values: WBC: 18.4; Lactate: 3.2; Creatinine: 1.8; ANC: 16.2; Platelets: 88.
Past/active diagnoses (ICD-coded): end stage renal disease, type 2 diabetes mellitus.
Chief complaints: fever, altered mental status.
Medications: insulin, erythropoietin, phosphate binders.

Clinical notes:
68 yo M on chronic HD presenting with fever to 38.9 and altered mental status. Has tunneled HD catheter in R IJ placed 3 months ago. Blood cultures drawn — two sets sent. History of prior MRSA bacteremia 1 year ago.

EXAMPLE OUTPUT:
{
  "bsi_indicators": [
    "fever (38.9°C) with tachycardia (HR 118) and elevated lactate (3.2)",
    "tunneled HD catheter (high-risk intravascular device)",
    "altered mental status suggesting possible systemic infection",
    "leukocytosis (WBC 18.4) with thrombocytopenia (platelets 88)",
    "prior MRSA bacteremia — known risk factor for recurrence"
  ],
  "against_bsi": [
    "no hypotension documented",
    "SpO2 only mildly reduced — may be baseline for ESRD"
  ],
  "reasoning": "This patient presents with multiple high-risk features for BSI: fever with tachycardia, elevated lactate indicating tissue hypoperfusion, significant leukocytosis, a tunneled HD catheter as a portal of entry, and a prior history of MRSA bacteremia. Altered mental status in the context of these vitals strongly raises concern for bacteremia. Thrombocytopenia may reflect early sepsis-induced consumptive process. The overall clinical picture is highly consistent with BSI.",
  "prediction": 1,
  "confidence": 0.92
}
<<<END>>>"""


def build_prediction_prompt(patient_text: str) -> str:
    """
    Chain-of-thought BSI prediction prompt.

    Pattern: role specification → structured patient data → explicit reasoning
    steps → constrained JSON output.  The model enumerates evidence for/against
    before committing to a prediction, which reduces hallucination and improves
    calibration on clinical tasks.
    """
    return f"""You are an expert infectious disease clinician evaluating a patient for blood stream infection (BSI/bacteremia).

Below is a structured patient summary followed by clinical notes. Your task is to predict whether this patient has a positive blood stream infection.

{_FEW_SHOT_EXAMPLE}

--------------------------------------------------

NOW EVALUATE THIS PATIENT:

PATIENT SUMMARY:
{patient_text}

--------------------------------------------------

INSTRUCTIONS — follow these steps in order:
Step 1: List up to 5 key clinical findings that INCREASE BSI probability (fever, tachycardia, indwelling devices, immunocompromise, lab abnormalities, etc.).
Step 2: List up to 3 factors that DECREASE BSI probability or suggest an alternative diagnosis.
Step 3: Write 2–4 sentences of clinical reasoning integrating the above evidence.
Step 4: State your final binary prediction (1 = positive BSI, 0 = negative BSI) and a confidence score (0.0–1.0).

OUTPUT FORMAT — return ONLY valid JSON followed immediately by {STOP_TOKEN}. No text before or after.

{{
  "bsi_indicators": ["finding1", "finding2"],
  "against_bsi": ["factor1"],
  "reasoning": "Your integrated clinical reasoning here.",
  "prediction": 1,
  "confidence": 0.85
}}
{STOP_TOKEN}

OUTPUT:
"""


# ── Generation config ─────────────────────────────────────────────────────────

@dataclass
class PredictionGenerationConfig:
    """Generation config tuned for structured JSON prediction with CoT."""
    max_new_tokens: int = 600
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
        "PATIENT SUMMARY:",
    ])


# ── JSON parsing ──────────────────────────────────────────────────────────────

def parse_prediction_json(raw: str) -> Dict:
    """
    Parse LLM output into a validated prediction dict.

    Expected keys: bsi_indicators (list), against_bsi (list),
                   reasoning (str), prediction (0/1), confidence (float).

    Falls back to partial extraction if full JSON parse fails.
    """
    cleaned = raw.strip().replace(STOP_TOKEN, "").strip()

    for text in [cleaned, _fix_common_json_issues(cleaned)]:
        # Direct parse
        try:
            obj = json.loads(text)
            return _validate_prediction(obj)
        except (json.JSONDecodeError, ValueError):
            pass

        # Extract first JSON object
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                obj = json.loads(match.group())
                return _validate_prediction(obj)
            except (json.JSONDecodeError, ValueError):
                pass

    # Partial recovery: extract prediction and confidence from any text
    partial = _recover_prediction_fields(cleaned)
    if partial is not None:
        return partial

    raise ValueError(f"Could not parse prediction JSON: {raw[:300]}")


def _fix_common_json_issues(text: str) -> str:
    fixed = re.sub(r",\s*}", "}", text)
    fixed = re.sub(r",\s*]", "]", fixed)
    return fixed


def _validate_prediction(obj: dict) -> Dict:
    """Validate and normalise the parsed prediction dict."""
    if not isinstance(obj, dict):
        raise ValueError("Not a dict")

    prediction = obj.get("prediction")
    confidence = obj.get("confidence")

    if prediction is None:
        raise ValueError("Missing 'prediction' key")

    try:
        prediction = int(float(prediction))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid prediction value: {prediction}")

    if prediction not in (0, 1):
        raise ValueError(f"prediction must be 0 or 1, got {prediction}")

    try:
        confidence = float(confidence) if confidence is not None else None
        if confidence is not None:
            confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = None

    return {
        "bsi_indicators": obj.get("bsi_indicators", []),
        "against_bsi":    obj.get("against_bsi", []),
        "reasoning":      str(obj.get("reasoning", "")),
        "prediction":     prediction,
        "confidence":     confidence,
    }


def _recover_prediction_fields(text: str) -> Optional[Dict]:
    """Last-resort partial extraction: find prediction and confidence values."""
    pred_match = re.search(r'"prediction"\s*:\s*(0|1)', text)
    conf_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text)
    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)

    if pred_match is None:
        return None

    return {
        "bsi_indicators": [],
        "against_bsi":    [],
        "reasoning":      reasoning_match.group(1) if reasoning_match else "",
        "prediction":     int(pred_match.group(1)),
        "confidence":     float(conf_match.group(1)) if conf_match else None,
    }


# ── Text generation ───────────────────────────────────────────────────────────

def generate_prediction(
    model,
    tokenizer,
    patient_text: str,
    gen_config: PredictionGenerationConfig,
    model_type: str,
    context_length: int,
) -> str:
    """Generate chain-of-thought BSI prediction for a single patient."""
    from transformers import TextIteratorStreamer

    # Notes are the last section of patient_text; truncate to fit context.
    # The prompt overhead (instructions + example) is ~900 tokens.
    truncated_text = truncate_notes_to_fit(
        patient_text, tokenizer, context_length,
        prompt_overhead_tokens=900,
        max_new_tokens=gen_config.max_new_tokens,
    )

    raw_prompt = build_prediction_prompt(truncated_text)

    if model_type == "gemma":
        messages = [{"role": "user", "content": raw_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    elif model_type == "qwen":
        messages = [{"role": "user", "content": raw_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    else:
        prompt = raw_prompt

    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        **inputs,
        "max_new_tokens": gen_config.max_new_tokens,
        "min_new_tokens": gen_config.min_new_tokens,
        "do_sample":      gen_config.do_sample,
        "num_beams":      gen_config.num_beams,
        "repetition_penalty": gen_config.repetition_penalty,
        "pad_token_id":   tokenizer.eos_token_id or 0,
    }
    if gen_config.do_sample:
        gen_kwargs["temperature"] = gen_config.temperature
        gen_kwargs["top_p"]       = gen_config.top_p

    if gen_config.stream:
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        gen_kwargs["streamer"] = streamer

        thread_exc: list = []

        def _generate_wrapper(**kwargs):
            try:
                model.generate(**kwargs)
            except Exception as e:
                thread_exc.append(e)
                streamer.end()

        thread = Thread(target=_generate_wrapper, kwargs=gen_kwargs)
        thread.start()

        generated = ""
        for chunk in streamer:
            generated += chunk
            if gen_config.verbose:
                print(chunk, end="", flush=True)
            if STOP_TOKEN in generated:
                break
        thread.join()
        if thread_exc:
            raise thread_exc[0]
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
    "primarymrn", "EncounterKey", "Positive",
    "prediction", "confidence", "reasoning",
    "bsi_indicators", "against_bsi",
    "parse_success", "notes_length", "model",
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
    gen_config: PredictionGenerationConfig,
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
    pbar = tqdm(remaining.iterrows(), total=len(remaining), desc="Predicting BSI")

    for _, row in pbar:
        encounter_key = str(row["EncounterKey"])
        mrn           = str(row.get("primarymrn", ""))
        label         = row.get("Positive", None)
        notes         = str(row["CombinedNotes"]) if "CombinedNotes" in row.index and pd.notna(row["CombinedNotes"]) else ""
        notes_length  = len(notes)

        pbar.set_postfix(ek=encounter_key, notes_len=notes_length, fails=parse_failures)

        patient_text = build_patient_text(row)

        if dry_run:
            buffer.append({
                "primarymrn":    mrn,
                "EncounterKey":  encounter_key,
                "Positive":      label,
                "prediction":    None,
                "confidence":    None,
                "reasoning":     "[DRY RUN]",
                "bsi_indicators": "[]",
                "against_bsi":   "[]",
                "parse_success": True,
                "notes_length":  notes_length,
                "model":         model_name,
            })
            done_encounters.add(encounter_key)
            if len(buffer) >= save_every:
                append_results(output_file, buffer)
                buffer = []
            continue

        try:
            raw_output = generate_prediction(
                model, tokenizer, patient_text, gen_config, model_type, context_length,
            )

            try:
                result = parse_prediction_json(raw_output)
                parse_success = True
            except ValueError as e:
                logger.warning(f"Parse failure for {encounter_key}: {e}")
                result = {
                    "bsi_indicators": [],
                    "against_bsi":    [],
                    "reasoning":      raw_output.replace(STOP_TOKEN, "").strip(),
                    "prediction":     None,
                    "confidence":     None,
                }
                parse_success = False
                parse_failures += 1

            buffer.append({
                "primarymrn":    mrn,
                "EncounterKey":  encounter_key,
                "Positive":      label,
                "prediction":    result["prediction"],
                "confidence":    result["confidence"],
                "reasoning":     result["reasoning"],
                "bsi_indicators": json.dumps(result["bsi_indicators"]),
                "against_bsi":   json.dumps(result["against_bsi"]),
                "parse_success": parse_success,
                "notes_length":  notes_length,
                "model":         model_name,
            })

        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM for encounter {encounter_key} (notes_len={notes_length}). Skipping.")
            torch.cuda.empty_cache()
            buffer.append({
                "primarymrn":    mrn,
                "EncounterKey":  encounter_key,
                "Positive":      label,
                "prediction":    None,
                "confidence":    None,
                "reasoning":     "ERROR: OOM",
                "bsi_indicators": "[]",
                "against_bsi":   "[]",
                "parse_success": False,
                "notes_length":  notes_length,
                "model":         model_name,
            })
        except Exception as e:
            logger.error(f"Error for encounter {encounter_key}: {e}")
            buffer.append({
                "primarymrn":    mrn,
                "EncounterKey":  encounter_key,
                "Positive":      label,
                "prediction":    None,
                "confidence":    None,
                "reasoning":     f"ERROR: {e}",
                "bsi_indicators": "[]",
                "against_bsi":   "[]",
                "parse_success": False,
                "notes_length":  notes_length,
                "model":         model_name,
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
        description="LLM-based BSI prediction from tabular + notes data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()), required=True,
        help="Model to use for prediction",
    )
    parser.add_argument(
        "--data", type=str, default=str(DEFAULT_DATA),
        help="Path to combined_bsi_dataset.csv (output of prepare_dataset.py)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUT),
        help="Directory for prediction output CSV",
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
        help="Limit to first N patients (for testing)",
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
        help="Process encounters in descending order (run a second parallel job from the end)",
    )
    parser.add_argument(
        "--chunk", type=int, default=None,
        help="0-indexed chunk to process (use with --n_chunks). Maps to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--n_chunks", type=int, default=None,
        help="Total number of parallel chunks. Each chunk processes every n_chunks-th remaining encounter.",
    )

    gen = parser.add_argument_group("Generation parameters")
    gen.add_argument("--max_new_tokens", type=int, default=600)
    gen.add_argument("--min_new_tokens", type=int, default=10)
    gen.add_argument("--temperature", type=float, default=0.3)
    gen.add_argument("--top_p", type=float, default=0.9)
    gen.add_argument("--no_stream", action="store_true", default=False)
    gen.add_argument("--verbose", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset: {args.data}")
    df = pd.read_csv(args.data, low_memory=False)
    df["EncounterKey"] = df["EncounterKey"].astype(str)
    logger.info(f"  {len(df):,} encounters loaded")

    if args.n_patients is not None:
        df = df.head(args.n_patients)
        logger.info(f"  Limited to {len(df)} patients (--n_patients)")

    # Output file
    output_file = os.path.join(
        args.output_dir, f"bsi_llm_predictions_{args.model}.csv"
    )

    # Retry-failure mode: drop previously failed rows so they get re-run
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

    gen_config = PredictionGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=not args.no_stream,
        verbose=args.verbose,
        do_sample=True,
    )

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
        model_type    = model_cfg["model_type"]
        context_length = model_cfg["context_length"]

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
