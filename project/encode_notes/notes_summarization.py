#!/usr/bin/env python3
"""
Summarize patient clinical notes for predicting positive Blood Stream Infection (BSI).

Adapted from the xHAIM pipeline (Step 1: task-relevant summarization).
Supports MedGemma-27B and Qwen3-14B-AWQ models.

Usage:
    python summarize_notes.py --model medgemma-27b
    python summarize_notes.py --model qwen3-14b --batch_index 0
    python summarize_notes.py --model qwen3-14b --batch_start 0 --batch_end 5
"""

import argparse
import glob
import gc
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Generation configuration (defaults match the provided GenerationConfig)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 500
    min_new_tokens: int = 40
    temperature: float = 0.7
    top_p: float = 0.8
    stream: bool = True
    verbose: bool = False
    batch_size: int = 1
    do_sample: bool = True
    num_beams: int = 1
    no_repeat_ngram_size: int = 0
    repetition_penalty: float = 1.0
    early_stopping: bool = False

    stop_sequences: List[str] = field(default_factory=lambda: [
        "Let me know",
        "Please let",
        "Would you like",
        "END OF SUMMARY",
        "END OF RESPONSE",
        "```\n\n",
        "python",
        "```assistant",
        "stop here",
        "STOP.",
        ".assistant",
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "medgemma-27b": {
        "model_id": "google/medgemma-27b-text-it",
        "model_type": "gemma",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "context_length": 32768,
        "load_kwargs": {
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        },
    },
    "qwen3-14b": {
        "model_id": "Qwen/Qwen3-14B",
        "model_type": "qwen",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "context_length": 32768,
        "load_kwargs": {
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        },
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# BSI-focused summarization prompt
# ──────────────────────────────────────────────────────────────────────────────

BSI_PROMPT_TEMPLATE = """You are a medical summarization assistant. Your task is to provide a single, focused summary of the following patient clinical notes with special attention to blood stream infection (BSI).

Instructions:
1. Provide ONLY ONE concise summary.
2. Summarize the patient's overall clinical status, relevant medical history, comorbidities, and current conditions, with specific attention to BSI risk factors and indicators.
3. Prioritize information related to BSI, but include all clinically relevant information.
4. Pay special attention to:
   - Blood culture results (positive or negative)
   - Bacteremia, sepsis, or systemic inflammatory response
   - Fever, elevated white blood cell count, or inflammatory markers (CRP, procalcitonin)
   - Presence of central venous catheters, PICC lines, or other indwelling devices
   - Immunosuppression or neutropenia
   - Recent surgeries, procedures, or wounds
   - Antibiotic use and changes in antibiotic regimen
   - Signs of end-organ dysfunction possibly related to infection
5. Include information about BSI ONLY if explicitly mentioned (either present, suspected, or ruled out).
6. Do NOT state "there is no mention of BSI" if it is simply not referenced.
7. Base the summary *only* on the provided notes.
8. Do not make up information or add your own medical opinions.
9. Do not offer a diagnosis or predict outcomes. Focus on objective factual reporting.
10. Stop completely after providing the summary.

If there is no information in the patient notes (e.g., they are empty), say so and stop.

START OF PATIENT NOTES:

{notes}

END OF PATIENT NOTES.

Based on the above notes, provide a medical summary with emphasis on blood stream infection (BSI) risk factors and indicators when present. Focus on factual reporting:
The patient"""


def format_bsi_prompt(notes: str) -> str:
    return BSI_PROMPT_TEMPLATE.format(notes=notes)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def login_huggingface(token: str):
    """Log in to HuggingFace Hub."""
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        logger.info("HuggingFace login successful")
    except Exception as e:
        logger.warning(f"HuggingFace login failed: {e}. Proceeding anyway.")


def load_model(model_name: str, hf_token: Optional[str] = None, quantize_4bit: bool = False):
    """Load model and tokenizer from the registry."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_REGISTRY[model_name]
    model_id = cfg["model_id"]

    logger.info(f"Loading model: {model_id}")

    token_kwargs = {"token": hf_token} if hf_token else {}

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=cfg["trust_remote_code"],
        **token_kwargs,
    )

    load_kwargs = {**cfg["load_kwargs"]}

    try:
        import flash_attn  # noqa: F401
        load_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("flash_attn found — using FlashAttention-2")
    except ImportError:
        load_kwargs.setdefault("attn_implementation", "eager")
        logger.info("flash_attn not found — falling back to eager attention")

    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=cfg["torch_dtype"],
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Using 4-bit quantization (bitsandbytes NF4)")
    else:
        load_kwargs["torch_dtype"] = cfg["torch_dtype"]

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=cfg["trust_remote_code"],
        **load_kwargs,
        **token_kwargs,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info(f"Model loaded successfully")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Text generation
# ──────────────────────────────────────────────────────────────────────────────

def clean_response(text: str, stop_sequences: List[str]) -> str:
    """Truncate response at stop sequences."""
    cleaned = text
    cleaned_lower = cleaned.lower()
    for seq in stop_sequences:
        seq_lower = seq.lower()
        if seq_lower in cleaned_lower:
            pos = cleaned_lower.index(seq_lower)
            cleaned = cleaned[:pos].strip()
            cleaned_lower = cleaned.lower()
    return cleaned.strip()


def truncate_notes_to_fit(
    notes: str,
    tokenizer,
    context_length: int,
    prompt_overhead_tokens: int = 600,
    max_new_tokens: int = 500,
) -> str:
    """Truncate notes so the full prompt fits within the model's context window."""
    available_tokens = context_length - prompt_overhead_tokens - max_new_tokens
    if available_tokens <= 0:
        available_tokens = 1024

    note_tokens = tokenizer.encode(notes, add_special_tokens=False)
    if len(note_tokens) <= available_tokens:
        return notes

    truncated_tokens = note_tokens[:available_tokens]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    logger.debug(
        f"Truncated notes from {len(note_tokens)} to {available_tokens} tokens"
    )
    return truncated_text + "\n[... notes truncated due to length ...]"


def generate_summary(
    model,
    tokenizer,
    notes: str,
    gen_config: GenerationConfig,
    model_type: str,
    context_length: int,
) -> str:
    """Generate a BSI-focused summary for a single patient's notes."""
    from transformers import TextIteratorStreamer

    truncated_notes = truncate_notes_to_fit(
        notes, tokenizer, context_length,
        max_new_tokens=gen_config.max_new_tokens,
    )
    raw_prompt = format_bsi_prompt(truncated_notes)

    if model_type == "qwen":
        messages = [{"role": "user", "content": raw_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    elif model_type == "gemma":
        messages = [{"role": "user", "content": raw_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt = raw_prompt

    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        **inputs,
        "max_new_tokens": gen_config.max_new_tokens,
        "min_new_tokens": gen_config.min_new_tokens,
        "do_sample": gen_config.do_sample,
        "num_beams": gen_config.num_beams,
        "no_repeat_ngram_size": gen_config.no_repeat_ngram_size,
        "repetition_penalty": gen_config.repetition_penalty,
        "early_stopping": gen_config.early_stopping,
        "pad_token_id": tokenizer.eos_token_id or 0,
    }
    if gen_config.do_sample:
        gen_kwargs["temperature"] = gen_config.temperature
        gen_kwargs["top_p"] = gen_config.top_p

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

    return clean_response(generated, gen_config.stop_sequences)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def get_batch_files(data_dir: str) -> List[str]:
    """Return sorted list of batch CSV files."""
    pattern = os.path.join(data_dir, "base_*_raw_notes_*.csv")
    files = sorted(glob.glob(pattern), key=_batch_sort_key)
    return files


def _batch_sort_key(path: str) -> int:
    """Extract start index from filename for sorting."""
    name = Path(path).stem
    try:
        parts = name.replace("base_HH_raw_notes_", "").split("_")
        return int(parts[0])
    except (ValueError, IndexError):
        return 0


def load_batch(filepath: str) -> pd.DataFrame:
    """Load a single batch CSV, reading only the required columns."""
    logger.info(f"Loading batch: {filepath}")
    df = pd.read_csv(filepath, usecols=["primarymrn", "EncounterKey", "CombinedNotes"])
    logger.info(f"  Loaded {len(df)} rows ({df['EncounterKey'].nunique()} unique encounters)")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Progress tracking & resume
# ──────────────────────────────────────────────────────────────────────────────

def load_existing_results(output_file: str) -> pd.DataFrame:
    """Load existing results for resume capability."""
    if os.path.exists(output_file):
        return pd.read_csv(output_file)
    return pd.DataFrame(columns=["primarymrn", "EncounterKey", "summary", "notes_length", "model"])


def save_results(output_file: str, results_df: pd.DataFrame):
    """Save results to CSV."""
    results_df.to_csv(output_file, index=False)


def append_results(output_file: str, new_rows: List[dict]):
    """Append new results to the output file."""
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(output_file, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_batch(
    df: pd.DataFrame,
    model,
    tokenizer,
    gen_config: GenerationConfig,
    model_name: str,
    model_type: str,
    context_length: int,
    output_file: str,
    save_every: int = 10,
):
    """Process a single batch of encounters."""
    existing = load_existing_results(output_file)
    done_encounters = set(existing["EncounterKey"].astype(str).values)
    logger.info(f"  {len(done_encounters)} encounters already processed, skipping those")

    remaining_df = df[~df["EncounterKey"].astype(str).isin(done_encounters)]
    buffer = []
    pbar = tqdm(remaining_df.iterrows(), total=len(remaining_df), desc="Summarizing")

    for idx, row in pbar:
        encounter_key = str(row["EncounterKey"])

        mrn = str(row["primarymrn"])
        notes = str(row["CombinedNotes"]) if pd.notna(row["CombinedNotes"]) else ""
        pbar.set_postfix(ek=encounter_key, notes_len=len(notes))

        if not notes or notes == "nan":
            buffer.append({
                "primarymrn": mrn,
                "EncounterKey": encounter_key,
                "summary": "",
                "notes_length": 0,
                "model": model_name,
            })
            done_encounters.add(encounter_key)
            continue

        try:
            summary = generate_summary(
                model, tokenizer, notes, gen_config, model_type, context_length,
            )
            buffer.append({
                "primarymrn": mrn,
                "EncounterKey": encounter_key,
                "summary": summary,
                "notes_length": len(notes),
                "model": model_name,
            })
        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM for encounter {encounter_key} (notes_len={len(notes)}). Skipping.")
            torch.cuda.empty_cache()
            buffer.append({
                "primarymrn": mrn,
                "EncounterKey": encounter_key,
                "summary": "ERROR: OOM",
                "notes_length": len(notes),
                "model": model_name,
            })
        except Exception as e:
            logger.error(f"Error for encounter {encounter_key}: {e}")
            buffer.append({
                "primarymrn": mrn,
                "EncounterKey": encounter_key,
                "summary": f"ERROR: {e}",
                "notes_length": len(notes),
                "model": model_name,
            })

        done_encounters.add(encounter_key)

        if len(buffer) >= save_every:
            append_results(output_file, buffer)
            buffer = []

    if buffer:
        append_results(output_file, buffer)

    logger.info(f"  Batch complete. Results saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize clinical notes for BSI prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()), required=True,
        help="Model to use for summarization",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory containing batch CSV files",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "output"),
        help="Directory for output summaries",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token for gated models",
    )
    parser.add_argument(
        "--batch_index", type=int, default=None,
        help="Process only this batch index (0-indexed)",
    )
    parser.add_argument(
        "--batch_start", type=int, default=None,
        help="Start batch index (inclusive)",
    )
    parser.add_argument(
        "--batch_end", type=int, default=None,
        help="End batch index (exclusive)",
    )
    parser.add_argument(
        "--save_every", type=int, default=10,
        help="Save intermediate results every N patients",
    )

    parser.add_argument(
        "--quantize_4bit", action="store_true", default=False,
        help="Load model in 4-bit quantization (NF4) to reduce VRAM usage",
    )

    gen_group = parser.add_argument_group("Generation parameters")
    gen_group.add_argument("--max_new_tokens", type=int, default=500)
    gen_group.add_argument("--min_new_tokens", type=int, default=40)
    gen_group.add_argument("--temperature", type=float, default=0.7)
    gen_group.add_argument("--top_p", type=float, default=0.8)
    gen_group.add_argument("--do_sample", action="store_true", default=True)
    gen_group.add_argument("--no_stream", action="store_true", default=False)
    gen_group.add_argument("--verbose", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.hf_token:
        login_huggingface(args.hf_token)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=not args.no_stream,
        verbose=args.verbose,
        do_sample=args.do_sample,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    model_cfg = MODEL_REGISTRY[args.model]
    model, tokenizer = load_model(args.model, args.hf_token, quantize_4bit=args.quantize_4bit)

    batch_files = get_batch_files(args.data_dir)
    if not batch_files:
        logger.error(f"No batch files found in {args.data_dir}")
        return

    logger.info(f"Found {len(batch_files)} batch files")

    if args.batch_index is not None:
        batch_files = [batch_files[args.batch_index]]
    elif args.batch_start is not None:
        end = args.batch_end if args.batch_end is not None else len(batch_files)
        batch_files = batch_files[args.batch_start:end]

    logger.info(f"Processing {len(batch_files)} batch(es)")

    for fpath in batch_files:
        batch_name = Path(fpath).stem
        output_file = os.path.join(
            args.output_dir, f"bsi_summaries_{args.model}_{batch_name}.csv"
        )

        df = load_batch(fpath)

        process_batch(
            df=df,
            model=model,
            tokenizer=tokenizer,
            gen_config=gen_config,
            model_name=args.model,
            model_type=model_cfg["model_type"],
            context_length=model_cfg["context_length"],
            output_file=output_file,
            save_every=args.save_every,
        )

        del df
        gc.collect()

    logger.info("All batches complete.")


if __name__ == "__main__":
    main()
