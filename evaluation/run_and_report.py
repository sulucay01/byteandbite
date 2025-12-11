#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_and_report.py
- Runs two local LLMs via Ollama on a question set
- Injects an intent+guideline as a second system message (simple/static)
- Saves per-model CSVs
- Computes quick metrics (latency, length OK, I-don't-know rate, hallucination rate, TR-mention rate)
- Writes an Experiment Card (Markdown) with the results

Requirements:
  pip install pandas numpy requests
Assumes:
  - Ollama is running at 127.0.0.1:11434
  - Models are pulled: llama3.1:8b-instruct-q4_K_M, mistral:7b-instruct-q4_K_M
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import statistics
import time
from typing import List, Dict, Tuple

import pandas as pd
import requests


# ----------------------- CONFIG DEFAULTS -----------------------

DEFAULT_MODELS = [
    ("llama3.1:8b-instruct-q4_K_M", "llama31_8b"),
    ("mistral:7b-instruct-q4_K_M", "mistral7b"),
]

BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant for food & restaurants.\n"
    "Answer in English in 1–3 concise sentences unless asked otherwise.\n"
    "If you are unsure, say you don’t know.\n"
    "Keep answers general; do not invent specific venues, addresses, menus, phone numbers, or URLs.\n"
    "Avoid mentioning any specific country unless explicitly asked.\n"
)

# Simple, static intent block for this baseline (can be replaced by classifier later)
DEFAULT_INTENT_LABEL = "menu_inquiry"
DEFAULT_INTENT_GUIDELINE = (
    "The user's intent is 'menu_inquiry'. Summarize menu highlights in neutral, country-agnostic terms. "
    "Avoid specific venue names, addresses, and URLs."
)

# Quick-check regex patterns
IDK_PAT = re.compile(r"\b(i (do not|don't) know|not sure|unsure)\b", re.I)
HALLU_PAT = re.compile(r"(street|avenue|road|no\.\s?\d+|\b[A-Z][a-z]+ Restaurant\b|\+\d{6,}|http[s]?://)", re.I)
TR_PAT = re.compile(r"\b(Turkey|Türkiye|Istanbul|Ankara|Izmir)\b", re.I)

# Word-length target range
TARGET_MIN_WORDS = 15
TARGET_MAX_WORDS = 60

# Ollama endpoint defaults
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434


# ----------------------- CORE FUNCTIONS -----------------------

def build_messages(system_base: str, intent_block: str, user_text: str) -> List[Dict[str, str]]:
    """Constructs message list in priority order: system(base) -> system(intent) -> user"""
    return [
        {"role": "system", "content": system_base},
        {"role": "system", "content": intent_block},
        {"role": "user", "content": user_text},
    ]


def chat_ollama(model: str,
                messages: List[Dict[str, str]],
                num_ctx: int = 2048,
                temperature: float = 0.3,
                repeat_penalty: float = 1.1,
                host: str = OLLAMA_HOST,
                port: int = OLLAMA_PORT,
                timeout_s: int = 600) -> str:
    """Calls Ollama /api/chat and returns assistant content."""
    url = f"http://{host}:{port}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "repeat_penalty": repeat_penalty
        },
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f.read().splitlines() if x.strip()]


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def run_model_on_questions(model_tag: str,
                           shortname: str,
                           questions: List[str],
                           system_prompt: str,
                           intent_block: str,
                           outdir: str,
                           num_ctx: int,
                           temperature: float,
                           host: str,
                           port: int) -> pathlib.Path:
    """Runs a single model over all questions and saves CSV. Returns path to CSV."""
    rows = []
    for q in questions:
        messages = build_messages(system_prompt, intent_block, q)
        t0 = time.time()
        try:
            ans = chat_ollama(model_tag, messages, num_ctx=num_ctx, temperature=temperature, host=host, port=port)
        except Exception as e:
            ans = f"__ERROR__: {type(e).__name__}: {e}"
        dt = time.time() - t0
        rows.append({
            "model": shortname,
            "question": q,
            "answer": ans,
            "latency_sec": round(dt, 3),
            "answer_words": len(ans.split())
        })
        print(f"[{shortname}] {q[:60]} -> {round(dt,2)}s")

    df = pd.DataFrame(rows)
    out_csv = pathlib.Path(outdir) / f"{shortname}_llm_only_run.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return out_csv


def quick_flags_df(df: pd.DataFrame) -> pd.DataFrame:
    """Adds quick-check flags to a per-model results DF."""
    def _flags(row):
        text = str(row["answer"])
        return pd.Series({
            "len_ok": (TARGET_MIN_WORDS <= int(row["answer_words"]) <= TARGET_MAX_WORDS),
            "idk_flag": bool(IDK_PAT.search(text)),
            "halluc_flag": bool(HALLU_PAT.search(text)),
            "turkey_mentioned": bool(TR_PAT.search(text)),
        })
    return pd.concat([df, df.apply(_flags, axis=1)], axis=1)


def summarize_flags(df: pd.DataFrame, model_name: str) -> Dict[str, float]:
    """Produces aggregate metrics for a model DF with flags."""
    out = {
        "model": model_name,
        "n": int(len(df)),
        "latency_avg_s": round(float(df["latency_sec"].mean()), 3) if len(df) else 0.0,
        "len_ok_rate": round(float(df["len_ok"].mean()), 3) if len(df) else 0.0,
        "idk_rate": round(float(df["idk_flag"].mean()), 3) if len(df) else 0.0,
        "halluc_rate": round(float(df["halluc_flag"].mean()), 3) if len(df) else 0.0,
        "turkey_mention_rate": round(float(df["turkey_mentioned"].mean()), 3) if len(df) else 0.0,
    }
    return out


def write_experiment_card(card_path: pathlib.Path,
                          date_str: str,
                          hardware_desc: str,
                          models_info: List[Tuple[str, str]],
                          prompt_desc: str,
                          questions_path: str,
                          summary_rows: List[Dict[str, float]]) -> None:
    """Writes a Markdown experiment card summarizing results."""
    def _fmt_row(r: Dict[str, float]) -> str:
        return f"| {r['model']} | {r['latency_avg_s']:.3f} | {r['len_ok_rate']:.3f} | {r['idk_rate']:.3f} | {r['halluc_rate']:.3f} | {r['turkey_mention_rate']:.3f} |"

    models_list_str = ", ".join([f"{m} ({s})" for m, s in models_info])
    md = f"""# Experiment Card — LLM-only Baseline

**Date:** {date_str}  
**Hardware:** {hardware_desc}  
**Models:** {models_list_str}  
**Prompt (system):** {prompt_desc}  
**Data:** Questions file: `{questions_path}`

## Results (Automatic Quick Checks)
| Model | Latency(s) | Len OK | I-don’t-know | Halluc% | TR-mention% |
|------|------------:|-------:|-------------:|--------:|------------:|
"""
    for r in summary_rows:
        md += _fmt_row(r) + "\n"

    md += """

## Notes
- No RAG data used in this baseline.
- Answers are constrained to be generic and short; country/venue specifics are avoided by prompt design.
- Intent is injected as a secondary system message (static in this baseline).

## Next Steps
- Integrate a lightweight BERT intent classifier to select the intent dynamically.
- Add RAG layer with Yelp corpus and a vector DB; inject retrieved context as an additional system message.
- Expand evaluation with human rubrics (Accuracy/Coverage/Clarity/Safety).
"""
    card_path.write_text(md, encoding="utf-8")
    print(f"Experiment Card written to: {card_path}")


# ----------------------- CLI -----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Run two local LLMs, compute quick metrics, and write an Experiment Card.")
    ap.add_argument("--questions", default="questions.txt", help="Path to questions file (one question per line).")
    ap.add_argument("--outdir", default="outputs", help="Output directory for CSVs and summaries.")
    ap.add_argument("--cards_dir", default="experiment_cards", help="Directory to write the experiment card.")
    ap.add_argument("--num_ctx", type=int, default=2048, help="Ollama context window.")
    ap.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature.")
    ap.add_argument("--host", default=OLLAMA_HOST, help="Ollama host.")
    ap.add_argument("--port", type=int, default=OLLAMA_PORT, help="Ollama port.")
    # Optional customizations
    ap.add_argument("--system_prompt", default=BASE_SYSTEM_PROMPT, help="Base system prompt text.")
    ap.add_argument("--intent_block", default=DEFAULT_INTENT_GUIDELINE, help="Intent+guideline block (second system message).")
    return ap.parse_args()


def main():
    args = parse_args()

    # Ensure dirs
    ensure_dirs(args.outdir, args.cards_dir)

    # Load questions
    questions = load_lines(args.questions)
    if not questions:
        raise SystemExit(f"No questions found in: {args.questions}")

    # Run each model and save CSV
    csv_paths = []
    for model_tag, short in DEFAULT_MODELS:
        csv_path = run_model_on_questions(
            model_tag=model_tag,
            shortname=short,
            questions=questions,
            system_prompt=args.system_prompt,
            intent_block=args.intent_block,
            outdir=args.outdir,
            num_ctx=args.num_ctx,
            temperature=args.temperature,
            host=args.host,
            port=args.port
        )
        csv_paths.append((short, csv_path))

    # Quick checks and summary
    summary_rows = []
    for short, csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df_flags = quick_flags_df(df)
        flagged_path = pathlib.Path(str(csv_path).replace(".csv", "_with_flags.csv"))
        df_flags.to_csv(flagged_path, index=False)
        summary = summarize_flags(df_flags, model_name=short)
        summary_rows.append(summary)

    # Write combined summary file
    summary_df = pd.DataFrame(summary_rows)
    summary_out = pathlib.Path(args.outdir) / "summary_quick_checks.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"Saved: {summary_out}")

    # Experiment card
    today = datetime.date.today().isoformat()
    hardware_desc = "Local GPU (RTX 2070 Max-Q, 8 GB), Ollama ≥ 0.12.9"
    prompt_desc = "Short, generic, safety-constrained; avoid specific venues/countries. Secondary system message carries the intent guideline."
    card_path = pathlib.Path(args.cards_dir) / f"{today}-baseline-llm-only.md"
    write_experiment_card(
        card_path=card_path,
        date_str=today,
        hardware_desc=hardware_desc,
        models_info=DEFAULT_MODELS,
        prompt_desc=prompt_desc,
        questions_path=args.questions,
        summary_rows=summary_rows
    )


if __name__ == "__main__":
    main()
