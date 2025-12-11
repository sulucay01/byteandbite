#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from pathlib import Path
import requests
import pandas as pd


SYSTEM_PROMPT_DEFAULT = (
    "You are a helpful assistant for food & restaurants.\n"
    "Answer in English in 1–3 concise sentences unless asked otherwise.\n"
    "Start answer with Answer:.\n"
    "If you are unsure, say you don’t know.\n"
    "You need to be absolutely sure while suggesting a restaurant. "
    "If you are not sure, say you don’t know.\n"
    "Keep answers general; do NOT invent venues, addresses, menus, phone numbers, or URLs.\n"
)

# (model_tag_on_ollama, short_name)
DEFAULT_MODELS = [
    ("llama3.1:8b-instruct-q4_K_M", "llama31_8b"),
    ("mistral:7b-instruct-q4_K_M", "mistral7b"),
]


def chat_ollama(model: str, system: str, user: str, *, timeout: int = 600) -> str:
    """Single turn chat against local Ollama /api/chat with basic retries."""
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "num_ctx": 2048,
            "temperature": 0.3,
            "repeat_penalty": 1.1,
        },
        "stream": False,
    }

    last_exc = None
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"]
        except Exception as e:
            last_exc = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Ollama chat failed after retries: {last_exc}")


def load_questions(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [q.strip() for q in lines if q.strip()]


def run_once(model_tag: str, shortname: str, questions: list[str], system_prompt: str, outdir: Path) -> Path:
    rows: list[dict] = []
    for q in questions:
        t0 = time.time()
        ans = chat_ollama(model_tag, system_prompt, q)
        dt = time.time() - t0
        rows.append(
            {
                "model": shortname,
                "question": q,
                "answer": ans,
                "latency_sec": round(dt, 3),
                "answer_words": len(ans.split()),
            }
        )
        print(f"[{shortname}] {q[:60]}... -> {round(dt, 2)}s")

    df = pd.DataFrame(rows)
    out_csv = outdir / f"{shortname}_llm_only_run.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return out_csv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Baseline LLM run against Ollama")
    ap.add_argument("--questions", default="questions.txt", help="Path to questions file")
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Override models as pairs: tag1 name1 tag2 name2 ... (even count). "
             "Example: --models llama3.1:8b-instruct-q4_K_M llama31_8b",
    )
    ap.add_argument("--system_prompt_file", default=None, help="Optional custom system prompt file")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    models = DEFAULT_MODELS
    if args.models:
        if len(args.models) % 2 != 0:
            raise SystemExit("ERROR: --models must be pairs: <tag short> ...")
        it = iter(args.models)
        models = [(tag, short) for tag, short in zip(it, it)]

    system_prompt = (
        Path(args.system_prompt_file).read_text(encoding="utf-8")
        if args.system_prompt_file
        else SYSTEM_PROMPT_DEFAULT
    )

    questions = load_questions(args.questions)
    if not questions:
        raise SystemExit(f"No questions found in {args.questions}")

    for model_tag, short in models:
        run_once(model_tag, short, questions, system_prompt, outdir)


if __name__ == "__main__":
    main()
