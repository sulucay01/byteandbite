# app.py — LLM-ONLY VERSION (Crash-proof JSON + robust streaming)
# Byte & Bite RAG backend (FastAPI)
# - Intent: LLM-only (no BERT)
# - Streaming: Ollama /api/chat streaming -> plain text
# - Fixes:
#   1) No KeyError on 'response' (contextualizer)
#   2) Robust streaming parsing (message/content or response)
#   3) min_rating uses is not None
#   4) Better debug logs for Ollama errors

import os, time, traceback, re, json
from typing import List, Optional, Iterable, Tuple

import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue, Range
import requests
from sentence_transformers import SentenceTransformer

from constants import CUISINES_LIST, DIETARY_OPTIONS


# ---------------- Device / Model ----------------
def pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_model(name: str):
    dev = pick_device()
    print(f"[Embed] loading '{name}' on device: {dev}", flush=True)
    try:
        return SentenceTransformer(name, device=dev)
    except Exception as e:
        print(f"[Embed] failed on {dev}: {e} -> falling back to CPU", flush=True)
        return SentenceTransformer(name, device="cpu")


# ---------------- Config ----------------
COLLECTION  = os.getenv("Q_COLLECTION", "restaurants")
MODEL_NAME  = os.getenv("EMBED_MODEL", "intfloat/e5-small-v2")

# Important:
# - GEN_MODEL: which Ollama model to use
# - OLLAMA_URL: where ollama is
GEN_MODEL   = os.getenv("GEN_MODEL",  "llama3.1:8b-instruct-q4_K_M")
OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

E5_PREFIX   = os.getenv("E5_PREFIX", "query: ")

Q_HOST = os.getenv("Q_HOST", "127.0.0.1")
Q_PORT = int(os.getenv("Q_PORT", "6333"))

print("\n### SYSTEM START: LLM-ONLY INTENT VERSION (STABLE) ###\n", flush=True)

app    = FastAPI(title="Byte&Bite Search API")
_model = load_model(MODEL_NAME)
_cli   = QdrantClient(host=Q_HOST, port=Q_PORT)


# ---------------- Helpers ----------------
def build_filter(city, state, cuisines, diets, min_rating, max_price) -> Optional[Filter]:
    conds = []
    if city:
        conds.append(FieldCondition(key="city", match=MatchValue(value=city)))
    if state:
        conds.append(FieldCondition(key="state", match=MatchValue(value=state)))
    if cuisines:
        conds.append(FieldCondition(key="cuisines", match=MatchAny(any=cuisines)))
    if diets:
        conds.append(FieldCondition(key="diets", match=MatchAny(any=diets)))

    # FIX: must be is not None
    if min_rating is not None:
        conds.append(FieldCondition(key="rating", range=Range(gte=float(min_rating))))

    if max_price is not None:
        conds.append(FieldCondition(key="price", range=Range(lte=int(max_price))))

    return Filter(must=conds) if conds else None


def qdrant_query(qvec, limit, flt):
    # Qdrant client API can differ across versions; keep a fallback.
    try:
        resp = _cli.query_points(
            collection_name=COLLECTION,
            query=qvec,
            limit=limit,
            with_payload=True,
            filter=flt
        )
        return resp.points or []
    except Exception:
        resp = _cli.search(
            collection_name=COLLECTION,
            query_vector=qvec,
            limit=limit,
            with_payload=True,
            query_filter=flt
        )
        return resp or []


def apply_hard_filters(points, city=None, cuisines=None, diets=None, min_rating=None, max_price=None):
    out = []
    for p in points:
        pl = getattr(p, "payload", {}) or {}
        ok = True

        # City Check (fuzzy contains)
        if city:
            p_city = str(pl.get("city") or "").lower().strip()
            t_city = str(city).lower().strip()
            if t_city not in p_city:
                ok = False

        # Cuisine Check
        if cuisines:
            p_cuisines = [str(c).lower() for c in (pl.get("cuisines") or [])]
            t_cuisines = [str(c).lower() for c in cuisines]
            if not any(tc in pc for tc in t_cuisines for pc in p_cuisines):
                ok = False

        # Diet Check
        if diets:
            p_diets = [str(d).lower() for d in (pl.get("diets") or [])]
            for req in diets:
                req_lower = str(req).lower()
                keyword = (
                    "vegan" if "vegan" in req_lower else
                    "vegetarian" if "vegetarian" in req_lower else
                    "gluten" if "gluten" in req_lower else
                    req_lower
                )
                if not any(keyword in avail for avail in p_diets):
                    ok = False
                    break

        # Rating Check
        if min_rating is not None:
            if float(pl.get("rating") or 0.0) < float(min_rating):
                ok = False

        # Price Check
        if max_price is not None:
            price = pl.get("price")
            if price is None or int(price) > int(max_price):
                ok = False

        if ok:
            out.append(p)

    return out


# ---------------- Visual Card Logic ----------------
def build_context_block_visual(points, max_items=5):
    lines = []
    seen = set()
    price_map = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}

    for p in points:
        pl = getattr(p, "payload", {}) or {}
        name = pl.get("name")
        if not name or name in seen:
            continue
        seen.add(name)

        c_list = pl.get("cuisines", [])
        c_str = ", ".join(c_list) if isinstance(c_list, list) else str(c_list)

        d_list = pl.get("diets", [])
        d_str = ", ".join(d_list) if isinstance(d_list, list) else "None"

        raw_price = pl.get("price")
        price_symbol = price_map.get(int(raw_price), "?") if raw_price is not None else "?"

        lines.append(
            f"RESTAURANT: {name}\n"
            f" - City: {pl.get('city')}\n"
            f" - Cuisine: {c_str}\n"
            f" - Diets: {d_str}\n"
            f" - Price: {price_symbol}\n"
            f" - Rating: {pl.get('rating')}\n"
            f"----------------"
        )
        if len(lines) >= max_items:
            break

    return "\n".join(lines)


def _iter_ollama_stream_chat(payload: dict, timeout: int = 120) -> Iterable[str]:
    """
    Robust Ollama streamer:
    - supports both /api/chat style chunks: {"message":{"content":"..."}, "done":false}
    - and /api/generate style chunks: {"response":"..."}
    - if {"error": "..."} appears, yields a visible error line
    """
    try:
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    d = json.loads(line)

                    if d.get("error"):
                        yield f"\n[OLLAMA ERROR] {d['error']}\n"
                        continue

                    msg = (d.get("message") or {}).get("content")
                    if msg:
                        yield msg
                        continue

                    resp = d.get("response")
                    if resp:
                        yield resp
                        continue

                    # ignore done/status chunks silently
                except Exception:
                    # ignore malformed lines
                    continue
    except Exception as e:
        yield f"\n[STREAM ERROR] {e}\n"


def generate_visual_response_stream(model_tag: str, query: str, context: str) -> Iterable[str]:
    system_prompt = (
        "You are Byte & Bite, a helpful restaurant assistant. "
        "Your task is to format the provided restaurant data into clean, readable cards. "
        "You MUST ONLY use the data provided in [RAW DATA]. Do not invent or list restaurants not present in [RAW DATA]. "
        "If the [RAW DATA] says 'NO MATCHING RESTAURANTS FOUND', you must state that and not provide any recommendations."
    )

    user_prompt = f"""
[RAW DATA START]
{context}
[RAW DATA END]

USER QUERY: "{query}"

INSTRUCTIONS:
1. If [RAW DATA] contains restaurants, format them using the card format below.
2. If [RAW DATA] contains "NO MATCHING RESTAURANTS FOUND", reply with: "I couldn't find any restaurants matching your specific criteria."
3. DO NOT generate a list or use your own knowledge. STRICTLY use [RAW DATA].
4. Required Format per restaurant (repeat for each match in RAW DATA):

### 🍽️ **[Name]**
- 📍 **City:** [City]
- 🍲 **Cuisine:** [Cuisine]
- 🥗 **Diets:** [Diets]
- ⭐️ **Rating:** [Rating]
- 💵 **Price:** [Price]
*(Write a short, engaging 1-sentence comment based on the data)*
"""

    payload = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "options": {"temperature": 0.1}
    }

    yield from _iter_ollama_stream_chat(payload, timeout=120)


def generate_ollama_chat_stream(model_tag: str, user_query: str) -> Iterable[str]:
    payload = {
        "model": model_tag,
        "messages": [{"role": "user", "content": user_query}],
        "stream": True
    }
    yield from _iter_ollama_stream_chat(payload, timeout=120)


# ---------------- Filter Extraction (LLM Only) ----------------
def _safe_ollama_generate(prompt: str, timeout: int = 15) -> str:
    """
    Non-stream generate call (Ollama /api/generate).
    Returns "" on failure. Never raises.
    """
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": GEN_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 200}
            },
            timeout=timeout
        )
        # If status not ok, try to parse error
        if not r.ok:
            try:
                j = r.json()
                print(f"[Ollama generate HTTP {r.status_code}] {j}", flush=True)
            except Exception:
                print(f"[Ollama generate HTTP {r.status_code}] {r.text[:200]}", flush=True)
            return ""

        j = r.json()
        return j.get("response", "") or ""
    except Exception as e:
        print(f"[Ollama generate error] {e}", flush=True)
        return ""


def extract_filters_with_llm(user_query: str) -> dict:
    prompt = (
        f'Extract filters from: "{user_query}"\n'
        "Return JSON with keys: city, state, cuisines[], diets[] (e.g. 'Vegan options'), "
        "min_rating, max_price(1-4).\n"
        "Rules: 'Cheap' -> max_price:2.\n"
        "Output ONLY JSON."
    )

    filters = {}
    txt = _safe_ollama_generate(prompt, timeout=15)

    # Attempt to parse JSON object from model output
    try:
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            filters = json.loads(m.group(0))
    except Exception:
        filters = {}

    # Normalize list fields
    if isinstance(filters.get("cuisines"), str):
        filters["cuisines"] = [filters["cuisines"]]
    if isinstance(filters.get("diets"), str):
        filters["diets"] = [filters["diets"]]

    uq_lower = user_query.lower()

    # 1) Cuisine detection from constants
    if not filters.get("cuisines"):
        filters["cuisines"] = []
    existing_cuisines = {c.lower() for c in filters["cuisines"]}

    for cuisine in CUISINES_LIST:
        c = cuisine.lower()
        if c in uq_lower and c not in existing_cuisines:
            filters["cuisines"].append(cuisine)

    # 2) Diet detection from constants
    if not filters.get("diets"):
        filters["diets"] = []
    existing_diets = {d.lower() for d in filters["diets"]}

    for diet in DIETARY_OPTIONS:
        diet_lower = diet.lower()
        matched = False

        if diet_lower in uq_lower:
            matched = True
        elif "vegetarian" in diet_lower and ("vegetarian" in uq_lower or "veggie" in uq_lower):
            matched = True
        elif "vegan" in diet_lower and "vegan" in uq_lower:
            matched = True
        elif "gluten" in diet_lower and ("gluten" in uq_lower or "gf" in uq_lower or "celiac" in uq_lower):
            matched = True
        elif "halal" in diet_lower and "halal" in uq_lower:
            matched = True

        if matched and diet_lower not in existing_diets:
            filters["diets"].append(diet)

    # 3) Remove diet terms from cuisines, dedupe
    final_cuisines = []
    for c in filters["cuisines"]:
        c_norm = str(c).lower()
        is_diet = False
        for d in DIETARY_OPTIONS:
            d_norm = d.lower()
            if d_norm in c_norm or c_norm in d_norm:
                if d not in filters["diets"]:
                    filters["diets"].append(d)
                is_diet = True
        if not is_diet:
            final_cuisines.append(c)

    filters["cuisines"] = sorted(list(set(final_cuisines)))
    filters["diets"] = sorted(list(set(filters["diets"])))

    # 4) City fallback: "in/at <city ...>"
    city_match = re.search(r"\b(?:in|at)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)", user_query, re.IGNORECASE)
    if city_match:
        detected = city_match.group(1).strip()
        split_keywords = [" with", " where", " that", " for", " under", " above", " near"]
        for kw in split_keywords:
            if kw in detected.lower():
                detected = detected.lower().split(kw)[0].title()

        if detected.lower() not in ["cheap", "food", "place", "rating", "restaurant", "restaurants"]:
            filters["city"] = detected.title()

    # 5) Price fallback
    if "cheap" in uq_lower and not filters.get("max_price"):
        filters["max_price"] = 2

    # 6) Rating fallback
    rating_match = re.search(r"(?:rating|stars?|over|above|>)\s*(\d+(\.\d+)?)", user_query, re.IGNORECASE)
    if rating_match:
        try:
            val = float(rating_match.group(1))
            if 0 < val <= 5:
                filters["min_rating"] = val
        except Exception:
            pass

    return filters


def extract_filters_from_history(messages: List[dict]) -> dict:
    if len(messages) < 2:
        return {}
    user_queries = [m.get("content", "") for m in messages[:-1] if m.get("role") == "user"]
    if not user_queries:
        return {}
    combined_history = " ".join(user_queries)
    return extract_filters_with_llm(combined_history)


def is_rag_query(q: str) -> bool:
    RAG_KEYWORDS = ["restaurant", "restaurants", "food", "eat", "place", "where", "cheap", "dinner", "lunch", "breakfast", "cafe", "bar"]
    ql = (q or "").lower()
    return any(k in ql for k in RAG_KEYWORDS)


def contextualize_and_classify(messages: List[dict], model_tag: str) -> Tuple[str, bool, dict]:
    """
    Crash-proof query contextualizer:
    - tries Ollama generate to get JSON {rewritten, intent}
    - if Ollama returns non-JSON or errors -> fallback to heuristic
    """
    if not messages:
        return "", False, {}

    current_query = messages[-1].get("content", "") or ""
    if len(messages) < 2:
        return current_query, is_rag_query(current_query), {}

    historical_filters = extract_filters_from_history(messages)

    hist_txt = "\n".join([f"{m.get('role')}: {str(m.get('content',''))[:200]}" for m in messages[-6:]])

    prompt = (
        "Rewrite the user's last query to be standalone.\n"
        f"HISTORY:\n{hist_txt}\n\n"
        f"LAST QUERY: \"{current_query}\"\n\n"
        "Return ONLY JSON like:\n"
        "{\"rewritten\": \"...\", \"intent\": \"SEARCH\"}\n"
        "Use intent=SEARCH if it looks like a restaurant search, else intent=CHAT.\n"
    )

    resp_text = _safe_ollama_generate(prompt, timeout=15)
    if not resp_text:
        # Ollama failed -> fallback
        return current_query, is_rag_query(current_query), historical_filters

    try:
        match = re.search(r"\{.*\}", resp_text, re.S)
        if not match:
            print("[Contextualize Warning] LLM did not return JSON. Fallback to heuristic.", flush=True)
            return current_query, is_rag_query(current_query), historical_filters

        d = json.loads(match.group(0))
        rewritten = d.get("rewritten") or current_query
        intent = (d.get("intent") or "").upper()
        is_search = (intent == "SEARCH") or is_rag_query(rewritten)

        return rewritten, is_search, historical_filters

    except Exception as e:
        print(f"[Contextualize Error] {e}. Fallback to heuristic.", flush=True)
        return current_query, is_rag_query(current_query), historical_filters


def process_chat_request(
    q: str,
    is_search: bool,
    model_tag: str,
    top_k: int = 10,
    city=None,
    state=None,
    cuisines=None,
    diets=None,
    min_rating=None,
    max_price=None,
    historical_filters=None,
):
    if not is_search:
        return StreamingResponse(generate_ollama_chat_stream(model_tag, q), media_type="text/plain")

    # 1) Extract from current query first
    auto_filters = extract_filters_with_llm(q)

    city       = city or auto_filters.get("city")
    state      = state or auto_filters.get("state")
    cuisines   = cuisines or auto_filters.get("cuisines")
    diets      = diets or auto_filters.get("diets")

    # FIX: be careful with 0/None
    if min_rating is None:
        min_rating = auto_filters.get("min_rating")
    if max_price is None:
        max_price = auto_filters.get("max_price")

    # 2) Merge history to fill missing
    if historical_filters:
        if not city:
            city = historical_filters.get("city")
        if not state:
            state = historical_filters.get("state")
        if not cuisines:
            cuisines = historical_filters.get("cuisines")
        if not diets:
            diets = historical_filters.get("diets")
        if min_rating is None:
            min_rating = historical_filters.get("min_rating")
        if max_price is None:
            max_price = historical_filters.get("max_price")

    print(f"[FINAL SEARCH] City:{city}, State:{state}, Cuisines:{cuisines}, Diets:{diets}, Rating>={min_rating}, MaxPrice<={max_price}", flush=True)

    # Retrieval
    qvec = _model.encode([f"{E5_PREFIX}{q}"], normalize_embeddings=True)[0].tolist()
    flt = build_filter(city, state, cuisines, diets, min_rating, max_price)

    # Fetch more then hard filter
    points = qdrant_query(qvec, max(25, top_k), flt)
    points = apply_hard_filters(
        points,
        city=city,
        cuisines=cuisines,
        diets=diets,
        min_rating=min_rating,
        max_price=max_price,
    )

    ctx_text = build_context_block_visual(points[:5], max_items=5)
    if not ctx_text:
        ctx_text = "NO MATCHING RESTAURANTS FOUND."

    return StreamingResponse(
        generate_visual_response_stream(model_tag, q, ctx_text),
        media_type="text/plain"
    )


# ---------------- Schemas & Routes ----------------
class SearchResult(BaseModel):
    name: Optional[str]
    city: Optional[str]
    price: Optional[int]
    rating: Optional[float]


class ChatRequest(BaseModel):
    messages: List[dict]
    model: Optional[str] = GEN_MODEL
    filters: Optional[dict] = None


@app.post("/chat")
def chat_post(req: ChatRequest):
    try:
        q, is_search, hist = contextualize_and_classify(req.messages, req.model)
        f = req.filters or {}

        return process_chat_request(
            q=q,
            is_search=is_search,
            model_tag=req.model,
            top_k=int(f.get("top_k", 5)),
            city=f.get("city"),
            state=f.get("state"),
            cuisines=f.get("cuisines"),
            diets=f.get("diets"),
            min_rating=f.get("min_rating"),
            max_price=f.get("max_price"),
            historical_filters=hist,
        )

    except Exception:
        print("CRASH LOG:", traceback.format_exc(), flush=True)
        return StreamingResponse(
            iter(["I encountered a temporary system error. Please try asking again."]),
            media_type="text/plain"
        )


@app.get("/search", response_model=List[SearchResult])
def search_get(q: str, city: str = None, min_rating: float = None):
    qvec = _model.encode([f"{E5_PREFIX}{q}"], normalize_embeddings=True)[0].tolist()
    points = qdrant_query(qvec, 10, None)
    points = apply_hard_filters(points, city=city, min_rating=min_rating)

    res = []
    for p in points:
        pl = getattr(p, "payload", {}) or {}
        res.append(SearchResult(
            name=pl.get("name"),
            city=pl.get("city"),
            price=pl.get("price"),
            rating=pl.get("rating"),
        ))
    return res


@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/chat", "/search", "/device", "/healthz"]}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/device")
def device():
    return {"device": pick_device(), "model": MODEL_NAME, "collection": COLLECTION, "qdrant": f"{Q_HOST}:{Q_PORT}"}
