# app.py — FINAL VERSION (Stable & Error-Proof)
# Fixes: Prevents crash when LLM fails to return JSON in Contextualizer.

import os, time, traceback, re, json
from typing import List, Optional, Iterable, Tuple
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue, Range
import requests
from sentence_transformers import SentenceTransformer
from backend.constants import CUISINES_LIST, DIETARY_OPTIONS

# ---------------- Device / Model ----------------
def pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    except: pass
    return "cpu"

def load_model(name: str):
    dev = pick_device()
    print(f"[Embed] loading '{name}' on device: {dev}", flush=True)
    try:
        return SentenceTransformer(name, device=dev)
    except Exception as e:
        print(f"[Embed] failed on {dev}: {e} -> falling back to CPU", flush=True)
        return SentenceTransformer(name, device="cpu")

# Config
COLLECTION  = os.getenv("Q_COLLECTION", "restaurants")
MODEL_NAME  = os.getenv("EMBED_MODEL", "intfloat/e5-small-v2")
GEN_MODEL   = os.getenv("GEN_MODEL",  "llama3.1:8b-instruct-q4_K_M")
OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
E5_PREFIX   = os.getenv("E5_PREFIX", "query: ")
INTENT_MODEL_DIR = os.getenv("INTENT_MODEL_DIR", "bert_intent_multilabel_model")

# Start
print("\n### SYSTEM START: STABLE VERSION (JSON Safety Added) ###\n", flush=True)

app    = FastAPI(title="Byte&Bite Search API")
_model = load_model(MODEL_NAME)
_cli   = QdrantClient(host=os.getenv("Q_HOST","127.0.0.1"), port=int(os.getenv("Q_PORT","6333")))

# ---------------- BERT Intent Classifier ----------------
_intent_model = None
_intent_tokenizer = None
_intent_id2tag = None
_intent_max_length = 64
_intent_threshold = 0.5

def load_intent_classifier():
    global _intent_model, _intent_tokenizer, _intent_id2tag
    if _intent_model is not None: return _intent_model, _intent_tokenizer, _intent_id2tag
    try:
        from transformers import BertForSequenceClassification, BertTokenizerFast
        import torch
        model_path = Path(INTENT_MODEL_DIR)
        if not model_path.exists(): return None, None, None
        _intent_model = BertForSequenceClassification.from_pretrained(str(model_path))
        _intent_tokenizer = BertTokenizerFast.from_pretrained(str(model_path))
        tags_file = model_path / "all_tags.npy"
        if tags_file.exists():
            all_tags = np.load(str(tags_file), allow_pickle=True)
            _intent_id2tag = {i: tag for i, tag in enumerate(all_tags)}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _intent_model.to(device)
        _intent_model.eval()
        return _intent_model, _intent_tokenizer, _intent_id2tag
    except Exception as e:
        print(f"[Intent] Warning: {e}")
        return None, None, None

def get_bert_intents(query: str, threshold: float = None) -> List[Tuple[str, float]]:
    if not query or not query.strip(): return []
    model, tokenizer, id2tag = load_intent_classifier()
    if model is None: return []
    try:
        import torch
        encodings = tokenizer(query, padding=True, truncation=True, max_length=_intent_max_length, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**encodings).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        thr = threshold if threshold is not None else _intent_threshold
        indices = np.where(probs >= thr)[0]
        results = [(id2tag[i], float(probs[i])) for i in indices]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    except: return []

# ---------------- Helpers ----------------
def build_filter(city, state, cuisines, diets, min_rating, max_price) -> Optional[Filter]:
    conds = []
    if city:      conds.append(FieldCondition(key="city",    match=MatchValue(value=city)))
    if state:     conds.append(FieldCondition(key="state",   match=MatchValue(value=state)))
    if cuisines:  conds.append(FieldCondition(key="cuisines", match=MatchAny(any=cuisines)))
    if diets:     conds.append(FieldCondition(key="diets",    match=MatchAny(any=diets)))
    if min_rating is not None:
        conds.append(FieldCondition(key="rating", range=Range(gte=float(min_rating))))
    if max_price is not None:
        conds.append(FieldCondition(key="price", range=Range(lte=int(max_price))))

    return Filter(must=conds) if conds else None

def is_open_now(payload: dict) -> bool:
    from datetime import datetime
    dow_map = ["mon","tue","wed","thu","fri","sat","sun"]
    now = datetime.now(); dow = dow_map[now.weekday()]
    minutes = now.hour*60 + now.minute
    hours = (payload or {}).get("open", {}).get(dow, [])
    for s,e in hours:
        if s <= minutes <= e: return True
        if e > 1440 and (s <= minutes+1440 <= e): return True
    return False

def qdrant_query(qvec, limit, flt):
    try:
        resp = _cli.query_points(collection_name=COLLECTION, query=qvec, limit=limit, with_payload=True, filter=flt)
        return resp.points or []
    except:
        resp = _cli.search(collection_name=COLLECTION, query_vector=qvec, limit=limit, with_payload=True, query_filter=flt)
        return resp or []

def apply_hard_filters(points, city=None, cuisines=None, diets=None, min_rating=None, max_price=None):
    out = []
    for p in points:
        pl = getattr(p, "payload", {}) or {}
        ok = True
        
        # City Check (Fuzzy)
        if city:
            p_city = str(pl.get("city") or "").lower().strip()
            t_city = str(city).lower().strip()
            if t_city not in p_city: ok = False
            
        if cuisines:
            p_cuisines = [str(c).lower() for c in (pl.get("cuisines") or [])]
            t_cuisines = [str(c).lower() for c in cuisines]
            if not any(tc in pc for tc in t_cuisines for pc in p_cuisines): ok = False
        
        # Diet Check
        if diets:
            p_diets = [str(d).lower() for d in (pl.get("diets") or [])]
            for req in diets:
                req_lower = req.lower()
                keyword = "vegan" if "vegan" in req_lower else \
                          "vegetarian" if "vegetarian" in req_lower else \
                          "gluten" if "gluten" in req_lower else req_lower
                if not any(keyword in avail for avail in p_diets):
                    ok = False
                    break

        if min_rating is not None and float(pl.get("rating") or 0.0) < float(min_rating): ok = False
        
        if max_price is not None:
            price = pl.get("price")
            if price is None or int(price) > int(max_price): ok = False
            
        if ok: out.append(p)
    return out

# ---------------- Visual Card Logic ----------------

def build_context_block_visual(points, max_items=5):
    lines = []
    seen = set()
    price_map = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}

    for p in points:
        pl = getattr(p, "payload", {}) or {}
        name = pl.get("name")
        if not name or name in seen: continue
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
        if len(lines) >= max_items: break
    return "\n".join(lines)

def generate_visual_response_stream(model_tag: str, query: str, context: str, intents: List[Tuple[str, float]] = None) -> Iterable[str]:
    if intents:
        intent_str = ", ".join([f"{tag} ({score:.2f})" for tag, score in intents])
        yield f"**Detected Intents:** {intent_str}\n\n"
    else:
        yield f"**Detected Intents:** None detected (below threshold)\n\n"

    system_prompt = (
        "You are Byte & Bite, a helpful restaurant assistant. "
        "Your task is to format the provided restaurant data into clean, readable cards. "
        "You MUST ONLY use the data provided in [RAW DATA]. Do not invent or list restaurants not present in [RAW DATA]. Don not use any of your previous knowledge or information. "
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
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], 
        "stream": True, 
        "options": {"temperature": 0.1}
    }

    try:
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=120) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        d = json.loads(line)
                        if "message" in d: yield d["message"]["content"]
                        elif "response" in d: yield d["response"]
                    except: pass
    except Exception as e: yield f"Error: {e}"

def generate_ollama_chat_stream(model_tag: str, user_query: str, intents: List[Tuple[str, float]] = None) -> Iterable[str]:
    if intents:
        intent_str = ", ".join([f"{tag} ({score:.2f})" for tag, score in intents])
        yield f"**Detected Intents:** {intent_str}\n\n"
    else:
         yield f"**Detected Intents:** None detected\n\n"

    try:
        with requests.post(f"{OLLAMA_URL}/api/chat", json={"model": model_tag, "messages": [{"role": "user", "content": user_query}], "stream": True}) as r:
            for line in r.iter_lines():
                if line:
                    d = json.loads(line)
                    if "message" in d: yield d["message"]["content"]
    except: yield "Chat Error"

# ---------------- Filter Extraction (Logic Fixed) ----------------

def extract_filters_with_llm(user_query: str, bert_intents: List[Tuple[str, float]] = None) -> dict:
    if bert_intents is None:
        bert_intents = get_bert_intents(user_query)
    intent_guidance = ""
    if bert_intents:
        relevant = [l for l, c in bert_intents if l != "OUT_OF_SCOPE"]
        if relevant: intent_guidance = f"Intents: {', '.join(relevant)}"

    prompt = f"""Extract filters from: "{user_query}" {intent_guidance}\nReturn JSON: city, state, cuisines[], diets[] (e.g. "Vegan options"), min_rating, max_price(1-4).\nRules: "Cheap"->max_price:2."""
    
    filters = {}
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json={"model": GEN_MODEL, "prompt": prompt, "stream": False, "options":{"num_predict":200}}, timeout=10)
        txt = r.json().get("response","")
        m = re.search(r"\{.*\}", txt, re.S)
        if m: filters = json.loads(m.group(0))
    except: pass

    # Normalize
    if isinstance(filters.get("cuisines"), str): filters["cuisines"] = [filters["cuisines"]]
    if isinstance(filters.get("diets"), str): filters["diets"] = [filters["diets"]]
    
    # --- REGEX / KEYWORD MATCHING FIXES ---
    uq_lower = user_query.lower()
    
    # 1. Enhanced Cuisine Detection (from constants)
    if not filters.get("cuisines"): filters["cuisines"] = []
    existing_cuisines = {c.lower() for c in filters["cuisines"]}
    
    for cuisine in CUISINES_LIST:
        if cuisine.lower() in uq_lower and cuisine.lower() not in existing_cuisines:
            filters["cuisines"].append(cuisine)

    # 2. Enhanced Diet Detection (from constants)
    if not filters.get("diets"): filters["diets"] = []
    existing_diets = {d.lower() for d in filters["diets"]}

    for diet in DIETARY_OPTIONS:
        diet_lower = diet.lower()
        matched = False
        
        # Check full match
        if diet_lower in uq_lower:
            matched = True
        # Check keywords
        elif "vegetarian" in diet_lower and ("vegetarian" in uq_lower or "veggie" in uq_lower):
            matched = True
        elif "vegan" in diet_lower and "vegan" in uq_lower:
            matched = True
        elif "gluten" in diet_lower and ("gluten" in uq_lower or "gf" in uq_lower or "celiac" in uq_lower):
            matched = True
        elif "halal" in diet_lower and "halal" in uq_lower:
            matched = True
            
        if matched and diet.lower() not in existing_diets:
            filters["diets"].append(diet)

    # 3. Sanitize & Deduplicate
    # Remove diet terms from cuisines if they appear in diets or are diet keywords
    final_cuisines = []
    for c in filters["cuisines"]:
        c_norm = c.lower()
        # If it's a diet keyword, ensure it's in diets and skip adding to cuisines
        is_diet = False
        for d in DIETARY_OPTIONS:
            d_norm = d.lower()
            if d_norm in c_norm or c_norm in d_norm: # Fuzzy match "vegan" in "vegan options"
                 if d not in filters["diets"]: filters["diets"].append(d)
                 is_diet = True
        
        if not is_diet:
            final_cuisines.append(c)
            
    filters["cuisines"] = list(set(final_cuisines)) # Dedupe
    filters["diets"] = list(set(filters["diets"])) # Dedupe

    # 4. City Fallback
    # Stop at common prepositions/keywords to prevent over-capture (e.g. "Austin with...")
    city_match = re.search(r"\b(?:in|at)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)", user_query, re.IGNORECASE)
    if city_match:
        detected = city_match.group(1).strip()
        # Truncate if we hit keywords like "with", "where", "that"
        split_keywords = [" with", " where", " that", " for", " under", " above", " near"]
        for kw in split_keywords:
            if kw in detected.lower():
                detected = detected.lower().split(kw)[0].title()
                
        if detected.lower() not in ["cheap", "food", "place", "rating", "restaurant", "restaurants"]:
            filters["city"] = detected.title()

    # 5. Price Fallback (Explicit "Cheap" Check)
    if "cheap" in uq_lower and not filters.get("max_price"):
        filters["max_price"] = 2


    rating_match = re.search(r"(?:rating|stars?|over|above|>)\s*(\d+(\.\d+)?)", user_query, re.IGNORECASE)
    if rating_match:
        try:
            val = float(rating_match.group(1))
            if 0 < val <= 5: filters["min_rating"] = val
        except: pass

    # Diet Regex
    # (Removed legacy block as it is covered above)
    # uq_lower = user_query.lower()
    # if not filters.get("diets"): filters["diets"] = []
    # if "vegan" in uq_lower... 
    
    return filters

def extract_filters_from_history(messages: List[dict], model_tag: str) -> dict:
    if len(messages) < 2: return {}
    user_queries = [m.get("content", "") for m in messages[:-1] if m.get("role") == "user"]
    if not user_queries: return {}
    combined_history = " ".join(user_queries)
    try:
        return extract_filters_with_llm(combined_history)
    except: return {}

def is_rag_query(q):
    RAG_KEYWORDS = ["restaurant", "food", "eat", "place", "where", "cheap", "dinner", "lunch"]
    return any(k in q.lower() for k in RAG_KEYWORDS)

def contextualize_and_classify(messages: List[dict], model_tag: str) -> Tuple[str, bool, dict]:
    """CRASH-PROOF REWRITER"""
    if not messages: return "", False, {}
    current_query = messages[-1]["content"]
    if len(messages) < 2: return current_query, is_rag_query(current_query), {}

    historical_filters = extract_filters_from_history(messages, model_tag)
    
    hist_txt = "\n".join([f"{m['role']}: {m['content'][:200]}" for m in messages[-6:]])
    prompt = f"""Rewrite query to be standalone. History: {hist_txt}. Last: "{current_query}". Return JSON {{'rewritten': '...', 'intent': 'SEARCH'}}"""
    
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json={"model": GEN_MODEL, "prompt": prompt, "stream": False}, timeout=15)
        resp_text = r.json()["response"]
        
        # GÜVENLİK: Eğer regex bulamazsa, raw text'i dön, patlama.
        match = re.search(r"\{.*\}", resp_text, re.S)
        if match:
            d = json.loads(match.group(0))
            return d.get("rewritten", current_query), (d.get("intent")=="SEARCH"), historical_filters
        else:
            print(f"[Contextualize Warning] LLM did not return JSON. Using raw query.")
            return current_query, True, historical_filters
    except Exception as e:
        print(f"[Contextualize Error] {e}")
        return current_query, True, historical_filters

def process_chat_request(q, is_search, model_tag, top_k=10, city=None, state=None, cuisines=None, diets=None, min_rating=None, max_price=None, open_now=False, historical_filters=None):
    # BERT Intents calculated first for both flows
    bert_intents = get_bert_intents(q)
    print(f"[BERT Intents] Query: {q} -> Intents: {bert_intents}", flush=True)

    # 0. Check BERT Intent for Out of Scope
    if any(intent == "OUT_OF_SCOPE" for intent, score in bert_intents):
         intent_str = ", ".join([f"{tag} ({score:.2f})" for tag, score in bert_intents])
         return StreamingResponse(iter([f"**Detected Intents:** {intent_str}\n\nI apologize, but I am designed to assist with restaurant-related queries only. This request is out of my scope."]), media_type="text/plain")

    if not is_search:
        return StreamingResponse(generate_ollama_chat_stream(model_tag, q, intents=bert_intents), media_type="text/plain")

    # --- FIX: CURRENT QUERY FIRST, THEN MERGE HISTORY ---
    auto_filters = extract_filters_with_llm(q, bert_intents=bert_intents)
    
    city = city or auto_filters.get("city")
    cuisines = cuisines or auto_filters.get("cuisines")
    diets = diets or auto_filters.get("diets") # Added diets
    min_rating = min_rating or auto_filters.get("min_rating")
    max_price = max_price or auto_filters.get("max_price")

    # 2. Merge History (Fill Gaps)
    if historical_filters:
        if not city: city = historical_filters.get("city")
        if not cuisines: cuisines = historical_filters.get("cuisines")
        if not diets: diets = historical_filters.get("diets") # Added diets merge
        if min_rating is None: min_rating = historical_filters.get("min_rating")
        if max_price is None: max_price = historical_filters.get("max_price")

    print(f"[FINAL SEARCH] City:{city}, Diets:{diets}, Rating>={min_rating}", flush=True)

    # Retrieval
    qvec = _model.encode([f"{E5_PREFIX}{q}"], normalize_embeddings=True)[0].tolist()
    flt = build_filter(city, state, cuisines, diets, min_rating, max_price)
    
    limit = max(int(top_k) * 5, 25)
    points = qdrant_query(qvec, limit, flt)
    points = apply_hard_filters(points, city=city, cuisines=cuisines, diets=diets, min_rating=min_rating, max_price=max_price)
    
    # Sort by Rating (High to Low)
    points.sort(key=lambda p: float((getattr(p, "payload", {}) or {}).get("rating") or 0.0), reverse=True)

    # Görsel Kartlar
    ctx_text = build_context_block_visual(points[:5], max_items=5)
    if not ctx_text: ctx_text = "NO MATCHING RESTAURANTS FOUND."

    return StreamingResponse(generate_visual_response_stream(model_tag, q, ctx_text, intents=bert_intents), media_type="text/plain")

# ---------------- Schemas & Routes ----------------
class SearchResult(BaseModel):
    name: Optional[str]; city: Optional[str]; price: Optional[int]; rating: Optional[float]

class ChatRequest(BaseModel):
    messages: List[dict]; model: Optional[str] = GEN_MODEL; filters: Optional[dict] = None

@app.post("/chat")
def chat_post(req: ChatRequest):
    # ANA ENDPOINT İÇİN GÜVENLİK
    try:
        q, is_search, hist = contextualize_and_classify(req.messages, req.model)
        f = req.filters or {}
        return process_chat_request(
            q=q, is_search=is_search, model_tag=req.model,
            top_k=int(f.get('top_k', 5)), city=f.get('city'), cuisines=f.get('cuisines'),
            diets=f.get('diets'), # Pass from frontend if exists
            min_rating=f.get('min_rating'), max_price=f.get('max_price'),
            historical_filters=hist
        )
    except Exception as e:
        print("CRASH LOG:", traceback.format_exc())
        return StreamingResponse(iter([f"I encountered a temporary system error. Please try asking again."]), media_type="text/plain")

# --- RESTORED /search ENDPOINT ---
@app.get("/search", response_model=List[SearchResult])
def search_get(q: str, city: str = None, min_rating: float = None):
    qvec = _model.encode([f"{E5_PREFIX}{q}"], normalize_embeddings=True)[0].tolist()
    points = qdrant_query(qvec, 10, None)
    points = apply_hard_filters(points, city=city, min_rating=min_rating)
    res = []
    for p in points:
        pl = p.payload
        res.append(SearchResult(name=pl.get("name"), city=pl.get("city"), price=pl.get("price"), rating=pl.get("rating")))
    return res

@app.get("/")
def root(): return {"status": "ok", "endpoints": ["/chat", "/search", "/device"]}

@app.get("/healthz")
def healthz(): return {"status": "ok"}

@app.get("/device")
def device():
    dev = pick_device()
    return {"device": dev, "model": MODEL_NAME}
