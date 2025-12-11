# app.py — RAG search + RAG chat (Ollama), Qdrant version-compatible,
#          auto filter extraction + rating override + hard post-filter,
#          e5-friendly "query:" prefix for embeddings.
<<<<<<< Updated upstream

import os, time, traceback, re, json
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
=======
#
# UPDATED: 
# 1. Strict separation of Semantic Adjectives (Healthy, Romantic) vs Hard Filters (Italian, NY).
# 2. Contextualizer now carries over location from history forcefully.
# 3. Diet extraction logic tightened to prevent hallucinations.

import os, time, traceback, re, json
from typing import List, Optional, Iterable, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import StreamingResponse
>>>>>>> Stashed changes
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue, Range
import requests

# ---------------- Device / Model ----------------
def pick_device() -> str:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
            return "mps"
    except Exception:
        pass
    return "cpu"

def load_model(name: str):
    from sentence_transformers import SentenceTransformer
    dev = pick_device()
    print(f"[Embed] loading '{name}' on device: {dev}", flush=True)
    try:
        return SentenceTransformer(name, device=dev)
    except Exception as e:
        print(f"[Embed] failed on {dev}: {e} -> falling back to CPU", flush=True)
        return SentenceTransformer(name, device="cpu")

COLLECTION  = os.getenv("Q_COLLECTION", "restaurants")
MODEL_NAME  = os.getenv("EMBED_MODEL", "intfloat/e5-small-v2")
GEN_MODEL   = os.getenv("GEN_MODEL",  "llama3.1:8b-instruct-q4_K_M")  # Ollama tag
GEN_SEED    = os.getenv("GEN_SEED")
OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

# When using e5, it's best to prefix with "query: "
E5_PREFIX   = os.getenv("E5_PREFIX", "query: ")

<<<<<<< Updated upstream
=======
RAG_KEYWORDS = [
    "restaurant", "suggest", "recommend", "give", "where", "place", "eat",
    "food", "cuisine", "dining", "cheap", "affordable", "price", "rating",
    "vegan", "gluten", "vegetarian", "options", "healthy", "spicy", "breakfast", "dinner"
]

SIMPLE_CHAT_PROMPT = (
    "You are a helpful assistant. Your primary purpose is to help users find restaurants. "
    "If the user asks what you are or who you are, identify yourself as a restaurant recommendation assistant. "
    "For other general conversational queries (like 'Hi', 'How are you?'), answer politely and briefly. "
    "Always answer in plain text. Do not use Markdown, bullet points, or bolding."
)


>>>>>>> Stashed changes
app    = FastAPI(title="Byte&Bite Search API")
_model = load_model(MODEL_NAME)
_cli   = QdrantClient(host=os.getenv("Q_HOST","127.0.0.1"), port=int(os.getenv("Q_PORT","6333")))

# ---------------- Helpers ----------------
def build_filter(
    city: Optional[str], state: Optional[str],
    cuisines: Optional[List[str]], diets: Optional[List[str]],
    min_rating: Optional[float], max_price: Optional[int],
) -> Optional[Filter]:
    conds = []
    if city:      conds.append(FieldCondition(key="city",    match=MatchValue(value=city)))
    if state:     conds.append(FieldCondition(key="state",   match=MatchValue(value=state)))
    if cuisines:  conds.append(FieldCondition(key="cuisines", match=MatchAny(any=cuisines)))
    if diets:     conds.append(FieldCondition(key="diets",    match=MatchAny(any=diets)))
    if min_rating is not None: conds.append(FieldCondition(key="rating", range=Range(gte=min_rating)))
    if max_price  is not None: conds.append(FieldCondition(key="price",  range=Range(lte=max_price)))
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
    """Try query_points; fallback to search() for older clients."""
    try:
        resp = _cli.query_points(
            collection_name=COLLECTION,
            query=qvec, limit=limit, with_payload=True, filter=flt
        )
        return resp.points or []
    except Exception:
        resp = _cli.search(
            collection_name=COLLECTION,
            query_vector=qvec, limit=limit, with_payload=True, query_filter=flt
        )
        return resp or []

# ---- RAG LLM helpers ----
def build_context_block(points, max_items=15):
    lines, rows = [], []
<<<<<<< Updated upstream
    for p in points[:max_items]:
        pl = getattr(p, "payload", {}) or {}
        rows.append({
            "name": pl.get("name"), "city": pl.get("city"), "state": pl.get("state"),
            "price": pl.get("price"), "rating": pl.get("rating"),
            "cuisines": pl.get("cuisines"), "diets": pl.get("diets"),
            "website": pl.get("website"), "id": pl.get("_id")
        })
        lines.append(
            f"- {pl.get('name')} | city={pl.get('city')} state={pl.get('state')} "
            f"price={pl.get('price')} rating={pl.get('rating')} "
            f"cuisines={pl.get('cuisines')} diets={pl.get('diets')} website={pl.get('website')} id={pl.get('_id')}"
        )
    return "\n".join(lines), rows

def is_restaurant_search_query(user_query: str) -> bool:
    """Detect if the query is actually a restaurant search vs a greeting/conversational query."""
    query_lower = user_query.lower().strip()
    
    # Common greetings and conversational phrases
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", 
                 "how are you", "what's up", "thanks", "thank you", "bye", "goodbye", 
                 "see you", "ok", "okay", "sure", "yes", "no"]
    
    # If it's just a greeting, it's not a restaurant search
    if query_lower in greetings or len(query_lower.split()) <= 2:
        # Check if it contains restaurant-related keywords
        restaurant_keywords = ["restaurant", "food", "dining", "eat", "cuisine", "menu", 
                              "reservation", "table", "rating", "price", "cheap", "expensive",
                              "gluten", "vegan", "vegetarian", "diet", "breakfast", "lunch", 
                              "dinner", "brunch", "pizza", "burger", "sushi", "italian", 
                              "chinese", "mexican", "french", "japanese", "indian", "thai"]
        
        # If it's a short query without restaurant keywords, it's likely a greeting
        if not any(keyword in query_lower for keyword in restaurant_keywords):
            return False
    
    # If query contains restaurant-related terms or location terms, it's a search
    search_indicators = ["find", "search", "recommend", "suggest", "show", "list", 
                        "where", "what", "which", "best", "good", "near", "in", "at"]
    
    # Check if it's a question about restaurants
    if any(indicator in query_lower for indicator in search_indicators):
        return True
    
    # If it mentions a city, cuisine, or dietary requirement, it's likely a search
    if any(keyword in query_lower for keyword in ["city", "cuisine", "diet", "rating", "price", 
                                                   "cheap", "budget", "expensive", "affordable"]):
        return True
    
    # Default: if query is longer than 3 words, assume it's a search
    return len(query_lower.split()) > 3

def rag_prompt(user_query: str, context_text: str) -> str:
    return f"""You are a helpful restaurant recommender assistant. 

IMPORTANT: 
- If the user query is a greeting (like "hi", "hello") or not asking about restaurants, respond naturally and ask how you can help with restaurant recommendations.
- If the user is asking about restaurants, use ONLY the info in CONTEXT to provide recommendations.
- If CONTEXT is empty or not relevant, politely say you couldn't find matching restaurants and suggest refining the search (e.g., specify a city, cuisine, or relax filters).

When providing restaurant recommendations:
- Return 3–5 bullets, best-first
- Each bullet: name, short reason, price level, rating, and end with (source: name or website)
- Be concise and helpful
=======
    seen_restaurants = set()
    
    for p in points:
        pl = getattr(p, "payload", {}) or {}
        name = pl.get("name")
        city = pl.get("city")
        
        if not name or not city:
            continue

        unique_key = (name.lower().strip(), city.lower().strip())
        
        if unique_key not in seen_restaurants:
            seen_restaurants.add(unique_key)
            
            rows.append({
                "name": pl.get("name"), "city": pl.get("city"), "state": pl.get("state"),
                "price": pl.get("price"), "rating": pl.get("rating"),
                "cuisines": pl.get("cuisines"), "diets": pl.get("diets"),
                "website": pl.get("website"), "id": pl.get("_id")
            })
            lines.append(
                f"- {pl.get('name')} | city={pl.get('city')} state={pl.get('state')} "
                f"price={pl.get('price')} rating={pl.get('rating')} "
                f"cuisines={pl.get('cuisines')} diets={pl.get('diets')} website={pl.get('website')} id={pl.get('_id')}"
            )
        
        if len(rows) >= max_items:
            break
            
    return "\n".join(lines), rows

# ----

def is_rag_query(user_query: str) -> bool:
    """Detect if the query is a restaurant search based on keywords."""
    query_lower = user_query.lower().strip()
    if not query_lower:
        return False
    return any(keyword in query_lower for keyword in RAG_KEYWORDS)

def rag_prompt(user_query: str, context_text: str) -> str:
    return f"""You are a restaurant recommender. You MUST ONLY use restaurants from the CONTEXT below.

CRITICAL RULES:
1. ONLY list restaurants that appear EXACTLY in the CONTEXT below.
2. Do NOT invent, create, or mention ANY restaurant that is NOT in the CONTEXT.
3. Use EXACT restaurant names, prices, and ratings as shown in the CONTEXT.
4. If you know a restaurant name but it's NOT in the CONTEXT, it does NOT exist - do NOT mention it.

OUTPUT FORMAT:
- One restaurant per line.
- Format: Restaurant Name - Price: X, Rating: Y - Brief description
- Only list restaurants from the CONTEXT that match the query.
- If no restaurants match, say: "I couldn't find matching restaurants."
>>>>>>> Stashed changes

USER QUERY: {user_query}

CONTEXT (ONLY these restaurants exist - do not mention others):
{context_text}

Response (start with restaurant name, no prefix):
"""

<<<<<<< Updated upstream
def generate_with_ollama(model_tag: str, prompt: str, temperature: float = 0.2, max_tokens: int = 350) -> str:
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
=======
def generate_ollama_rag_stream(model_tag: str, prompt: str, temperature: float = 0.1, max_tokens: int = 400) -> Iterable[str]:
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "stream": True,
>>>>>>> Stashed changes
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }
    if GEN_SEED is not None:
        try:
            payload["options"]["seed"] = int(GEN_SEED)
        except ValueError:
            pass
<<<<<<< Updated upstream
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()

# ---- Auto filter extraction (LLM) + heuristics ----
def extract_filters_with_llm(user_query: str) -> dict:
    """Ask the generator to extract filters as JSON, then apply regex overrides."""
=======
    
    # Buffer to collect initial chunks and strip prefixes
    first_chunk = True
    buffer = ""
    
    try:
        with requests.post(f"{OLLAMA_URL}/api/generate", json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if "response" in data:
                            chunk = data["response"]
                            
                            if first_chunk:
                                buffer += chunk
                                # Once we have enough content, check for prefixes
                                if len(buffer) > 10 or "\n" in buffer:
                                    buffer_lower = buffer.lower().strip()
                                    # Remove common prefixes
                                    prefixes = ["ai\n", "ai ", "assistant:", "here are", "here's", "here is"]
                                    for prefix in prefixes:
                                        if buffer_lower.startswith(prefix):
                                            # Remove prefix and any following whitespace
                                            idx = buffer.lower().find(prefix)
                                            if idx == 0:
                                                buffer = buffer[len(prefix):].lstrip()
                                            break
                                    first_chunk = False
                                    if buffer:
                                        yield buffer
                                        buffer = ""
                            else:
                                yield chunk
                                
                        if data.get("done", False):
                            if buffer:
                                yield buffer
                            break
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        print(f"[Stream] Ollama request failed: {e}")
        yield f"Error: Could not connect to Ollama. {e}"

def generate_ollama_chat_stream(model_tag: str, user_query: str) -> Iterable[str]:
    payload = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": SIMPLE_CHAT_PROMPT},
            {"role": "user", "content": user_query}
        ],
        "stream": True,
        "options": {"temperature": 0.3}
    }

    try:
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if data.get("message", {}).get("content"):
                            yield data["message"]["content"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        print(f"[Stream] Ollama chat request failed: {e}")
        yield f"Error: Could not connect to Ollama. {e}"


# ---- Auto filter extraction (LLM) + heuristics ----

def extract_filters_with_llm(user_query: str) -> dict:
    """Ask the generator to extract filters as JSON. Strict separation of Semantic vs Hard filters."""
    
    # PROMPT UPDATED for better semantic handling
>>>>>>> Stashed changes
    prompt = f"""
Extract restaurant search filters from the user query.

User query: "{user_query}"

Return STRICT JSON with keys:
city (string or null),
state (string or null),
cuisines (list of strings or null),
diets (list of strings or null),
min_rating (number or null),
max_price (integer 1-4 or null for $-$$$$).

<<<<<<< Updated upstream
Rules:
- If the query says "5.0 rating" or "at least 4.7", set min_rating to that exact number.
- If the query says "under $$", "cheap", "budget", map to max_price=2. If "under $$$" -> 3, etc.
- If multiple cuisines appear, include them all.
- For diets: map "gluten free", "gluten-free", "gluten free options" -> "Gluten free options"
- For diets: map "vegan", "vegan options" -> "Vegan options"
- For diets: map "vegetarian", "vegetarian friendly", "vegetarian options" -> "Vegetarian friendly"
- If the query mentions dietary requirements, extract them into the diets list.
=======
IMPORTANT RULES:
1. **Adjectives are NOT Cuisines:** Do NOT treat words like "healthy", "good", "best", "romantic", "spicy", "cheap", "fast" as cuisines.
   - INCORRECT: cuisines: ["Healthy"] 
   - CORRECT: cuisines: null (Let the search engine handle "healthy")
2. **Cuisines must be Categories:** Only use specific categories like "Italian", "Chinese", "Pizza", "Burger", "Sushi", "Indian", "Turkish", "Mediterranean", "Mexican", "Thai", etc.
3. **Negations:** If user says "No Pizza" or "Not Italian", do NOT include "Pizza" or "Italian" in the cuisines list. Leave it null.
4. **Price:** "cheap", "affordable", "budget" -> max_price: 2. "under $$$" -> 3.
5. **Diets:** Only map explicit dietary restrictions.
   - "gluten free" -> "Gluten free options"
   - "vegan" -> "Vegan options"
   - "vegetarian" -> "Vegetarian friendly"
   - Ignore other adjectives for diets.
>>>>>>> Stashed changes

Example 1: "Healthy food in New York"
-> {{"city":"New York", "state":null, "cuisines":null, "diets":null, "min_rating":null, "max_price":null}}

Example 2: "Cheap Italian place"
-> {{"city":null, "state":null, "cuisines":["Italian"], "diets":null, "min_rating":null, "max_price":2}}

Answer ONLY with JSON.
"""
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.0, "num_predict": 220}},
        timeout=30
    )
    r.raise_for_status()
    txt = r.json().get("response","").strip()
    m = re.search(r"\{.*\}", txt, re.S)
    result = {}
    if m:
        try:
            result = json.loads(m.group(0))
        except Exception:
            result = {}

    # normalize list types
    if result.get("cuisines") is not None and not isinstance(result["cuisines"], list):
        result["cuisines"] = [str(result["cuisines"])]
    if result.get("diets") is not None and not isinstance(result["diets"], list):
        result["diets"] = [str(result["diets"])]
    
    # normalize diet values strict check
    if result.get("diets"):
        normalized_diets = []
        diet_mapping = {
            "gluten free": "Gluten free options",
            "gluten-free": "Gluten free options",
            "gluten free options": "Gluten free options",
            "vegan": "Vegan options",
            "vegan options": "Vegan options",
            "vegetarian": "Vegetarian friendly",
            "vegetarian friendly": "Vegetarian friendly",
            "vegetarian options": "Vegetarian friendly"
        }
        for d in result["diets"]:
            d_lower = str(d).lower().strip()
            # Strict mapping to avoid hallucinations like "Healthy" becoming a diet
            if d in ["Gluten free options", "Vegan options", "Vegetarian friendly"]:
                normalized_diets.append(d)
            elif d_lower in diet_mapping:
                normalized_diets.append(diet_mapping[d_lower])
            elif "gluten" in d_lower and "free" in d_lower:
                normalized_diets.append("Gluten free options")
            elif "vegan" in d_lower:
                normalized_diets.append("Vegan options")
            elif "vegetarian" in d_lower:
                normalized_diets.append("Vegetarian friendly")
            # intentionally dropped the 'else keep' clause to be safe against hallucinations
            
        result["diets"] = list(set(normalized_diets))

    # numeric sanitation
    try:
        if result.get("max_price") is not None:
            result["max_price"] = int(result["max_price"])
    except Exception:
        result["max_price"] = None
    try:
        if result.get("min_rating") is not None:
            result["min_rating"] = float(result["min_rating"])
    except Exception:
        result["min_rating"] = None

    return result

def rating_from_text(user_query: str) -> Optional[float]:
    """Extract explicit rating constraints from raw text."""
    m = re.search(r"(?:rating|rated|at\s+least)\s*([1-5](?:\.[0-9])?)", user_query, re.I)
    if not m:
        m = re.search(r"([1-5](?:\.[0-9])?)\s*-?\s*star", user_query, re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def apply_hard_filters(points, city=None, cuisines=None, diets=None, min_rating=None, max_price=None):
    """Client-side guarantee that results satisfy filters exactly."""
    out = []
    for p in points:
        pl = getattr(p, "payload", {}) or {}
        ok = True
        if city and (pl.get("city") or "") != city:
            ok = False
        if cuisines:
            pc = pl.get("cuisines") or []
            # Standard "Any" match.
            if not any(c in pc for c in cuisines):
                ok = False
        if diets:
            pd = pl.get("diets") or []
            if not any(d in pd for d in diets):
                ok = False
        if min_rating is not None and float(pl.get("rating") or 0.0) < float(min_rating):
            ok = False
        if max_price is not None:
            price = pl.get("price")
            if price is None or int(price) > int(max_price):
                ok = False
        if ok:
            out.append(p)
    return out

# ---------------- Schemas ----------------
class SearchResult(BaseModel):
    name: Optional[str]; score: float
    city: Optional[str]; state: Optional[str]
    price: Optional[int]; rating: Optional[float]
    cuisines: Optional[List[str]]; website: Optional[str]
    id: Optional[int]

<<<<<<< Updated upstream
class ChatResponse(BaseModel):
    answer: str
    used_sources: List[dict]

# ---------------- Health / Info ----------------
@app.get("/")
def root():   return {"ok": True, "collection": COLLECTION, "model": MODEL_NAME}

@app.get("/healthz")
def healthz(): return {"status": "ok"}

@app.get("/device")
def device():
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    except Exception:
        dev, name = "unknown", "N/A"
    return {"embed_model": MODEL_NAME, "gen_model": GEN_MODEL, "device": dev, "gpu_name": name}

# ---------------- Search (Retrieval only) ----------------
=======
class ChatRequest(BaseModel):
    messages: List[dict]
    model: Optional[str] = GEN_MODEL
    filters: Optional[dict] = None

# ---------------- Search Endpoint ----------------
>>>>>>> Stashed changes
@app.get("/search", response_model=List[SearchResult])
def search(
    q: str = Query(..., description="query text; e5 prefers 'query: ...'"),
    top_k: int = 10, city: Optional[str] = None, state: Optional[str] = None,
    cuisines: Optional[List[str]] = Query(None), diets: Optional[List[str]] = Query(None),
    min_rating: Optional[float] = None, max_price: Optional[int] = None, open_now: bool = False,
):
    try:
        # rating override from text (even if UI sends a value)
        explicit = rating_from_text(q)
        if explicit is not None:
            min_rating = max(float(min_rating or 0), explicit)

        t0 = time.time()
        q_for_embed = f"{E5_PREFIX}{q}".strip()
        qvec = _model.encode([q_for_embed], normalize_embeddings=True)
        if isinstance(qvec, list): qvec = np.array(qvec)
        qvec = qvec[0].astype("float32").tolist()

        flt   = build_filter(city, state, cuisines, diets, min_rating, max_price)
        limit = top_k*3 if open_now else top_k
        points = qdrant_query(qvec, limit, flt)

        if open_now:
            kept=[]
            for p in points:
                if is_open_now(getattr(p,"payload",{}) or {}): kept.append(p)
                if len(kept)>=top_k: break
            points = kept

        # enforce filters client-side too
        points = apply_hard_filters(points, city=city, cuisines=cuisines, diets=diets,
                                    min_rating=min_rating, max_price=max_price)

        results=[]
        for p in points[:top_k]:
            pl = getattr(p,"payload",{}) or {}
            results.append(SearchResult(
                name=pl.get("name"), score=float(getattr(p,"score",0.0)),
                city=pl.get("city"), state=pl.get("state"),
                price=pl.get("price") if pl.get("price") is not None else None,
                rating=pl.get("rating") if pl.get("rating") is not None else None,
                cuisines=pl.get("cuisines"), website=pl.get("website"), id=pl.get("_id")
            ))
        print(f"[Search] latency={(time.time()-t0)*1000:.1f} ms, returned {len(results)}", flush=True)
        return results
    except Exception as e:
        print("[ERROR] /search failed:", e); traceback.print_exc()
        raise HTTPException(500, str(e))

<<<<<<< Updated upstream
# ---------------- Chat (RAG: Retrieval + Generation) ----------------
@app.get("/chat", response_model=ChatResponse)
def chat(
    q: str = Query(..., description="user request; e5 prefers 'query: ...'"),
=======
# ---------------- Chat Logic ----------------

def contextualize_and_classify(messages: List[dict], model_tag: str) -> Tuple[str, bool]:
    """
    Uses LLM to:
    1. Rewrite the last user message to be standalone, strictly preserving location AND topic context.
    2. Classify the intent.
    """
    if not messages:
        return "", False
    
    current_query = messages[-1]["content"]
    if len(messages) < 2:
        return current_query, is_rag_query(current_query)

    history_text = ""
    # Use last 4 messages to keep context fresh
    for m in messages[-5:-1]:
        role = "User" if m["role"] == "user" else "Assistant"
        history_text += f"{role}: {m['content']}\n"
    
    # UPDATED PROMPT
    prompt = f"""Analyze the conversation.

Conversation History:
{history_text}

Current User Message: "{current_query}"

Your tasks:
1. **Rewrite**: Create a standalone search query that merges the Current Message with the History.
   - **Location Retention**: If a CITY/STATE was established (e.g., NYC, Austin) and the user did not change it, YOU MUST INCLUDE IT.
   - **Topic Retention**: If the user is refining a search (e.g., "no pizza", "cheaper ones", "what about pasta"), KEEP the original cuisine/topic (e.g., "Italian").
   - **Example 1**: History="Italian in NYC", Current="No pizza" -> Rewritten="Italian restaurants in NYC that do not serve pizza"
   - **Example 2**: History="Burgers in Austin", Current="Cheap ones" -> Rewritten="Cheap burger restaurants in Austin"
   - **Example 3**: History="Sushi in Tokyo", Current="Actually I want Tacos" -> Rewritten="Tacos in Tokyo" (Topic changed, location kept)
   
2. **Classify**: INTENT is "SEARCH" if the user wants food info. INTENT is "CHAT" if it is just a greeting.

Return STRICT JSON:
{{
  "rewritten_query": "string",
  "intent": "SEARCH" or "CHAT"
}}
"""

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model_tag, 
                "prompt": prompt, 
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 150}
            },
            timeout=10
        )
        r.raise_for_status()
        resp_text = r.json().get("response", "").strip()
        
        m = re.search(r"\{.*\}", resp_text, re.S)
        if m:
            data = json.loads(m.group(0))
            rewritten = data.get("rewritten_query", current_query)
            intent = data.get("intent", "SEARCH").upper()
            print(f"[Contextualize] Original: '{current_query}' -> Rewritten: '{rewritten}' | Intent: {intent}", flush=True)
            return rewritten, (intent == "SEARCH")
        else:
            print(f"[Contextualize] JSON parse failed, raw: {resp_text}")
            return current_query, is_rag_query(current_query)

    except Exception as e:
        print(f"[Contextualize] Failed: {e}")
        return current_query, is_rag_query(current_query)

def process_chat_request(
    q: str,
    is_search_intent: bool,
    model_tag: str,
    top_k: int = 10,
    city: Optional[str] = None,
    state: Optional[str] = None,
    cuisines: Optional[List[str]] = None,
    diets: Optional[List[str]] = None,
    min_rating: Optional[float] = None,
    max_price: Optional[int] = None,
    open_now: bool = False,
):
    if not is_search_intent:
        print(f"[Chat] Intent is CHAT, streaming simple chat for: {q}", flush=True)
        return StreamingResponse(
            generate_ollama_chat_stream(model_tag, q), 
            media_type="text/plain"
        )

    print(f"[Chat] Intent is SEARCH, starting retrieval for: {q}", flush=True)

    user_provided_filters = any([city, state, cuisines, diets, min_rating, max_price])
    
    if not user_provided_filters:
        try:
            inferred = extract_filters_with_llm(q)
            city       = inferred.get("city") or city
            state      = inferred.get("state") or state
            cuisines   = inferred.get("cuisines") or cuisines
            diets      = inferred.get("diets") or diets
            min_rating = inferred.get("min_rating") or min_rating
            max_price  = inferred.get("max_price") or max_price
            print(f"[AutoFilter] inferred={inferred}", flush=True)
        except Exception as e:
            print(f"[AutoFilter] LLM extraction failed: {e}, continuing without filters", flush=True)
    
    explicit = rating_from_text(q)
    if explicit is not None:
        min_rating = max(float(min_rating or 0), explicit)

    # 1) retrieve
    q_for_embed = f"{E5_PREFIX}{q}".strip()
    qvec = _model.encode([q_for_embed], normalize_embeddings=True)
    if isinstance(qvec, list): qvec = np.array(qvec)
    qvec = qvec[0].astype("float32").tolist()
    flt  = build_filter(city, state, cuisines, diets, min_rating, max_price)
    
    limit = top_k * 4
    points = qdrant_query(qvec, limit, flt)

    points = apply_hard_filters(points, city=city, cuisines=cuisines, diets=diets,
                                min_rating=min_rating, max_price=max_price)

    if open_now:
        kept=[]
        for p in points:
            if is_open_now(getattr(p,"payload",{}) or {}): kept.append(p)
            if len(kept)>=top_k: break
        points = kept

    # 2) build context + prompt
    # Increase context size when filtering by cuisine to show more options
    max_context_items = 20 if cuisines else 15
    ctx_text, rows = build_context_block(points, max_items=max_context_items)
    prompt = rag_prompt(q, ctx_text)

    # 3) generate (Ollama) - use lower temperature to reduce hallucinations
    return StreamingResponse(
        generate_ollama_rag_stream(model_tag, prompt, temperature=0.1), 
        media_type="text/plain"
    )

@app.post("/chat")
def chat_post(req: ChatRequest):
    try:
        q, is_search = contextualize_and_classify(req.messages, req.model)
        filters = req.filters or {}
        return process_chat_request(
            q=q,
            is_search_intent=is_search,
            model_tag=req.model,
            top_k=int(filters.get('top_k', 10)),
            city=filters.get('city'),
            state=filters.get('state'),
            cuisines=filters.get('cuisines'),
            diets=filters.get('diets'),
            min_rating=filters.get('min_rating'),
            max_price=filters.get('max_price'),
            open_now=filters.get('open_now', False)
        )
    except Exception as e:
        print("[ERROR] /chat POST failed:", e); traceback.print_exc()
        raise HTTPException(500, str(e))

@app.get("/chat")
def chat_get(
    q: str = Query(..., description="user request"),
    model_tag: str = Query(GEN_MODEL, description="Ollama model tag"),
>>>>>>> Stashed changes
    top_k: int = 10, city: Optional[str] = None, state: Optional[str] = None,
    cuisines: Optional[List[str]] = Query(None), diets: Optional[List[str]] = Query(None),
    min_rating: Optional[float] = None, max_price: Optional[int] = None, open_now: bool = False,
):
    try:
<<<<<<< Updated upstream
        # Check if this is actually a restaurant search query
        if not is_restaurant_search_query(q):
            # Handle greetings and non-restaurant queries
            greeting_responses = {
                "hi": "Hello! I'm here to help you find great restaurants. What are you looking for?",
                "hello": "Hello! I can help you find restaurants. What cuisine or location are you interested in?",
                "hey": "Hey there! I'm your restaurant assistant. How can I help you today?",
                "thanks": "You're welcome! Feel free to ask if you need more restaurant recommendations.",
                "thank you": "You're welcome! Happy to help with restaurant suggestions anytime.",
                "bye": "Goodbye! Enjoy your dining experience!",
                "goodbye": "Goodbye! Have a great day!"
            }
            
            q_lower = q.lower().strip()
            if q_lower in greeting_responses:
                return ChatResponse(
                    answer=greeting_responses[q_lower],
                    used_sources=[]
                )
            else:
                # Generic friendly response for other short queries
                return ChatResponse(
                    answer="Hello! I'm here to help you find restaurants. You can ask me about restaurants by city, cuisine, dietary requirements, price range, or ratings. What would you like to search for?",
                    used_sources=[]
                )

        # Store original user-provided filters to check if we need LLM extraction
        user_provided_filters = any([city, state, cuisines, diets, min_rating, max_price])
        
        # If no filters provided by user, try to infer with LLM
        if not user_provided_filters:
            try:
                inferred = extract_filters_with_llm(q)
                city       = inferred.get("city") or city
                state      = inferred.get("state") or state
                cuisines   = inferred.get("cuisines") or cuisines
                diets      = inferred.get("diets") or diets
                min_rating = inferred.get("min_rating") or min_rating
                max_price  = inferred.get("max_price") or max_price
                print(f"[AutoFilter] inferred={inferred}", flush=True)
            except Exception as e:
                print(f"[AutoFilter] LLM extraction failed: {e}, continuing without filters", flush=True)
                # Continue without extracted filters - will rely on semantic search only
        
        # rating override from text (even if UI sends a value or LLM extracted it)
        explicit = rating_from_text(q)
        if explicit is not None:
            min_rating = max(float(min_rating or 0), explicit)

        # 1) retrieve (with e5 prefix)
        q_for_embed = f"{E5_PREFIX}{q}".strip()
        qvec = _model.encode([q_for_embed], normalize_embeddings=True)
        if isinstance(qvec, list): qvec = np.array(qvec)
        qvec = qvec[0].astype("float32").tolist()
        flt  = build_filter(city, state, cuisines, diets, min_rating, max_price)
        limit = top_k*2 if open_now else top_k*2
        points = qdrant_query(qvec, limit, flt)

        # guarantee hard filters client-side
        points = apply_hard_filters(points, city=city, cuisines=cuisines, diets=diets,
                                    min_rating=min_rating, max_price=max_price)

        if open_now:
            kept=[]
            for p in points:
                if is_open_now(getattr(p,"payload",{}) or {}): kept.append(p)
                if len(kept)>=top_k: break
            points = kept

        # 2) build context + prompt
        ctx_text, rows = build_context_block(points, max_items=8)
        prompt = rag_prompt(q, ctx_text)

        # 3) generate (Ollama)
        answer = generate_with_ollama(GEN_MODEL, prompt)
        return ChatResponse(answer=answer, used_sources=rows[:5] if rows else [])

    except Exception as e:
        print("[ERROR] /chat failed:", e); traceback.print_exc()
        raise HTTPException(500, str(e))
=======
        is_search = is_rag_query(q)
        return process_chat_request(
            q=q,
            is_search_intent=is_search,
            model_tag=model_tag,
            top_k=top_k,
            city=city,
            state=state,
            cuisines=cuisines,
            diets=diets,
            min_rating=min_rating,
            max_price=max_price,
            open_now=open_now
        )
    except Exception as e:
        print("[ERROR] /chat GET failed:", e); traceback.print_exc()
        raise HTTPException(500, str(e))

@app.get("/")
def root():   return {"ok": True, "collection": COLLECTION, "model": MODEL_NAME}

@app.get("/healthz")
def healthz(): return {"status": "ok"}

@app.get("/device")
def device():
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    except Exception:
        dev, name = "unknown", "N/A"
    return {"embed_model": MODEL_NAME, "gen_model": GEN_MODEL, "device": dev, "gpu_name": name}
>>>>>>> Stashed changes
