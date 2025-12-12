#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, sys, os
from typing import Any, Dict, Iterable, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny, MatchValue, Range

# ---------- IO ----------
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# ---------- Device & Embeddings ----------
def pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # macOS Metal
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def get_model(name: str):
    device = pick_device()
    print(f"[Embed] loading '{name}' on device: {device}")
    return SentenceTransformer(name, device=device)

# ---------- Qdrant helpers ----------
def ensure_collection(cli: QdrantClient, name: str, dim: int):
    try:
        cli.get_collection(name)
        print(f"[Qdrant] Using existing collection: {name}")
    except Exception:
        cli.recreate_collection(name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        print(f"[Qdrant] Created collection: {name}")

def build_filter(city: Optional[str], state: Optional[str], top_tags: Optional[List[str]], meal_types: Optional[List[str]], dining_options: Optional[List[str]], diets: Optional[List[str]], max_price: Optional[int], cuisines_any: Optional[List[str]], min_rating: Optional[float]) -> Optional[Filter]:
    conds = []
    if city: conds.append(FieldCondition(key="city", match=MatchValue(value=city)))
    if max_price is not None: conds.append(FieldCondition(key="price", range=Range(lte=max_price)))
    if min_rating is not None: conds.append(FieldCondition(key="rating", range=Range(gte=min_rating)))
    if cuisines_any: conds.append(FieldCondition(key="cuisines", match=MatchAny(any=cuisines_any)))
    if state: conds.append(FieldCondition(key="state", match=MatchValue(value=state)))
    if top_tags: conds.append(FieldCondition(key="top_tags", match=MatchAny(any=top_tags)))
    if meal_types: conds.append(FieldCondition(key="meal_types", match=MatchAny(any=meal_types)))
    if dining_options: conds.append(FieldCondition(key="dining_options", match=MatchAny(any=dining_options)))
    if diets: conds.append(FieldCondition(key="diets", match=MatchAny(any=diets)))
    return Filter(must=conds) if conds else None

def is_open_now(payload: Dict[str, Any]) -> bool:
    from datetime import datetime
    dow_map = ["mon","tue","wed","thu","fri","sat","sun"]
    now = datetime.now()
    dow = dow_map[now.weekday()]
    minutes = now.hour*60 + now.minute
    ranges = (payload.get("open") or {}).get(dow, [])
    for start, end in ranges:
        if start <= minutes <= end: return True
        if end > 1440 and (start <= minutes+1440 <= end): return True
    return False

# ---------- Commands ----------
def cmd_index(args):
    cli = QdrantClient(host=args.host, port=args.port)
    docs = list(read_jsonl(args.input))
    if not docs:
        print("No docs found.", file=sys.stderr); sys.exit(1)

    texts = [d.get("text","") for d in docs]
    payloads = [{**(d.get("metadata") or {}), "_id": d.get("id")} for d in docs]

    model = get_model(args.model)
    vecs = model.encode(texts, batch_size=args.batch, show_progress_bar=True, normalize_embeddings=True)

    ensure_collection(cli, args.collection, vecs.shape[1])

    step = args.upload_step
    total = len(docs)
    for i in range(0, total, step):
        chunk = vecs[i:i+step].astype("float32")
        points = [PointStruct(id=i+j, vector=v.tolist(), payload=payloads[i+j]) for j, v in enumerate(chunk)]
        cli.upsert(collection_name=args.collection, points=points, wait=True)
        print(f"Upserted {min(i+step, total)}/{total}")
    print("[Index] Done.")

def cmd_search(args):
    cli = QdrantClient(host=args.host, port=args.port)
    model = get_model(args.model)
    qvec = model.encode([args.query], normalize_embeddings=True)[0].astype("float32").tolist()
    flt = build_filter(args.city, args.state, args.top_tags, args.meal_types, args.dining_options, args.diets, args.max_price, args.cuisine, args.min_rating)
    limit = args.top_k*3 if args.open_now else args.top_k
    res = cli.query_points(
        collection_name=args.collection,
        query=qvec,
        query_filter=flt,
        limit=limit,
        with_payload=True
    )
    points = res.points
    if args.open_now:
        kept = []
        for p in points:
            if is_open_now(p.payload or {}):
                kept.append(p)
            if len(kept) >= args.top_k: break
        points = kept
    for i, p in enumerate(points, 1):
        pl = p.payload or {}
        print(f"{i:02d}. {pl.get('name')} | score={p.score:.4f}")
        print(f"    city={pl.get('city')} price={pl.get('price')} rating={pl.get('rating')} cuisines={pl.get('cuisines')}")
        print(f"    website={pl.get('website')}  id={pl.get('_id')}\n")

# ---------- CLI ----------
def cli():
    ap = argparse.ArgumentParser(description="Embed & search with Qdrant")
    ap.add_argument("--host", default=os.getenv("Q_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("Q_PORT", "6333")))
    ap.add_argument("--collection", default=os.getenv("Q_COLLECTION", "restaurants"))
    ap.add_argument("--model", default=os.getenv("EMBED_MODEL", "intfloat/e5-small-v2"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("index"); a.add_argument("--input", required=True); a.add_argument("--batch", type=int, default=64); a.add_argument("--upload_step", type=int, default=1000)
    s = sub.add_parser("search"); s.add_argument("--query", required=True); s.add_argument("--top_k", type=int, default=10); s.add_argument("--city"); s.add_argument("--state"); s.add_argument("--max_price", type=int); s.add_argument("--min_rating", type=float); s.add_argument("--cuisine", action="append"); s.add_argument("--top_tags", action="append"); s.add_argument("--meal_types", action="append"); s.add_argument("--dining_options", action="append"); s.add_argument("--diets", action="append"); s.add_argument("--open_now", action="store_true")
    return ap

if __name__ == "__main__":
    args = cli().parse_args()
    if args.cmd == "index": cmd_index(args)
    else: cmd_search(args)
