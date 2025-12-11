#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

STATE_ZIP_RE = re.compile(r"\b([A-Z]{2})\s+(\d{5}(?:-\d{4})?)\b")
TIME_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$")

def normalize_list_field(value: Any) -> List[str]:
    """
    JSON formatında liste alanları zaten listeler olarak gelir.
    Sadece None kontrolü ve string'e çevirme yapılır.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    # Eğer liste değilse (nadir durum), string'e çevirip tek elemanlı liste yap
    return [str(value).strip()] if str(value).strip() else []

def price_to_level(price: Any) -> Optional[int]:
    """
    '$', '$$', '$$$', '$$$$' → 1..4
    """
    if price is None:
        return None
    s = str(price).strip()
    if not s:
        return None
    # "$$ - $$$" gibi aralıklar görülebilir → ortalama al
    if "-" in s:
        parts = [p.strip() for p in s.split("-")]
        vals = [price_to_level(p) for p in parts]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals)) if vals else None
    if set(s) == {"$"}:
        return len(s)
    # "Moderate" gibi kelimeler varsa basit eşleme
    s_low = s.lower()
    if "cheap" in s_low or "inexpensive" in s_low or "low" in s_low:
        return 1
    if "moderate" in s_low or "mid" in s_low:
        return 2
    if "expens" in s_low:
        return 3
    if "very expens" in s_low or "ultra" in s_low:
        return 4
    return None

def parse_address(addr_data: Any) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    JSON formatında hem 'address' string'i hem de 'detailed_address' objesi olabilir.
    Önce detailed_address'i kontrol et, yoksa address string'ini parse et.
    """
    # Eğer detailed_address varsa, onu kullan
    if isinstance(addr_data, dict) and "detailed_address" in addr_data:
        det = addr_data["detailed_address"]
        street = det.get("street")
        city = det.get("city")
        state = det.get("state")
        postal = det.get("postal_code")
        return street, city, state, postal
    
    # Eğer doğrudan detailed_address objesi gelirse
    if isinstance(addr_data, dict):
        street = addr_data.get("street")
        city = addr_data.get("city")
        state = addr_data.get("state")
        postal = addr_data.get("postal_code")
        return street, city, state, postal
    
    # String address parse et
    if addr_data is None:
        return None, None, None, None
    s = str(addr_data).strip()
    if not s:
        return None, None, None, None

    # State + ZIP yakala
    m = STATE_ZIP_RE.search(s)
    state, postal = (m.group(1), m.group(2)) if m else (None, None)

    parts = [p.strip() for p in s.split(",")]
    street, city = None, None
    if len(parts) >= 3:
        street = ", ".join(parts[:-2])
        city = parts[-2]
    elif len(parts) == 2:
        street = parts[0]
        city = parts[1]
    else:
        street = s

    return street or None, city or None, state, postal

def format_time_string(time_str: str) -> Optional[str]:
    """
    "16:30:00" → "16:30" formatına çevirir (saniye kısmını kaldırır).
    """
    if not isinstance(time_str, str):
        return None
    m = TIME_RE.match(time_str.strip())
    if not m:
        return None
    h, mi, _ = m.groups()
    return f"{int(h):02d}:{int(mi):02d}"

def normalize_hours(open_hours_data: Any) -> Dict[str, List[Dict[str, str]]]:
    """
    JSON formatında open_hours zaten dict olarak gelir.
    {"mon":[{"open":"16:30:00","close":"22:00:00"}], ...}
    → {"mon":[{"open":"16:30","close":"22:00"}], ...}  (string formatında, saniye kısmı kaldırılmış)
    """
    result: Dict[str, List[Dict[str, str]]] = {}
    if open_hours_data is None:
        return result
    
    # Zaten dict ise direkt kullan
    if not isinstance(open_hours_data, dict):
        return result

    for dow, ranges in open_hours_data.items():
        norm: List[Dict[str, str]] = []
        if isinstance(ranges, list):
            for r in ranges:
                if not isinstance(r, dict):
                    continue
                open_str = format_time_string(r.get("open"))
                close_str = format_time_string(r.get("close"))
                if open_str is not None and close_str is not None:
                    norm.append({"open": open_str, "close": close_str})
        result[dow.lower()] = norm
    return result

def build_search_text(record: Dict[str, Any], max_keywords: int = 50) -> str:
    """
    JSON record'dan arama metni oluşturur.
    """
    kw = normalize_list_field(record.get("review_keywords"))[:max_keywords]
    tags = normalize_list_field(record.get("top_tags"))
    name = str(record.get("name") or "").strip()
    desc = str(record.get("description") or "").strip()
    cuisines = ", ".join(normalize_list_field(record.get("cuisines")))
    diets = ", ".join(normalize_list_field(record.get("diets")))
    parts = [
        name,
        desc,
        f"Top tags: {', '.join(tags)}" if tags else "",
        f"Popular: {', '.join(kw)}" if kw else "",
        f"Cuisines: {cuisines}" if cuisines else "",
        f"Diets: {diets}" if diets else "",
    ]
    return ". ".join([p for p in parts if p]).strip()

def transform_records(records: List[Dict[str, Any]], max_keywords: int) -> List[Dict[str, Any]]:
    """
    JSON record'ları normalize eder ve transform eder.
    """
    transformed = []
    seen_ids = set()
    
    for record in records:
        # Tekil id kontrolü
        record_id = record.get("id")
        if record_id in seen_ids:
            continue
        seen_ids.add(record_id)
        
        # Address parçalama
        # Önce detailed_address'i kontrol et, sonra address string'ini
        addr_data = record.get("detailed_address") or record.get("address")
        street, city, state, postal = parse_address(addr_data)
        
        # Fiyat seviyesi
        price_level = price_to_level(record.get("price_range"))
        
        # Liste alanlarını normalize et
        cuisines = normalize_list_field(record.get("cuisines"))
        diets = normalize_list_field(record.get("diets"))
        meal_types = normalize_list_field(record.get("meal_types"))
        dining_options = normalize_list_field(record.get("dining_options"))
        owner_types = normalize_list_field(record.get("owner_types"))
        top_tags = normalize_list_field(record.get("top_tags"))
        review_keywords = normalize_list_field(record.get("review_keywords"))[:max_keywords]
        
        # open_hours normalize
        open_norm = normalize_hours(record.get("open_hours"))
        
        # Arama metni
        search_text = build_search_text(record, max_keywords=max_keywords)
        
        # Transform edilmiş record oluştur
        transformed_record = {
            **record,  # Orijinal tüm alanları koru
            "street": street,
            "city": city,
            "state": state,
            "postal": postal,
            "price_level": price_level,
            "cuisines": cuisines,
            "diets": diets,
            "meal_types": meal_types,
            "dining_options": dining_options,
            "owner_types": owner_types,
            "top_tags": top_tags,
            "review_keywords_list": review_keywords,
            "open_norm": open_norm,
            "search_text": search_text,
        }
        
        transformed.append(transformed_record)
    
    return transformed

def to_index_records(transformed_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform edilmiş record'lardan RAG indeksleme için JSONL formatına çevirir.
    """
    records = []
    for rec in transformed_records:
        index_rec = {
            "id": rec.get("id"),
            "text": rec.get("search_text", ""),
            "metadata": {
                "name": rec.get("name"),
                "rating": float(rec["rating"]) if rec.get("rating") is not None else None,
                "price": int(rec["price_level"]) if rec.get("price_level") is not None else None,
                "city": rec.get("city"),
                "state": rec.get("state"),
                "postal": rec.get("postal"),
                "street": rec.get("street"),
                "cuisines": rec.get("cuisines", []),
                "diets": rec.get("diets", []),
                "meal_types": rec.get("meal_types", []),
                "dining_options": rec.get("dining_options", []),
                "open": rec.get("open_norm", {}),
                "website": rec.get("website"),
                "menu_link": rec.get("menu_link"),
                "address_raw": rec.get("address"),
                "top_tags": rec.get("top_tags", []),
            },
        }
        records.append(index_rec)
    return records

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    JSON veya JSONL dosyasını yükler.
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        # Dosya uzantısına göre format belirle
        if file_path.lower().endswith(".jsonl"):
            # JSONL: Her satır bir JSON objesi
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        else:
            # JSON: Tek bir JSON dosyası (array veya object)
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # Eğer tek bir obje ise, listeye çevir
                records = [data]
            else:
                raise ValueError(f"JSON dosyası beklenmeyen format: {type(data)}")
    
    return records

def main():
    ap = argparse.ArgumentParser(description="Prepare TripAdvisor JSON dataset for RAG indexing.")
    ap.add_argument("--in", dest="in_file", required=True, help="Input JSON or JSONL file.")
    ap.add_argument("--out_jsonl", default="index.jsonl", help="Output JSONL for vector indexing.")
    ap.add_argument("--out_parquet", default="normalized.parquet", help="Output Parquet for analysis.")
    ap.add_argument("--limit_keywords", type=int, default=50, help="Max review keywords to keep.")
    args = ap.parse_args()

    # JSON dosyasını yükle
    print(f"Loading JSON file: {args.in_file}")
    records = load_json_file(args.in_file)
    print(f"Loaded {len(records)} records")
    
    # Transform et
    print("Transforming records...")
    transformed_records = transform_records(records, max_keywords=args.limit_keywords)
    print(f"Transformed {len(transformed_records)} records")
    
    # Parquet için DataFrame oluştur
    print("Creating Parquet file...")
    df = pd.DataFrame(transformed_records)
    df.to_parquet(args.out_parquet, index=False)
    print(f"Saved Parquet: {args.out_parquet}")

    # JSONL (indeks belgeleri)
    print("Creating JSONL file...")
    index_records = to_index_records(transformed_records)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for rec in index_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved JSONL: {args.out_jsonl}")

    # Konsol özeti
    print("\n==== SAMPLE (first 1) ====")
    if index_records:
        sample_str = json.dumps(index_records[0], ensure_ascii=False, indent=2)
        print(sample_str[:1200])
        if len(sample_str) > 1200:
            print("... (truncated)")
    print(f"\nTotal rows: {len(transformed_records)}")
    print(f"Output files: {args.out_jsonl}, {args.out_parquet}")

if __name__ == "__main__":
    main()
