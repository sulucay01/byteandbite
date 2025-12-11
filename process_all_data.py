#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline script to process all raw JSON data files and merge them into unified outputs.
This script automatically finds all state JSON files (e.g., cleveland.json, new_york.json, texas.json)
and processes them into merged index.jsonl and normalized.parquet files.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set
import pandas as pd

# Import functions from prepare_rag_dataset.py
from prepare_rag_dataset import (
    load_json_file,
    transform_records,
    to_index_records,
)


def find_raw_json_files(data_dir: str, exclude_patterns: List[str] = None) -> List[Path]:
    """
    Find all raw JSON files in the data directory.
    Excludes processed output files like index.jsonl and normalized.parquet.
    """
    if exclude_patterns is None:
        exclude_patterns = ["index.jsonl", "normalized.parquet"]
    
    data_path = Path(data_dir)
    json_files = []
    
    for file_path in data_path.glob("*.json"):
        # Skip if it's an output file
        if any(pattern in file_path.name for pattern in exclude_patterns):
            continue
        json_files.append(file_path)
    
    return sorted(json_files)


def extract_state_name(file_path: Path) -> str:
    """
    Extract state name from filename (e.g., 'cleveland.json' -> 'Cleveland').
    """
    name = file_path.stem  # Get filename without extension
    # Convert snake_case or kebab-case to Title Case
    name = name.replace("_", " ").replace("-", " ")
    return name.title()


def process_single_file(
    file_path: Path,
    max_keywords: int = 50,
    verbose: bool = True
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process a single JSON file and return transformed records and index records.
    
    Returns:
        (transformed_records, index_records)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {file_path.name}")
        print(f"{'='*60}")
    
    # Load raw records
    records = load_json_file(str(file_path))
    if verbose:
        print(f"Loaded {len(records)} raw records")
    
    # Transform records
    transformed_records = transform_records(records, max_keywords=max_keywords)
    if verbose:
        print(f"Transformed {len(transformed_records)} records")
    
    # Add source state information to transformed records
    state_name = extract_state_name(file_path)
    for rec in transformed_records:
        # Add source_file field to track origin
        rec["source_file"] = file_path.name
        rec["source_state"] = state_name
    
    # Convert to index records
    index_records = to_index_records(transformed_records)
    
    # Add source information to index metadata
    for idx_rec, trans_rec in zip(index_records, transformed_records):
        if "metadata" in idx_rec:
            idx_rec["metadata"]["source_file"] = trans_rec.get("source_file")
            idx_rec["metadata"]["source_state"] = trans_rec.get("source_state")
    
    if verbose:
        print(f"Created {len(index_records)} index records")
    
    return transformed_records, index_records


def merge_all_data(
    json_files: List[Path],
    max_keywords: int = 50,
    deduplicate: bool = True,
    verbose: bool = True
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process all JSON files and merge the results.
    
    Args:
        json_files: List of JSON file paths to process
        max_keywords: Maximum number of keywords to keep per record
        deduplicate: Whether to remove duplicate records based on ID
        verbose: Whether to print progress information
    
    Returns:
        (all_transformed_records, all_index_records)
    """
    all_transformed_records = []
    all_index_records = []
    seen_ids: Set[Any] = set()
    
    total_files = len(json_files)
    
    if verbose:
        print(f"\n{'#'*60}")
        print(f"Processing {total_files} JSON file(s)")
        print(f"{'#'*60}\n")
    
    for idx, file_path in enumerate(json_files, 1):
        if verbose:
            print(f"\n[{idx}/{total_files}] Processing: {file_path.name}")
        
        try:
            transformed_records, index_records = process_single_file(
                file_path,
                max_keywords=max_keywords,
                verbose=verbose
            )
            
            # Deduplicate if requested
            if deduplicate:
                for trans_rec, idx_rec in zip(transformed_records, index_records):
                    record_id = trans_rec.get("id")
                    if record_id not in seen_ids:
                        seen_ids.add(record_id)
                        all_transformed_records.append(trans_rec)
                        all_index_records.append(idx_rec)
                    elif verbose:
                        print(f"  Skipping duplicate ID: {record_id} (from {trans_rec.get('source_file')})")
            else:
                all_transformed_records.extend(transformed_records)
                all_index_records.extend(index_records)
        
        except Exception as e:
            print(f"ERROR processing {file_path.name}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    if verbose:
        print(f"\n{'#'*60}")
        print(f"Processing complete!")
        print(f"Total transformed records: {len(all_transformed_records)}")
        print(f"Total index records: {len(all_index_records)}")
        if deduplicate:
            print(f"Unique IDs: {len(seen_ids)}")
        print(f"{'#'*60}\n")
    
    return all_transformed_records, all_index_records


def save_outputs(
    transformed_records: List[Dict[str, Any]],
    index_records: List[Dict[str, Any]],
    output_dir: str,
    out_jsonl: str = "index.jsonl",
    out_parquet: str = "normalized.parquet",
    verbose: bool = True
):
    """
    Save merged outputs to JSONL and Parquet files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save Parquet
    parquet_path = output_path / out_parquet
    if verbose:
        print(f"Creating Parquet file: {parquet_path}")
    df = pd.DataFrame(transformed_records)
    df.to_parquet(parquet_path, index=False)
    if verbose:
        print(f"Saved Parquet: {parquet_path} ({len(df)} rows)")
    
    # Save JSONL
    jsonl_path = output_path / out_jsonl
    if verbose:
        print(f"Creating JSONL file: {jsonl_path}")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in index_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if verbose:
        print(f"Saved JSONL: {jsonl_path} ({len(index_records)} records)")
    
    return parquet_path, jsonl_path


def print_summary(
    transformed_records: List[Dict[str, Any]],
    index_records: List[Dict[str, Any]],
    verbose: bool = True
):
    """
    Print summary statistics about the processed data.
    """
    if not verbose or not transformed_records:
        return
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Count by source state
    state_counts = {}
    for rec in transformed_records:
        state = rec.get("source_state", "Unknown")
        state_counts[state] = state_counts.get(state, 0) + 1
    
    print(f"\nRecords by source state:")
    for state, count in sorted(state_counts.items()):
        print(f"  {state}: {count:,}")
    
    # Count by actual state (from address)
    actual_state_counts = {}
    for rec in transformed_records:
        state = rec.get("state")
        if state:
            actual_state_counts[state] = actual_state_counts.get(state, 0) + 1
    
    if actual_state_counts:
        print(f"\nRecords by address state:")
        for state, count in sorted(actual_state_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {state}: {count:,}")
        if len(actual_state_counts) > 10:
            print(f"  ... and {len(actual_state_counts) - 10} more states")
    
    # Sample record
    print(f"\n{'='*60}")
    print("SAMPLE RECORD (first index record)")
    print("="*60)
    if index_records:
        sample_str = json.dumps(index_records[0], ensure_ascii=False, indent=2)
        print(sample_str[:1500])
        if len(sample_str) > 1500:
            print("\n... (truncated)")
    
    print(f"\n{'='*60}")
    print(f"Total records: {len(transformed_records):,}")
    print(f"Total index records: {len(index_records):,}")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Process all raw JSON data files and merge into unified outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all JSON files in ./raw_data, output to current directory
  python process_all_data.py --input_dir ./raw_data --output_dir .

  # Process all JSON files in ./raw_data, output to ./data
  python process_all_data.py --input_dir ./raw_data --output_dir ./data

  # Process specific files only
  python process_all_data.py --input_dir ./raw_data --files cleveland.json new_york.json

  # Disable deduplication
  python process_all_data.py --input_dir ./raw_data --no-deduplicate

  # Backward compatible: use --data_dir (sets both input and output)
  python process_all_data.py --data_dir ./data
        """
    )
    
    ap.add_argument(
        "--data_dir",
        default=None,
        help="Directory containing raw JSON files (deprecated: use --input_dir instead)"
    )
    ap.add_argument(
        "--input_dir",
        default=None,
        help="Directory containing raw JSON files (default: current directory)"
    )
    ap.add_argument(
        "--output_dir",
        default=None,
        help="Directory for output files (default: same as input_dir)"
    )
    ap.add_argument(
        "--files",
        nargs="*",
        help="Specific JSON files to process (if not provided, all .json files in data_dir will be processed)"
    )
    ap.add_argument(
        "--out_jsonl",
        default="index.jsonl",
        help="Output JSONL filename (default: index.jsonl)"
    )
    ap.add_argument(
        "--out_parquet",
        default="normalized.parquet",
        help="Output Parquet filename (default: normalized.parquet)"
    )
    ap.add_argument(
        "--limit_keywords",
        type=int,
        default=50,
        help="Max review keywords to keep per record (default: 50)"
    )
    ap.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable deduplication (keep all records even if IDs are duplicated)"
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = ap.parse_args()
    
    # Handle backward compatibility: if --data_dir is provided, use it for both input and output
    if args.data_dir is not None:
        input_dir = args.data_dir
        output_dir = args.output_dir if args.output_dir is not None else args.data_dir
    else:
        input_dir = args.input_dir if args.input_dir is not None else "."
        output_dir = args.output_dir if args.output_dir is not None else input_dir
    
    # Find JSON files to process
    if args.files:
        # Process specific files
        json_files = [Path(input_dir) / f for f in args.files]
        # Verify files exist
        json_files = [f for f in json_files if f.exists()]
        if not json_files:
            print(f"ERROR: No valid JSON files found from provided list: {args.files}")
            return
    else:
        # Find all JSON files automatically
        json_files = find_raw_json_files(input_dir)
        if not json_files:
            print(f"ERROR: No JSON files found in {input_dir}")
            print("Please ensure raw JSON files (e.g., cleveland.json, new_york.json) are in the input directory.")
            return
    
    if not args.quiet:
        print(f"Found {len(json_files)} JSON file(s) to process:")
        for f in json_files:
            print(f"  - {f.name}")
    
    # Process and merge all data
    transformed_records, index_records = merge_all_data(
        json_files,
        max_keywords=args.limit_keywords,
        deduplicate=not args.no_deduplicate,
        verbose=not args.quiet
    )
    
    if not transformed_records:
        print("ERROR: No records were processed. Exiting.")
        return
    
    # Save outputs
    parquet_path, jsonl_path = save_outputs(
        transformed_records,
        index_records,
        output_dir=output_dir,
        out_jsonl=args.out_jsonl,
        out_parquet=args.out_parquet,
        verbose=not args.quiet
    )
    
    # Print summary
    print_summary(transformed_records, index_records, verbose=not args.quiet)
    
    if not args.quiet:
        print(f"\n✓ Processing complete!")
        print(f"  Output files:")
        print(f"    - {jsonl_path}")
        print(f"    - {parquet_path}")


if __name__ == "__main__":
    main()

