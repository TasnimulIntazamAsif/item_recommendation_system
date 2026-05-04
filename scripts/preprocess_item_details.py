import argparse
import csv
import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple


DEMO_NAME_PATTERNS = [
    r"\bdemo\b",
    r"\btest\b",
    r"\bsample\b",
    r"\bdummy\b",
    r"\btrial\b",
    r"\btemp\b",
    r"\btemporary\b",
    r"\bplaceholder\b",
    r"\bpractice\b",
]

NOISE_TOKEN_PATTERNS = [
    r"^buy\d+$",
    r"^get\d+$",
    r"^g\d+$",
    r"^b\d+$",
    r"^b\d+g\d+[a-z]*$",
    r"^s\d{4,}$",
    r"^[a-z]{1,3}\d{4,}$",
    r"^win\d+$",
    r"^\d+x$",
]


def normalize_space(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def normalize_item_name(name: str) -> str:
    s = normalize_space(name).strip("\"'")
    s = normalize_space(s)
    s = re.sub(r"(?i)(\d)\s*(kg|gm|g|ml|l)\b", r"\1 \2", s)
    s = re.sub(r"(?i)\b(\d+)\s*(ltr|liter|litre)\b", r"\1 ltr", s)
    s = re.sub(r"(?i)\b(\d+)\s*(oz)\b", r"\1 oz", s)
    s = re.sub(r"(?i)\b(\d+)\s*(pcs|pc)\b", r"\1 pcs", s)
    s = re.sub(r"\s*[-_/]+\s*", " ", s)
    s = normalize_space(s)
    return s


def denoise_item_name(name: str) -> str:
    """
    Remove noisy promotional/coded fragments while keeping the core item phrase.
    """
    s = normalize_item_name(name)
    tokens = [t for t in s.split(" ") if t]
    kept: List[str] = []
    for token in tokens:
        t = token.casefold()
        if any(re.fullmatch(pat, t) for pat in NOISE_TOKEN_PATTERNS):
            continue
        if t in {"b1", "b2", "b3", "b4", "b5", "g1", "g2", "g3", "g4"}:
            continue
        # Drop very long mixed alpha-numeric identifiers (likely system/product noise).
        if len(t) >= 7 and re.search(r"[a-z]", t) and re.search(r"\d", t):
            continue
        kept.append(token)

    cleaned = " ".join(kept)
    cleaned = re.sub(r"(?i)\b(?:buy|get|win)\s*\d+\b", " ", cleaned)
    cleaned = re.sub(r"(?i)\bb\d+\s*g\d+\b", " ", cleaned)
    cleaned = re.sub(r"\b(?:buy|get)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = normalize_space(cleaned)
    return cleaned


def title_case_phrase(text: str) -> str:
    words = []
    for w in normalize_space(text).split(" "):
        if not w:
            continue
        if len(w) <= 2:
            words.append(w.upper())
        else:
            words.append(w.capitalize())
    return " ".join(words)


def item_merge_key(name: str) -> str:
    """
    Build a size-insensitive key:
    - remove weight/volume patterns (500gm, 1 kg)
    - remove standalone numeric tokens
    - normalize punctuation and spacing
    """
    s = normalize_item_name(name).casefold()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\s*(?:kg|gm|g|ml|l)\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\b\d+\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    if low == high:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def iqr_bounds(values: List[float], multiplier: float = 1.5) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def is_name_outlier(name: str, length_bounds: Tuple[float, float], token_bounds: Tuple[float, float]) -> bool:
    if not name:
        return True
    lower = name.casefold()
    if lower in {"na", "n/a", "null", "none", "unknown", "order cake"}:
        return True
    for pat in DEMO_NAME_PATTERNS:
        if re.search(pat, lower):
            return True
    if "http://" in lower or "https://" in lower or "www." in lower:
        return True

    letters = sum(1 for ch in name if ch.isalpha())
    digits = sum(1 for ch in name if ch.isdigit())
    if letters == 0:
        return True
    if letters <= 1 and digits >= 3:
        return True

    length = len(name)
    tokens = len([t for t in name.split(" ") if t])
    min_len, max_len = length_bounds
    min_tok, max_tok = token_bounds
    if length < 3 or length < min_len or length > max_len:
        return True
    if tokens < 1 or tokens < min_tok or tokens > max_tok:
        return True
    return False


def read_rows(input_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for raw in reader:
            if not raw:
                continue
            # Support both 3-column and 4-column files.
            if len(raw) >= 4:
                item_id, item_name, product_code = raw[0], raw[1], raw[2]
            elif len(raw) == 3:
                item_id, item_name, product_code = raw
            else:
                continue
            rows.append(
                {
                    "item_id": normalize_space(item_id),
                    "item_name_raw": denoise_item_name(item_name),
                    "product_code_raw": normalize_space(product_code),
                }
            )
    return rows


def preprocess(
    input_path: str,
    output_path: str,
    report_path: str,
    start_product_code: int = 100000,
    unique_only: bool = False,
) -> None:
    rows = read_rows(input_path)
    if not rows:
        raise ValueError("No readable rows found in input file.")

    lengths = [len(r["item_name_raw"]) for r in rows if r["item_name_raw"]]
    token_counts = [len([t for t in r["item_name_raw"].split(" ") if t]) for r in rows if r["item_name_raw"]]
    length_bounds = iqr_bounds(lengths, multiplier=1.5)
    token_bounds = iqr_bounds(token_counts, multiplier=1.5)

    cleaned_candidates: List[Dict[str, str]] = []
    outliers_removed = 0
    for row in rows:
        name = row["item_name_raw"]
        if is_name_outlier(name, length_bounds, token_bounds):
            outliers_removed += 1
            continue
        merge_key = item_merge_key(name)
        if not merge_key:
            outliers_removed += 1
            continue
        row["merge_key"] = merge_key
        cleaned_candidates.append(row)

    key_name_counter: Dict[str, Counter] = {}
    for row in cleaned_candidates:
        key = row["merge_key"]
        key_name_counter.setdefault(key, Counter())[row["item_name_raw"]] += 1

    key_to_canonical_name: Dict[str, str] = {}
    for key, counter in key_name_counter.items():
        most_common_name, _ = counter.most_common(1)[0]
        most_common_name = denoise_item_name(most_common_name)
        if len(most_common_name) >= len(key):
            key_to_canonical_name[key] = title_case_phrase(most_common_name)
        else:
            key_to_canonical_name[key] = title_case_phrase(key)

    unique_keys = sorted(key_to_canonical_name.keys())
    key_to_numeric_code = {
        key: str(start_product_code + idx) for idx, key in enumerate(unique_keys)
    }

    output_rows: List[Dict[str, str]] = []
    if unique_only:
        first_item_id_per_key: Dict[str, str] = {}
        for row in cleaned_candidates:
            key = row["merge_key"]
            if key not in first_item_id_per_key:
                first_item_id_per_key[key] = row["item_id"]

        for key in unique_keys:
            output_rows.append(
                {
                    "item_id": first_item_id_per_key.get(key, ""),
                    "item_name": key_to_canonical_name[key],
                    "product_code": key_to_numeric_code[key],
                }
            )
    else:
        for row in cleaned_candidates:
            key = row["merge_key"]
            output_rows.append(
                {
                    "item_id": row["item_id"],
                    "item_name": key_to_canonical_name[key],
                    "product_code": key_to_numeric_code[key],
                }
            )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "item_name", "product_code"])
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "input_path": input_path,
        "output_path": output_path,
        "total_input_rows": len(rows),
        "rows_after_cleaning": len(output_rows),
        "rows_removed_outliers": outliers_removed,
        "unique_item_names_before": len(set(r["item_name_raw"] for r in rows)),
        "unique_item_names_after_merge": len(unique_keys),
        "unique_only_output": unique_only,
        "length_bounds_iqr": {"min": length_bounds[0], "max": length_bounds[1]},
        "token_bounds_iqr": {"min": token_bounds[0], "max": token_bounds[1]},
        "product_code_numeric_start": start_product_code,
    }
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess ItemDetails CSV: normalize names, merge similar items, assign unique numeric product codes, remove outliers."
    )
    parser.add_argument(
        "--input",
        default="ItemDetails (1).csv",
        help="Input ItemDetails CSV path.",
    )
    parser.add_argument(
        "--output",
        default="data/ItemDetails_preprocessed.csv",
        help="Output cleaned CSV path.",
    )
    parser.add_argument(
        "--report",
        default="data/ItemDetails_preprocess_report.json",
        help="Output report JSON path.",
    )
    parser.add_argument(
        "--start-product-code",
        type=int,
        default=100000,
        help="Starting integer value for generated numeric product codes.",
    )
    parser.add_argument(
        "--unique-only",
        action="store_true",
        help="If set, output only real unique merged item names (one row per item).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        start_product_code=args.start_product_code,
        unique_only=args.unique_only,
    )
