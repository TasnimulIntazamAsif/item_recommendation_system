import argparse
import csv
import datetime as dt
import difflib
import json
import math
import os
import re
import statistics
from typing import Any, Dict, List, Set, Tuple


EXPECTED_COLUMNS = [
    "customerId",
    "customerName",
    "itemName",
    "itemId",
    "itemCode",
    "quantity",
    "orderDate",
    "netDiscount",
    "lineDiscount",
]

OUTPUT_COLUMNS = EXPECTED_COLUMNS + ["order_id"]

PLACEHOLDER_ITEMNAME_PATTERNS = [
    r"^avg[_\s-]*item$",
    r"^test(\b|[_\s-])",
    r"\btest\b",
    r"^test",
    r"^sample(\b|[_\s-])",
    r"^(new|var)$",
    r"^new(\b|[_\s-])",
    r"^var(\b|[_\s-])",
    r"purchase\s+item",
    r"vat\s*check",
    r"test\s*vat",
    r"^unknown$",
    r"^n/?a$",
    r"^none$",
    r"^null$",
    r"^undefined$",
]


def normalize_space(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def normalize_item_name(name: str) -> str:
    s = normalize_space(name)
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    s = s.replace('"""', '"').replace('""', '"')
    return normalize_space(s)


def normalize_order_date(value: str) -> str:
    """
    Convert datetime-like values to date-only format (YYYY-MM-DD).
    """
    s = normalize_space(value)
    if not s:
        return s
    try:
        return dt.datetime.fromisoformat(s).date().isoformat()
    except ValueError:
        pass
    try:
        # Handles values like "2023-04-12 00:00:00"
        return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").date().isoformat()
    except ValueError:
        return s


def item_name_match_key(name: str) -> str:
    """
    Aggressive normalization for matching/merging similar names.
    Keeps only alphanumerics and single spaces, casefolded.
    """
    s = normalize_item_name(name).casefold()
    s = re.sub(r"[^\w\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def item_name_tokens(name: str) -> List[str]:
    key = item_name_match_key(name)
    toks = [t for t in key.split(" ") if t]
    return [t for t in toks if len(t) >= 2]


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def name_similarity(a: str, b: str) -> Tuple[float, float]:
    ak = item_name_match_key(a)
    bk = item_name_match_key(b)
    seq = difflib.SequenceMatcher(a=ak, b=bk).ratio()
    tok = jaccard(set(item_name_tokens(a)), set(item_name_tokens(b)))
    return seq, tok


def is_unusual_or_fake_item_name(name: str) -> Tuple[bool, str]:
    s = normalize_item_name(name)
    if s == "":
        return True, "itemName_empty"
    if len(s) < 3:
        return True, "itemName_too_short"

    lower = s.lower()
    # Very common dummy values appear as standalone single words.
    if lower in {"new", "var"}:
        return True, "itemName_placeholder"
    for pat in PLACEHOLDER_ITEMNAME_PATTERNS:
        if re.search(pat, lower):
            return True, "itemName_placeholder"

    if "http://" in lower or "https://" in lower or "www." in lower:
        return True, "itemName_url_like"

    letters = sum(1 for ch in s if ch.isalpha())
    digits = sum(1 for ch in s if ch.isdigit())
    if letters <= 1 and digits >= 3:
        return True, "itemName_mostly_numeric"

    # Allow common punctuation used in product names.
    allowed_punct = set(" .,&/+-()[]%:'\"{}|")
    bad_punct = sum(1 for ch in s if (not ch.isalnum()) and (ch not in allowed_punct))
    if bad_punct >= 2:
        return True, "itemName_weird_punctuation"

    # "Testttttttt", "aaaaaa" style garbage.
    if re.search(r"(.)\1{5,}", lower):
        return True, "itemName_repetitive_garbage"

    if len(set(lower)) <= 2 and len(s) >= 8:
        return True, "itemName_repetitive_garbage"

    return False, "ok"


def parse_float(value: str) -> float:
    return float(str(value).strip())


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def iqr_upper_bound(values: List[float], multiplier: float = 3.0) -> float:
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    return q3 + multiplier * iqr


def to_clean_row(raw: Dict[str, str]) -> Dict[str, str]:
    row = {k: raw.get(k, "") for k in EXPECTED_COLUMNS}
    row["customerId"] = normalize_space(row["customerId"])
    row["customerName"] = normalize_space(row["customerName"])
    row["itemName"] = normalize_item_name(row["itemName"])
    row["itemId"] = normalize_space(row["itemId"])
    row["itemCode"] = normalize_space(row["itemCode"])
    row["quantity"] = normalize_space(row["quantity"])
    row["orderDate"] = normalize_order_date(row["orderDate"])
    row["netDiscount"] = normalize_space(row["netDiscount"])
    row["lineDiscount"] = normalize_space(row["lineDiscount"])
    return row


def is_valid_base(row: Dict[str, str]) -> Tuple[bool, str]:
    for col in EXPECTED_COLUMNS:
        if row[col] == "":
            return False, f"missing_{col}"

    bad_name, name_reason = is_unusual_or_fake_item_name(row["itemName"])
    if bad_name:
        return False, name_reason

    try:
        quantity = parse_float(row["quantity"])
        net_discount = parse_float(row["netDiscount"])
        line_discount = parse_float(row["lineDiscount"])
    except ValueError:
        return False, "invalid_numeric"

    if quantity <= 0:
        return False, "quantity_non_positive"
    if net_discount < 0 or line_discount < 0:
        return False, "negative_discount"

    try:
        dt.datetime.fromisoformat(row["orderDate"])
    except ValueError:
        return False, "invalid_order_date"

    return True, "ok"


def format_number(value: float) -> str:
    if abs(value - int(value)) < 1e-9:
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def build_order_id_map(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], str]:
    """
    Create opaque order IDs (no date string in the ID itself).
    Group key is still (customerId, orderDate), but exposed ID is ORD000001 style.
    """
    keys = sorted(
        {(normalize_space(r["customerId"]), normalize_space(r["orderDate"])) for r in rows},
        key=lambda x: (x[0], x[1]),
    )
    return {k: f"ORD{i:06d}" for i, k in enumerate(keys, start=1)}


def build_merge_map(
    item_counts: Dict[str, int],
    *,
    seq_threshold: float,
    token_threshold: float,
    max_candidates_per_bucket: int = 300,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Greedy canonicalization:
    - bucket by first 3 chars of match key (fast pruning)
    - choose best representative inside bucket by similarity thresholds
    """
    names = sorted(item_counts.keys(), key=lambda n: (-item_counts[n], n.casefold()))
    buckets: Dict[str, List[str]] = {}

    def bucket_key(name: str) -> str:
        k = item_name_match_key(name)
        return k[:3] if len(k) >= 3 else k

    # Representatives: most frequent name in each bucket becomes a rep by default.
    reps: Dict[str, List[str]] = {}
    for name in names:
        b = bucket_key(name)
        reps.setdefault(b, [])
        if not reps[b]:
            reps[b].append(name)

    merge_map: Dict[str, str] = {}
    merges: List[Dict[str, Any]] = []

    for name in names:
        b = bucket_key(name)
        candidates = reps.get(b, [])
        if len(candidates) > max_candidates_per_bucket:
            candidates = candidates[:max_candidates_per_bucket]

        best_rep = None
        best_seq = 0.0
        best_tok = 0.0

        for rep in candidates:
            seq, tok = name_similarity(name, rep)
            if seq >= seq_threshold and tok >= token_threshold:
                if seq > best_seq or (abs(seq - best_seq) < 1e-9 and tok > best_tok):
                    best_rep = rep
                    best_seq = seq
                    best_tok = tok

        if best_rep is None:
            # New representative for this bucket.
            reps.setdefault(b, []).append(name)
            merge_map[name] = name
            continue

        merge_map[name] = best_rep
        if best_rep != name:
            merges.append(
                {
                    "from": name,
                    "to": best_rep,
                    "from_count": item_counts.get(name, 0),
                    "seq_ratio": round(best_seq, 4),
                    "token_jaccard": round(best_tok, 4),
                }
            )

    summary = {
        "unique_item_names_before": len(item_counts),
        "unique_item_names_after": len(set(merge_map.values())),
        "total_merges": sum(1 for k, v in merge_map.items() if k != v),
        "seq_threshold": seq_threshold,
        "token_threshold": token_threshold,
        "merge_examples": merges[:50],
    }
    return merge_map, summary


def clean_dataset(
    input_path: str,
    output_path: str,
    report_path: str,
    *,
    min_item_freq: int = 2,
    min_basket_items: int = 2,
    merge_seq_threshold: float = 0.92,
    merge_token_threshold: float = 0.6,
) -> None:
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        raw_rows = [to_clean_row(r) for r in reader]

    total_rows = len(raw_rows)
    removal_reasons: Dict[str, int] = {}
    base_valid_rows: List[Dict[str, str]] = []
    for row in raw_rows:
        ok, reason = is_valid_base(row)
        if not ok:
            removal_reasons[reason] = removal_reasons.get(reason, 0) + 1
            continue
        base_valid_rows.append(row)

    item_counts: Dict[str, int] = {}
    for r in base_valid_rows:
        item_counts[r["itemName"]] = item_counts.get(r["itemName"], 0) + 1

    quantities = [parse_float(r["quantity"]) for r in base_valid_rows]
    net_discounts = [parse_float(r["netDiscount"]) for r in base_valid_rows]
    line_discounts = [parse_float(r["lineDiscount"]) for r in base_valid_rows]

    quantity_upper = iqr_upper_bound(quantities, multiplier=3.0)
    net_discount_upper = iqr_upper_bound(net_discounts, multiplier=3.0)
    # lineDiscount is often zero-inflated in retail exports; IQR can collapse to 0.
    # Use a high percentile cap so valid positive discounts are preserved.
    line_discount_upper = percentile(line_discounts, 0.995)

    cleaned_rows_pre_merge: List[Dict[str, str]] = []
    outlier_removed = 0
    for row in base_valid_rows:
        q = parse_float(row["quantity"])
        nd = parse_float(row["netDiscount"])
        ld = parse_float(row["lineDiscount"])
        if q > quantity_upper or nd > net_discount_upper or ld > line_discount_upper:
            outlier_removed += 1
            continue

        row["quantity"] = format_number(q)
        row["netDiscount"] = format_number(nd)
        row["lineDiscount"] = format_number(ld)
        cleaned_rows_pre_merge.append(row)

    merge_map, merge_summary = build_merge_map(
        item_counts,
        seq_threshold=merge_seq_threshold,
        token_threshold=merge_token_threshold,
    )

    for r in cleaned_rows_pre_merge:
        r["itemName"] = merge_map.get(r["itemName"], r["itemName"])

    # Create opaque order_id (without embedding date text) and build baskets.
    order_id_map = build_order_id_map(cleaned_rows_pre_merge)
    for r in cleaned_rows_pre_merge:
        key = (normalize_space(r["customerId"]), normalize_space(r["orderDate"]))
        r["order_id"] = order_id_map[key]

    baskets: Dict[str, Set[str]] = {}
    for r in cleaned_rows_pre_merge:
        baskets.setdefault(r["order_id"], set()).add(r["itemName"])

    # Remove single-item (or small) baskets
    kept_order_ids: Set[str] = set()
    removed_baskets = 0
    for oid, items in baskets.items():
        if len(items) >= min_basket_items:
            kept_order_ids.add(oid)
        else:
            removed_baskets += 1

    rows_after_basket_filter: List[Dict[str, str]] = [
        r for r in cleaned_rows_pre_merge if r["order_id"] in kept_order_ids
    ]

    # Rare item removal should be applied after basket filtering.
    merged_counts: Dict[str, int] = {}
    for r in rows_after_basket_filter:
        merged_counts[r["itemName"]] = merged_counts.get(r["itemName"], 0) + 1

    cleaned_rows: List[Dict[str, str]] = []
    rare_removed = 0
    for r in rows_after_basket_filter:
        if merged_counts.get(r["itemName"], 0) < min_item_freq:
            rare_removed += 1
            continue
        cleaned_rows.append(r)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    # Save baskets file (order_id -> items list)
    baskets_path = os.path.splitext(output_path)[0] + "_baskets.csv"
    basket_rows: List[Dict[str, str]] = []
    for oid in sorted(kept_order_ids):
        items = sorted(baskets.get(oid, set()), key=lambda x: x.casefold())
        sample_row = next((x for x in cleaned_rows_pre_merge if x["order_id"] == oid), None)
        basket_rows.append(
            {
                "order_id": oid,
                "customerId": sample_row["customerId"] if sample_row else "",
                "orderDate": sample_row["orderDate"] if sample_row else "",
                "basket_size": str(len(items)),
                "items": " | ".join(items),
            }
        )
    os.makedirs(os.path.dirname(baskets_path) or ".", exist_ok=True)
    with open(baskets_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["order_id", "customerId", "orderDate", "basket_size", "items"]
        )
        writer.writeheader()
        writer.writerows(basket_rows)

    mapping_path = os.path.splitext(output_path)[0] + "_itemName_merge_map.json"
    os.makedirs(os.path.dirname(mapping_path) or ".", exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {"merge_map": merge_map, "summary": merge_summary},
            f,
            indent=2,
            ensure_ascii=False,
        )

    report = {
        "input_path": input_path,
        "output_path": output_path,
        "baskets_path": baskets_path,
        "itemName_merge_map_path": mapping_path,
        "total_input_rows": total_rows,
        "rows_after_base_validation": len(base_valid_rows),
        "rows_removed_base_validation": total_rows - len(base_valid_rows),
        "rows_removed_outliers": outlier_removed,
        "min_basket_items": min_basket_items,
        "orders_total_after_outliers": len(baskets),
        "orders_removed_small_baskets": removed_baskets,
        "rows_removed_rare_items": rare_removed,
        "min_item_freq": min_item_freq,
        "total_rows_removed": total_rows - len(cleaned_rows),
        "total_rows_cleaned": len(cleaned_rows),
        "removal_reasons_base_validation": removal_reasons,
        "itemName_merge_summary": merge_summary,
        "outlier_thresholds": {
            "quantity_upper_iqr_3x": quantity_upper,
            "netDiscount_upper_iqr_3x": net_discount_upper,
            "lineDiscount_upper_iqr_3x": line_discount_upper,
        },
        "stats_before_filter": {
            "quantity_mean": statistics.mean(quantities) if quantities else 0,
            "netDiscount_mean": statistics.mean(net_discounts) if net_discounts else 0,
            "lineDiscount_mean": statistics.mean(line_discounts) if line_discounts else 0,
        },
    }

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean sales dataset: normalize item names, merge similar items, create order baskets, remove rare items."
    )
    parser.add_argument(
        "--input",
        default="Mangagerium_sales.csv",
        help="Path to raw input CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/Mangagerium_sales_clean.csv",
        help="Path to save cleaned CSV.",
    )
    parser.add_argument(
        "--report",
        default="data/Mangagerium_sales_clean_report.json",
        help="Path to save cleaning report JSON.",
    )
    parser.add_argument(
        "--min-item-freq",
        type=int,
        default=2,
        help="Remove items whose (post-merge) frequency is below this number.",
    )
    parser.add_argument(
        "--min-basket-items",
        type=int,
        default=2,
        help="Remove orders/baskets with fewer distinct items than this number (removes single-item baskets by default).",
    )
    parser.add_argument(
        "--merge-seq-threshold",
        type=float,
        default=0.92,
        help="Sequence similarity threshold for merging item names.",
    )
    parser.add_argument(
        "--merge-token-threshold",
        type=float,
        default=0.6,
        help="Token Jaccard threshold for merging item names.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_dataset(
        args.input,
        args.output,
        args.report,
        min_item_freq=args.min_item_freq,
        min_basket_items=args.min_basket_items,
        merge_seq_threshold=args.merge_seq_threshold,
        merge_token_threshold=args.merge_token_threshold,
    )
