import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build feature-engineering artifacts for recommendation system."
    )
    p.add_argument("--input", default="data/form_paste_clean_v2.csv", help="Cleaned input CSV path.")
    p.add_argument("--output-dir", default="data/features", help="Directory to write feature files.")
    p.add_argument(
        "--interaction-value",
        choices=["quantity", "frequency"],
        default="quantity",
        help="Value type for item-user matrix.",
    )
    p.add_argument(
        "--dense-onehot",
        action="store_true",
        help="Also write dense one-hot matrix (can be very large).",
    )
    p.add_argument(
        "--min-cooccur",
        type=int,
        default=1,
        help="Minimum co-purchase count to keep in co-occurrence output.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap for rows processed (0 means all rows).",
    )
    return p.parse_args()


def normalize_space(value: str) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def parse_float(value: str) -> float:
    try:
        return float(str(value).strip())
    except ValueError:
        return 0.0


def parse_order_datetime(text: str) -> Optional[datetime]:
    s = normalize_space(text)
    if not s:
        return None
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def derive_category(item_name: str, row: Dict[str, str]) -> str:
    # If dataset has explicit category column, prefer it.
    for key in ("category", "itemCategory", "item_type", "itemType"):
        if key in row and normalize_space(row[key]):
            return normalize_space(row[key])

    code = normalize_space(row.get("itemCode", ""))
    alpha_prefix = "".join(ch for ch in code if ch.isalpha())
    if alpha_prefix:
        return alpha_prefix[:4].upper()

    tokens = normalize_space(item_name).split()
    return tokens[0].upper() if tokens else "UNKNOWN"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    required_cols = {"customerId", "itemName", "orderDate", "order_id"}
    tx_items: Dict[str, Set[str]] = defaultdict(set)  # order_id -> unique items
    tx_item_qty: Dict[str, Counter] = defaultdict(Counter)  # order_id -> item -> qty sum
    user_item: Dict[str, Counter] = defaultdict(Counter)  # customerId -> item -> interaction
    item_pop_count: Counter = Counter()  # item -> row count
    item_pop_qty: Counter = Counter()  # item -> quantity sum
    item_category: Dict[str, str] = {}  # item -> category
    time_rows: List[Dict[str, object]] = []

    processed = 0
    with open(args.input, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in required_cols if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Input missing required columns: {missing}")

        for row in reader:
            if args.max_rows and processed >= args.max_rows:
                break

            order_id = normalize_space(row.get("order_id", ""))
            customer_id = normalize_space(row.get("customerId", ""))
            item = normalize_space(row.get("itemName", ""))
            order_date = normalize_space(row.get("orderDate", ""))
            qty = parse_float(row.get("quantity", "1"))

            if not order_id or not customer_id or not item:
                continue

            processed += 1
            tx_items[order_id].add(item)
            tx_item_qty[order_id][item] += qty
            item_pop_count[item] += 1
            item_pop_qty[item] += qty
            item_category.setdefault(item, derive_category(item, row))

            interaction_value = qty if args.interaction_value == "quantity" else 1.0
            user_item[customer_id][item] += interaction_value

            dt_obj = parse_order_datetime(order_date)
            time_rows.append(
                {
                    "order_id": order_id,
                    "customerId": customer_id,
                    "orderDate": order_date,
                    "month": dt_obj.month if dt_obj else "",
                    "day_of_week": dt_obj.strftime("%A") if dt_obj else "",
                    "hour": dt_obj.hour if dt_obj else 0,
                    "has_time_component": int(bool(dt_obj and ("T" in order_date or ":" in order_date))),
                }
            )

    # 3.1 One-Hot Matrix (Transaction x Item), sparse format always.
    onehot_sparse_rows: List[Dict[str, object]] = []
    for oid, items in tx_items.items():
        for item in sorted(items, key=str.casefold):
            onehot_sparse_rows.append({"order_id": oid, "itemName": item, "value": 1})
    write_csv(
        os.path.join(args.output_dir, "onehot_transaction_item_sparse.csv"),
        ["order_id", "itemName", "value"],
        onehot_sparse_rows,
    )

    # Optional dense one-hot matrix.
    if args.dense_onehot:
        all_items = sorted(item_pop_count.keys(), key=str.casefold)
        dense_path = os.path.join(args.output_dir, "onehot_transaction_item_dense.csv")
        with open(dense_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["order_id"] + all_items)
            for oid in sorted(tx_items.keys()):
                item_set = tx_items[oid]
                writer.writerow([oid] + [1 if item in item_set else 0 for item in all_items])

    # 3.2 Item-User Matrix (Customer x Item), sparse.
    item_user_rows: List[Dict[str, object]] = []
    for customer_id, items in user_item.items():
        for item, val in items.items():
            item_user_rows.append({"customerId": customer_id, "itemName": item, "value": round(val, 6)})
    write_csv(
        os.path.join(args.output_dir, "item_user_matrix_sparse.csv"),
        ["customerId", "itemName", "value"],
        item_user_rows,
    )

    # 3.3 Co-occurrence Matrix (Item x Item), upper-triangle sparse.
    cooccur: Counter = Counter()
    for items in tx_items.values():
        item_list = sorted(items, key=str.casefold)
        for a, b in combinations(item_list, 2):
            cooccur[(a, b)] += 1

    cooccur_rows: List[Dict[str, object]] = []
    for (a, b), cnt in cooccur.items():
        if cnt >= args.min_cooccur:
            cooccur_rows.append({"itemA": a, "itemB": b, "co_purchase_count": cnt})
    cooccur_rows.sort(key=lambda r: (-int(r["co_purchase_count"]), r["itemA"], r["itemB"]))
    write_csv(
        os.path.join(args.output_dir, "item_item_cooccurrence.csv"),
        ["itemA", "itemB", "co_purchase_count"],
        cooccur_rows,
    )

    # 3.4 Popularity Feature.
    pop_rows: List[Dict[str, object]] = []
    ranked = sorted(item_pop_count.keys(), key=lambda x: (-item_pop_count[x], -item_pop_qty[x], x.casefold()))
    for i, item in enumerate(ranked, start=1):
        pop_rows.append(
            {
                "rank": i,
                "itemName": item,
                "purchase_count": int(item_pop_count[item]),
                "total_quantity": round(item_pop_qty[item], 6),
            }
        )
    write_csv(
        os.path.join(args.output_dir, "item_popularity.csv"),
        ["rank", "itemName", "purchase_count", "total_quantity"],
        pop_rows,
    )

    # 3.5 Category Feature.
    category_rows = [{"itemName": item, "category": cat} for item, cat in sorted(item_category.items())]
    write_csv(
        os.path.join(args.output_dir, "item_category_map.csv"),
        ["itemName", "category"],
        category_rows,
    )

    # 3.6 Time Feature.
    write_csv(
        os.path.join(args.output_dir, "order_time_features.csv"),
        ["order_id", "customerId", "orderDate", "month", "day_of_week", "hour", "has_time_component"],
        time_rows,
    )

    # 3.7 Basket Size Feature.
    basket_rows: List[Dict[str, object]] = []
    for oid in sorted(tx_items.keys()):
        basket_size = len(tx_items[oid])
        total_qty = sum(tx_item_qty[oid].values())
        basket_rows.append(
            {
                "order_id": oid,
                "basket_size": basket_size,
                "total_quantity_in_basket": round(total_qty, 6),
            }
        )
    write_csv(
        os.path.join(args.output_dir, "basket_features.csv"),
        ["order_id", "basket_size", "total_quantity_in_basket"],
        basket_rows,
    )

    summary = {
        "input": args.input,
        "rows_processed": processed,
        "transactions": len(tx_items),
        "customers": len(user_item),
        "items": len(item_pop_count),
        "output_dir": args.output_dir,
        "files": [
            "onehot_transaction_item_sparse.csv",
            "onehot_transaction_item_dense.csv (optional via --dense-onehot)",
            "item_user_matrix_sparse.csv",
            "item_item_cooccurrence.csv",
            "item_popularity.csv",
            "item_category_map.csv",
            "order_time_features.csv",
            "basket_features.csv",
        ],
    }
    with open(os.path.join(args.output_dir, "feature_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
