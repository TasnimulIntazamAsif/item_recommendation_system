import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


DEFAULT_INPUT = "data/Mangagerium_sales_clean.csv"
DEFAULT_OUTPUT_TXT = "data/unique_item_names.txt"
DEFAULT_OUTPUT_JSON = "data/unique_item_names_report.json"


def normalize_space(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_item_name(name: str) -> str:
    s = normalize_space(name)
    # Remove wrapping quotes only (CSV already handles escaping).
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    # Normalize odd quote patterns like: """The Clear"" Lemongrass Infusion"
    s = s.replace('"""', '"').replace('""', '"')
    return normalize_space(s)


@dataclass
class ExtractResult:
    unique_names: List[str]
    counts: Dict[str, int]


def extract_unique_item_names(input_csv: str) -> ExtractResult:
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "itemName" not in reader.fieldnames:
            raise ValueError("Input CSV must contain an `itemName` column.")

        seen: Set[str] = set()
        kept: List[str] = []
        counts: Dict[str, int] = {}

        for row in reader:
            raw = row.get("itemName", "")
            name = normalize_item_name(raw)
            key = name.casefold()

            if name == "":
                continue
            counts[name] = counts.get(name, 0) + 1
            if key in seen:
                continue
            seen.add(key)
            kept.append(name)

    kept.sort(key=lambda x: x.casefold())
    return ExtractResult(unique_names=kept, counts=counts)


def write_outputs(result: ExtractResult, output_txt: str, output_json: str, input_csv: str) -> None:
    os.makedirs(os.path.dirname(output_txt) or ".", exist_ok=True)
    with open(output_txt, "w", encoding="utf-8", newline="\n") as f:
        # Write most frequent first (better for quick inspection)
        names_sorted = sorted(
            result.unique_names, key=lambda n: (-result.counts.get(n, 0), n.casefold())
        )
        for name in names_sorted:
            f.write(name + "\n")

    report = {
        "input_csv": input_csv,
        "output_txt": output_txt,
        "total_unique_item_names": len(result.unique_names),
        "top_items": [
            {"itemName": n, "count": result.counts.get(n, 0)}
            for n in sorted(result.unique_names, key=lambda x: (-result.counts.get(x, 0), x.casefold()))[:50]
        ],
        "notes": [
            "This script assumes the dataset is already cleaned (normalized/merged/rare-removed).",
        ],
    }

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract unique itemName list from cleaned dataset.")
    p.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path (default: cleaned dataset).")
    p.add_argument("--output-txt", default=DEFAULT_OUTPUT_TXT, help="Output text file path (one name per line).")
    p.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="Output report JSON path.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    res = extract_unique_item_names(args.input)
    write_outputs(res, args.output_txt, args.output_json, args.input)
