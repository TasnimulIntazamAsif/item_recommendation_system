import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert JSON data to CSV.")
    p.add_argument(
        "--input",
        default="",
        help="Input JSON file path. If omitted, JSON will be read from stdin (paste).",
    )
    p.add_argument(
        "--json-text",
        default="",
        help="Raw JSON text (if you don't want to use a file). For large JSON, prefer stdin paste.",
    )
    p.add_argument("--output", required=True, help="Output CSV file path.")
    p.add_argument(
        "--record-path",
        default="",
        help="Dot path to list of records inside JSON (e.g. data.items). If empty, JSON must be a list.",
    )
    p.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten nested objects into dot-separated columns.",
    )
    p.add_argument(
        "--list-sep",
        default=" | ",
        help="When flattening, join lists using this separator (default: ' | ').",
    )
    p.add_argument(
        "--end-token",
        default="END",
        help="When pasting JSON interactively, type this token on a new line to finish (default: END).",
    )
    return p.parse_args()


def get_by_dot_path(obj: Any, path: str) -> Any:
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(f"record-path not found at '{part}' in '{path}'")
    return cur


def flatten_value(value: Any, *, list_sep: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return list_sep.join(flatten_value(v, list_sep=list_sep) for v in value)
    if isinstance(value, dict):
        # If a dict sneaks in (flatten=False), stringify deterministically.
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def flatten_record(record: Dict[str, Any], *, list_sep: str, prefix: str = "") -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in record.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_record(v, list_sep=list_sep, prefix=key))
        elif isinstance(v, list):
            # If list contains dicts, keep as JSON; otherwise join.
            if any(isinstance(x, dict) for x in v):
                out[key] = json.dumps(v, ensure_ascii=False)
            else:
                out[key] = flatten_value(v, list_sep=list_sep)
        else:
            out[key] = flatten_value(v, list_sep=list_sep)
    return out


def records_to_rows(records: List[Any], *, flatten: bool, list_sep: str) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    field_set = set()

    for r in records:
        if not isinstance(r, dict):
            # If record isn't a dict, store under a single column.
            row = {"value": flatten_value(r, list_sep=list_sep)}
        else:
            row = flatten_record(r, list_sep=list_sep) if flatten else {k: flatten_value(v, list_sep=list_sep) for k, v in r.items()}
        rows.append(row)
        field_set.update(row.keys())

    fieldnames = sorted(field_set)
    return rows, fieldnames


def normalize_json_text(text: str) -> str:
    s = text.strip()
    # Users sometimes wrap JSON in parentheses like: ([{...}, {...}])
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return s


def read_input_json(args: argparse.Namespace) -> Any:
    if args.json_text.strip():
        try:
            return json.loads(normalize_json_text(args.json_text))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON provided via --json-text: {e}. "
                "Make sure keys/strings use double quotes, e.g. {\"a\": 1}."
            ) from e

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            return json.load(f)

    # Read from stdin (paste). If running in an interactive terminal (TTY),
    # support ending with a token (default: END) because EOF signals can be flaky.
    if sys.stdin.isatty():
        print("Paste JSON now. When finished, type the end token on a new line and press Enter.")
        print(f"End token: {args.end_token}")
        lines: List[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == args.end_token:
                break
            lines.append(line)
        raw = "\n".join(lines)
    else:
        raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError(
            "No JSON provided. Use --input FILE.json, or --json-text, or paste JSON via stdin."
        )
    try:
        return json.loads(normalize_json_text(raw))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON provided via stdin: {e}. "
            "If you pasted something like ([{...}]) that's OK, but make sure it's valid JSON "
            "(double quotes, no trailing commas)."
        ) from e


def main() -> None:
    args = parse_args()
    data = read_input_json(args)

    records = get_by_dot_path(data, args.record_path)
    if isinstance(records, dict):
        records = [records]
    elif not isinstance(records, list):
        records = [records]

    rows, fieldnames = records_to_rows(records, flatten=args.flatten, list_sep=args.list_sep)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
