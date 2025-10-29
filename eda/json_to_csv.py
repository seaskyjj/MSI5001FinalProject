#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert training history JSON to CSV.")
    parser.add_argument("json_path", type=Path, help="Path to training_history.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (defaults to same name as JSON).",
    )
    args = parser.parse_args()

    # Default to mirroring the input JSON name when ``--output`` is unspecified.
    output_path = args.output or args.json_path.with_suffix(".csv")

    with args.json_path.open("r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    if not isinstance(data, list) or not data:
        raise ValueError("Expected a non-empty JSON array of objects.")

    fieldnames = list(data[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Wrote {len(data)} rows to {output_path}")


if __name__ == "__main__":
    main()
