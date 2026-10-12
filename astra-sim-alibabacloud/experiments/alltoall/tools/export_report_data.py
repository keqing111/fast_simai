#!/usr/bin/env python3
"""Export sanitized campaign aggregates used by the tracked report figures."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = (
    "e1-pilot",
    "e2-one-seed",
    "e3-native-five-seed",
    "e4-fixed-total",
    "e5-fixed-per-device-scale",
    "e7-load-sweep",
)


def main() -> None:
    output = ROOT / "report_data"
    output.mkdir(parents=True, exist_ok=True)
    for campaign in CAMPAIGNS:
        source = ROOT / "results" / campaign / "aggregate.json"
        rows = json.loads(source.read_text(encoding="utf-8"))
        sanitized = [
            {key: value for key, value in row.items() if key != "attempt"}
            for row in rows
        ]
        (output / f"{campaign}.json").write_text(
            json.dumps(sanitized, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
