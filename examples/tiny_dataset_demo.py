from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import json
from pathlib import Path


def main() -> None:
    sample_dir = Path(__file__).resolve().parents[1] / "data" / "samples"
    for name in ["tiny_text.jsonl", "tiny_sft.jsonl", "tiny_preference.jsonl", "tiny_vision.jsonl"]:
        path = sample_dir / name
        print(f"\n{name}")
        with path.open("r", encoding="utf-8") as f:
            first = json.loads(f.readline())
        print(first)


if __name__ == "__main__":
    main()
