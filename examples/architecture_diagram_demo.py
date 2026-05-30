from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from apex.config import get_tiny_vision_config
from apex.utils.architecture_diagram import build_architecture_diagram


def main() -> None:
    cfg = get_tiny_vision_config()
    cfg.validate()
    print(build_architecture_diagram(cfg, title="APEX-1 Tiny Vision"))


if __name__ == "__main__":
    main()
