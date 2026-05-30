from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from apex.config import get_tiny_vision_config
from apex.model.apex_vision_model import APEX1VisionModel
from apex.utils.model_inspector import format_inspection_report, inspect_model


def main() -> None:
    cfg = get_tiny_vision_config()
    cfg.validate()
    model = APEX1VisionModel(cfg)
    report = inspect_model(model)
    print(format_inspection_report(report))


if __name__ == "__main__":
    main()
