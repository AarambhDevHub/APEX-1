from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from apex.config import APEXConfig, get_tiny_config, get_tiny_vision_config
from apex.model.apex_model import APEX1Model
from apex.model.apex_vision_model import APEX1VisionModel
from apex.utils.model_inspector import format_inspection_report, inspect_model


def load_config(path: str | None, vision: bool) -> APEXConfig:
    if path:
        cfg = APEXConfig.from_yaml(path)
    else:
        cfg = get_tiny_vision_config() if vision else get_tiny_config()
    if vision:
        cfg.vision.enabled = True
    cfg.validate()
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect APEX-1 model structure and parameters.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path. Defaults to tiny config.")
    parser.add_argument("--vision", action="store_true", help="Build APEX1VisionModel instead of APEX1Model.")
    parser.add_argument("--no-layers", action="store_true", help="Hide per-layer table.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.vision)
    model = APEX1VisionModel(cfg) if args.vision else APEX1Model(cfg)
    report = inspect_model(model)
    print(format_inspection_report(report, show_layers=not args.no_layers))


if __name__ == "__main__":
    main()
