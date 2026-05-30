from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from apex.config import APEXConfig, get_tiny_config, get_tiny_vision_config
from apex.utils.architecture_diagram import build_architecture_diagram, build_layer_table


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
    parser = argparse.ArgumentParser(description="Print APEX-1 architecture diagram.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path. Defaults to tiny config.")
    parser.add_argument("--vision", action="store_true", help="Show vision pipeline.")
    parser.add_argument("--table", action="store_true", help="Print Markdown layer table instead of ASCII diagram.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.vision)
    if args.table:
        print(build_layer_table(cfg))
    else:
        print(build_architecture_diagram(cfg, title="APEX-1 Architecture"))


if __name__ == "__main__":
    main()
