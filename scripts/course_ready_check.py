
#!/usr/bin/env python3
"""
APEX-1 Course Ready Check

This script verifies that the repository is ready to be used as the stable base
for the APEX-1 v3.0.0 course release.

Modes:
  quick     - static checks only: required files, versions, docs references
  examples  - quick checks + run all course demo scripts
  tests     - quick checks + run pytest
  full      - quick checks + examples + pytest

Output:
  outputs/course_ready_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "outputs" / "course_ready_report.json"

EXPECTED_VERSION = "3.0.0"

REQUIRED_FILES = [
    "README.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "apex/__init__.py",
    "MODEL_CARD.md",
    "COURSE_READY_CHECKLIST.md",
    "docs/38-course-ready-release.md",
    ".github/workflows/ci.yml",
    ".github/pull_request_template.md",
]

REQUIRED_DOCS = [
    "docs/00-introduction.md",
    "docs/01-project-structure.md",
    "docs/33-lora-peft-finetuning.md",
    "docs/34-lora-inference-and-merge.md",
    "docs/35-qlora-4bit-finetuning.md",
    "docs/36-dora-weight-decomposed-lora.md",
    "docs/37-adapter-dpo-alignment.md",
    "docs/38-course-ready-release.md",
]

EXAMPLE_COMMANDS = [
    ["python", "examples/forward_pass_demo.py"],
    ["python", "examples/generation_demo.py"],
    ["python", "examples/thinking_mode_demo.py"],
    ["python", "examples/mask_visualization.py"],
    ["python", "examples/vision_forward_demo.py"],
    ["python", "examples/lora_finetune_demo.py"],
    ["python", "examples/lora_generation_demo.py"],
    ["python", "examples/qlora_finetune_demo.py"],
    ["python", "examples/dora_finetune_demo.py"],
    ["python", "examples/adapter_dpo_demo.py"],
]

TEST_COMMANDS = [
    ["pytest", "tests/", "-v", "--tb=short"],
]


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def command_to_text(command: Iterable[str]) -> str:
    return " ".join(command)


def run_command(command: list[str], timeout: int) -> CheckResult:
    label = command_to_text(command)
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return CheckResult(label, "fail", f"command not found: {exc}")
    except subprocess.TimeoutExpired:
        return CheckResult(label, "fail", f"timed out after {timeout}s")

    output_tail = "\\n".join((completed.stdout + "\\n" + completed.stderr).splitlines()[-40:])
    if completed.returncode == 0:
        return CheckResult(label, "pass", output_tail)
    return CheckResult(label, "fail", output_tail)


def check_required_files() -> list[CheckResult]:
    results: list[CheckResult] = []
    for file_path in REQUIRED_FILES:
        exists = (ROOT / file_path).exists()
        results.append(
            CheckResult(
                f"required file: {file_path}",
                "pass" if exists else "fail",
                "found" if exists else "missing",
            )
        )
    return results


def check_required_docs() -> list[CheckResult]:
    results: list[CheckResult] = []
    for file_path in REQUIRED_DOCS:
        exists = (ROOT / file_path).exists()
        results.append(
            CheckResult(
                f"required doc: {file_path}",
                "pass" if exists else "fail",
                "found" if exists else "missing",
            )
        )
    return results


def check_versions() -> list[CheckResult]:
    results: list[CheckResult] = []

    pyproject = read_text("pyproject.toml") if (ROOT / "pyproject.toml").exists() else ""
    init_py = read_text("apex/__init__.py") if (ROOT / "apex/__init__.py").exists() else ""
    readme = read_text("README.md") if (ROOT / "README.md").exists() else ""
    changelog = read_text("CHANGELOG.md") if (ROOT / "CHANGELOG.md").exists() else ""

    checks = [
        ("pyproject version", f'version = "{EXPECTED_VERSION}"' in pyproject),
        ("package __version__", f'__version__ = "{EXPECTED_VERSION}"' in init_py),
        ("README mentions v3.0.0", "v3.0.0" in readme),
        ("CHANGELOG has v3.0.0", "## v3.0.0" in changelog),
    ]

    for name, ok in checks:
        results.append(CheckResult(name, "pass" if ok else "fail", ""))

    return results


def check_readme_course_consistency() -> list[CheckResult]:
    readme = read_text("README.md") if (ROOT / "README.md").exists() else ""

    checks = [
        ("README says 37 lessons", "37 Lessons" in readme or "37 lessons" in readme),
        ("README includes DoRA lesson", "36" in readme and "DoRA" in readme),
        ("README includes Adapter-DPO lesson", "37" in readme and "Adapter-DPO" in readme),
        ("README includes educational limitation note", "does not ship with a large pretrained checkpoint" in readme.lower()),
        ("README no private planning reference", "private planning" not in readme.lower()),
    ]

    return [CheckResult(name, "pass" if ok else "fail", "") for name, ok in checks]


def check_python_import() -> list[CheckResult]:
    return [run_command(["python", "-c", "import apex; print(apex.__version__)"], timeout=30)]


def run_examples() -> list[CheckResult]:
    return [run_command(command, timeout=180) for command in EXAMPLE_COMMANDS]


def run_tests() -> list[CheckResult]:
    return [run_command(command, timeout=900) for command in TEST_COMMANDS]


def write_report(mode: str, results: list[CheckResult]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total = len(results)
    failed = [result for result in results if result.status != "pass"]
    report = {
        "project": "APEX-1",
        "expected_version": EXPECTED_VERSION,
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": total,
            "passed": total - len(failed),
            "failed": len(failed),
            "status": "pass" if not failed else "fail",
        },
        "results": [asdict(result) for result in results],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check APEX-1 course-ready stability.")
    parser.add_argument(
        "--mode",
        choices=["quick", "examples", "tests", "full"],
        default="quick",
        help="Check mode to run.",
    )
    args = parser.parse_args()

    results: list[CheckResult] = []
    results.extend(check_required_files())
    results.extend(check_required_docs())
    results.extend(check_versions())
    results.extend(check_readme_course_consistency())
    results.extend(check_python_import())

    if args.mode in {"examples", "full"}:
        results.extend(run_examples())

    if args.mode in {"tests", "full"}:
        results.extend(run_tests())

    write_report(args.mode, results)

    failed = [result for result in results if result.status != "pass"]
    print(f"APEX-1 course-ready check: {len(results) - len(failed)}/{len(results)} passed")
    print(f"Report written to: {REPORT_PATH.relative_to(ROOT)}")

    if failed:
        print("\\nFailed checks:")
        for result in failed:
            print(f"- {result.name}: {result.detail[:300]}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
