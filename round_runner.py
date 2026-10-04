from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
SRC_FILE = SRC_DIR / "round_runner.py"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

spec = importlib.util.spec_from_file_location("round_runner_impl", SRC_FILE)
if spec is None or spec.loader is None:
    raise SystemExit(f"Cannot load runner implementation: {SRC_FILE}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


if __name__ == "__main__":
    module.main()
