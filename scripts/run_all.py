from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iopr.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "analysis.json"),
    )
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    output = run_pipeline(
        args.manifest,
        args.config,
        args.out,
    )
    print(f"Analysis complete: {output}")


if __name__ == "__main__":
    main()
