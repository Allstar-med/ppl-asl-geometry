from __future__ import annotations

import ast
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]

PROHIBITED_PATTERNS = [
    r"degrade_mask_to_dsc",
    r"target[_ -]?dsc",
    r"0\.76\s*[±+/-]",
]

files_to_check = list((ROOT / "src").rglob("*.py"))
files_to_check += [
    path
    for path in (ROOT / "scripts").rglob("*.py")
    if path.name != "verify_no_hardcoded_results.py"
]

failures = []

for path in files_to_check:
    text = path.read_text(encoding="utf-8")

    for pattern in PROHIBITED_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            failures.append(
                (str(path.relative_to(ROOT)), pattern)
            )

    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        function_name = None
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr

        if function_name != "legacy_iopr":
            continue

        radius = node.args[2] if len(node.args) >= 3 else None
        for keyword in node.keywords:
            if keyword.arg == "radius_voxels":
                radius = keyword.value

        if (
            isinstance(radius, ast.Constant)
            and isinstance(radius.value, (int, float))
            and not isinstance(radius.value, bool)
        ):
            failures.append((
                str(path.relative_to(ROOT)),
                f"literal legacy_iopr radius at line {node.lineno}",
            ))

if failures:
    print("FAILED:", failures)
    sys.exit(1)

print(
    "PASS: no prohibited target-conditioned/result-specific "
    "patterns found."
)
