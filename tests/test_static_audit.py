from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_static_audit_rejects_literal_legacy_radius(tmp_path):
    scripts = tmp_path / "scripts"
    source = tmp_path / "src"
    scripts.mkdir()
    source.mkdir()

    audit_script = scripts / "verify_no_hardcoded_results.py"
    shutil.copy2(
        ROOT / "scripts" / "verify_no_hardcoded_results.py",
        audit_script,
    )
    (source / "example.py").write_text(
        "prediction = legacy_iopr(organ_a, organ_b, 3)\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(audit_script)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "literal legacy_iopr radius" in completed.stdout
