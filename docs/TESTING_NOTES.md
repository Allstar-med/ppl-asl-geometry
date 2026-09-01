# Testing notes

Core numerical tests do not require image-I/O dependencies.

Geometry/resampling tests require `SimpleITK`, which is pinned in `requirements.txt`.
If `SimpleITK` is unavailable in the current environment, pytest marks those tests as skipped rather than failing collection.

For a release verification environment, install all pinned requirements and rerun:

```bash
pip install -r requirements.txt
pytest -q
python scripts/verify_no_hardcoded_results.py
```

The static audit also rejects a literal radius passed directly to
`legacy_iopr`; analysis code must obtain that value from configuration.
