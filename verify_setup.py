#!/usr/bin/env python3
"""Fast, API-free installation and repository sanity check."""
from pathlib import Path
import importlib
import sys

REQUIRED = [
    "numpy", "torch", "transformers", "sentence_transformers",
    "sklearn", "beir", "openai", "requests", "tqdm"
]

missing = []
for name in REQUIRED:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append((name, str(exc)))

required_paths = [
    Path("main.py"), Path("main_adaptive.py"), Path("run_defense.py"),
    Path("src/defense.py"), Path("gpt3.5_config.json.template"),
]
missing_paths = [str(p) for p in required_paths if not p.exists()]

print(f"Python: {sys.version.split()[0]}")
print(f"CUDA available: {__import__('torch').cuda.is_available() if not any(n == 'torch' for n, _ in missing) else 'unknown'}")
if missing:
    print("Missing/import-failing packages:")
    for name, error in missing:
        print(f"  - {name}: {error}")
if missing_paths:
    print("Missing repository files:")
    for path in missing_paths:
        print(f"  - {path}")
if missing or missing_paths:
    raise SystemExit(1)
print("TRIS setup check passed.")
