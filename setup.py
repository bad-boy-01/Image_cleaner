"""
setup.py

Dependency installer for the AI Image Cleaner.

Run this ONCE at the start of every Kaggle session before running app.py:

    !python Image_cleaner/setup.py

Or from inside the Image_cleaner directory:

    !python setup.py

What it installs
----------------
- simple-lama-inpainting  (LaMa inpainting backend)
- easyocr                 (usually pre-installed on Kaggle, re-checked here)
- opencv-python-headless  (usually pre-installed on Kaggle)
- pydantic / pydantic-settings  (configuration)

All installs use --quiet to keep output clean.
"""

from __future__ import annotations

import subprocess
import sys


PACKAGES = [
    "simple-lama-inpainting",
    "easyocr",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    # opencv-python-headless and numpy are pre-installed on Kaggle
]


def install(package: str) -> bool:
    """Pip-install *package*. Returns True on success."""
    print(f"  Installing {package} ...", end=" ", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", package],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("OK")
        return True
    else:
        print("FAILED")
        print(f"    stderr: {result.stderr.strip()}")
        return False


def main() -> None:
    print("\n=== AI Image Cleaner — dependency setup ===\n")
    failed = []
    for pkg in PACKAGES:
        if not install(pkg):
            failed.append(pkg)

    print()
    if failed:
        print(f"⚠  Some packages failed to install: {failed}")
        print("   Try installing them manually with:")
        for pkg in failed:
            print(f"     pip install {pkg}")
        sys.exit(1)
    else:
        print("✔  All dependencies installed successfully.")
        print("   You can now run:  python app.py --input <path>\n")


if __name__ == "__main__":
    main()
