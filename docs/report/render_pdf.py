#!/usr/bin/env python3
"""Build and render the NotGPU Attention technical report as an A4 PDF."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys


HERE = pathlib.Path(__file__).resolve().parent
HTML = HERE / "index.html"
PDF = HERE / "notgpu-attention-technical-report.pdf"


def find_chrome() -> pathlib.Path:
    """Return a Chrome/Chromium executable, including Playwright installations."""
    candidates: list[pathlib.Path] = []
    if os.environ.get("CHROME_BIN"):
        candidates.append(pathlib.Path(os.environ["CHROME_BIN"]))
    for name in ("google-chrome", "chromium", "chromium-browser"):
        if found := shutil.which(name):
            candidates.append(pathlib.Path(found))
    candidates.extend(
        sorted(
            pathlib.Path.home().glob(
                ".cache/ms-playwright/chromium-*/chrome-linux64/chrome"
            ),
            reverse=True,
        )
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise SystemExit(
        "Chrome or Chromium was not found. Install Chromium or set CHROME_BIN."
    )


def main() -> None:
    subprocess.run([sys.executable, str(HERE / "build_report.py")], check=True)
    chrome = find_chrome()
    subprocess.run(
        [
            str(chrome),
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-background-networking",
            "--allow-file-access-from-files",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF}",
            HTML.as_uri(),
        ],
        check=True,
        timeout=120,
    )
    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()
