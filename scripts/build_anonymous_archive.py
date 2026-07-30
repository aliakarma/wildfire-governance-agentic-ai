#!/usr/bin/env python3
"""Build an anonymised supplementary code archive for double-blind submission.

Produces ``dist/anonymous_archive/`` plus a matching ``.zip``, containing the
code, configs, and result CSVs a reviewer needs to reproduce the paper -- with
every author-, institution-, and repository-identifying string removed.

The working repository is never modified.

Usage:
    python scripts/build_anonymous_archive.py
    python scripts/build_anonymous_archive.py --check   # verify only, no write
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "dist" / "anonymous_archive"

# Directories never shipped: VCS, envs, caches, vendored deps, agent scratch,
# the manuscript itself (submitted separately), and back-filled result data.
EXCLUDE_DIRS = {
    ".git", ".venv", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".gstack", ".claude", ".cursor", "dist",
    "mlruns", ".ipynb_checkpoints", "Paper",
    # Build artefacts embed absolute paths, which on Windows contain the
    # author's username.
    ".next", ".turbo", "build", ".egg-info",
    # CI config carries a repository slug and is of no use to a reviewer.
    ".github",
}

# This script necessarily contains the names it scrubs, so it excludes itself.
# The rebuttal report is working material, not supplementary material.
EXCLUDE_FILES = {"build_anonymous_archive.py", "Reviewer_Response_Report.md"}

# Files lifted into the archive from otherwise-excluded trees: (source, dest).
# The appendix is the one part of Paper/ a reviewer must have, and it belongs at
# the archive root where it is the first thing seen.
EXTRA_FILES: list[tuple[str, str]] = [
    ("Paper/AAAI/supplementary.pdf", "supplementary.pdf"),
]

# Submission limit is 100 MB; fail well before it so the ceiling is discovered
# here rather than at the upload form.
MAX_ARCHIVE_MB = 95.0

EXCLUDE_PATH_PARTS = {
    # Byte-identical copies of the frozen targets: misleading to a reviewer.
    ("results", "reproduced"),
}

EXCLUDE_SUFFIXES = {
    ".pyc", ".pyo", ".log", ".aux", ".synctex", ".blg",
    # Trained weights are not shipped: ppo_gomdp_best.pt alone is 63 MB, and a
    # torch checkpoint is an internal zip so it barely compresses. Nothing on
    # the headline verification path needs it -- the 354-cell alignment gate,
    # the test suite, and the whole dashboard all run without one. TRAINING.md
    # documents how to reproduce it. The checkpoints/ directory itself is kept
    # (README + .gitkeep) so the training output path exists.
    ".pt", ".ckpt", ".pth",
}

# Identifying strings -> anonymous replacements. Order matters: longest first.
SCRUB: list[tuple[str, str]] = [
    ("Ali Akarma, Toqeer Ali Syed, Salman Jan, Hammad Muneer, Abdul Khadar Jilani",
     "Anonymous Authors"),
    ("Ali Akarma · Toqeer Ali Syed · Salman Jan · Hammad Muneer · Abdul Khadar Jilani",
     "Anonymous Authors"),
    ("443059463@stu.iu.edu.sa", "anonymous@example.com"),
    ("https://akarma-iu.github.io/wildfire-governance-agentic-ai/",
     "https://example.com/anonymous/"),
    ("https://github.com/akarma-iu/wildfire-governance-agentic-ai",
     "https://github.com/anonymous/anonymous-repo"),
    ("https://github.com/aliakarma/wildfire-governance-agentic-ai",
     "https://github.com/anonymous/anonymous-repo"),
    ("https://codecov.io/gh/aliakarma/wildfire-governance-agentic-ai",
     "https://codecov.io/gh/anonymous/anonymous-repo"),
    ("akarma-iu", "anonymous"),
    ("aliakarma", "anonymous"),
    ("Ali Akarma", "Anonymous Author"),
    ("Toqeer Ali Syed", "Anonymous Author"),
    ("Salman Jan", "Anonymous Author"),
    ("Hammad Muneer", "Anonymous Author"),
    ("Abdul Khadar Jilani", "Anonymous Author"),
    ("iu.edu.sa", "example.com"),
]

# Absolute paths leak the OS username. Applied as regexes after SCRUB.
PATH_SCRUB: list[tuple[str, str]] = [
    (r"[A-Za-z]:\\\\Users\\\\[^\\\\\"'\s]+", r"C:\\\\Users\\\\anonymous"),
    (r"[A-Za-z]:\\Users\\[^\\\"'\s]+", r"C:\\Users\\anonymous"),
    (r"/(?:home|Users)/[^/\"'\s]+", "/home/anonymous"),
]

# Residual patterns that must not survive scrubbing (case-insensitive).
FORBIDDEN = [
    r"akarma", r"toqeer", r"salman\s+jan", r"hammad", r"jilani",
    r"iu\.edu\.sa", r"443059463",
]

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh",
    ".json", ".csv", ".rst", ".in", ".bat", ".ps1", "",
}


def is_excluded(path: Path) -> bool:
    parts = path.relative_to(REPO).parts
    if any(p in EXCLUDE_DIRS for p in parts):
        return True
    if path.name in EXCLUDE_FILES:
        return True
    if any(p.endswith(".egg-info") for p in parts):
        return True
    for combo in EXCLUDE_PATH_PARTS:
        if len(parts) >= len(combo) and parts[: len(combo)] == combo:
            return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    return False


def scrub_text(text: str) -> str:
    for needle, replacement in SCRUB:
        text = text.replace(needle, replacement)
    for pattern, replacement in PATH_SCRUB:
        text = re.sub(pattern, replacement, text)
    return text


def scan_bytes(path: Path) -> list[str]:
    """Return FORBIDDEN patterns found in a file's raw bytes.

    Text files are scrubbed and then checked, but binaries (PDF, .npz, images)
    are copied verbatim and would otherwise never be checked at all. A PDF in
    particular can carry an author name in its metadata or an embedded stream.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return []
    return [p for p in FORBIDDEN if re.search(p.encode(), data, re.IGNORECASE)]


def collect_files() -> list[Path]:
    return [
        p for p in REPO.rglob("*")
        if p.is_file() and not is_excluded(p)
    ]


def build(check_only: bool = False) -> int:
    files = collect_files()
    if not check_only:
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        OUT_DIR.mkdir(parents=True)

    n_scrubbed = 0
    violations: list[str] = []

    for src in files:
        rel = src.relative_to(REPO)
        dst = OUT_DIR / rel

        if src.suffix.lower() in TEXT_SUFFIXES:
            try:
                original = src.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                if not check_only:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                continue

            cleaned = scrub_text(original)
            if cleaned != original:
                n_scrubbed += 1

            for pattern in FORBIDDEN:
                if re.search(pattern, cleaned, re.IGNORECASE):
                    violations.append(f"{rel}: matches /{pattern}/")

            if not check_only:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(cleaned, encoding="utf-8")
        else:
            for pattern in scan_bytes(src):
                violations.append(f"{rel}: binary matches /{pattern}/")
            if not check_only:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    # Files lifted out of otherwise-excluded trees.
    n_extra = 0
    for rel_src, rel_dst in EXTRA_FILES:
        src = REPO / rel_src
        if not src.is_file():
            violations.append(f"{rel_src}: required file is missing")
            continue
        for pattern in scan_bytes(src):
            violations.append(f"{rel_src}: binary matches /{pattern}/")
        n_extra += 1
        if not check_only:
            dst = OUT_DIR / rel_dst
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    print(f"files considered : {len(files)}")
    print(f"files scrubbed   : {n_scrubbed}")
    print(f"extra files      : {n_extra} ({', '.join(d for _, d in EXTRA_FILES)})")

    if violations:
        print(f"\nFAIL — {len(violations)} identifying string(s) survived:")
        for v in violations[:25]:
            print(f"  {v}")
        return 1

    print("anonymity check  : PASS (no identifying strings remain)")

    if not check_only:
        archive = shutil.make_archive(str(OUT_DIR), "zip", root_dir=OUT_DIR)
        size_mb = Path(archive).stat().st_size / 1e6
        print(f"\nwrote {archive} ({size_mb:.1f} MB)")

        if size_mb > MAX_ARCHIVE_MB:
            print(
                f"\nFAIL - archive is {size_mb:.1f} MB, over the "
                f"{MAX_ARCHIVE_MB:.0f} MB build limit (submission cap is 100 MB)."
            )
            return 1
        print(f"size check       : PASS ({size_mb:.1f} / {MAX_ARCHIVE_MB:.0f} MB)")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify anonymity without writing the archive")
    args = ap.parse_args()
    sys.exit(build(check_only=args.check))
