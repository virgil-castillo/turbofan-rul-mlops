"""Download or verify the NASA C-MAPSS turbofan dataset.

Usage:
    turbofan-download-data --kaggle   # download via Kaggle API
    turbofan-download-data --check    # verify files are present
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from turbofan.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

EXPECTED_FILES: list[str] = [
    f"{split}_FD00{i}.txt"
    for split in ("train", "test", "RUL")
    for i in range(1, 5)
]

RAW_DIR: Path = Path("data/raw")
KAGGLE_DATASET: str = "behrad3d/nasa-cmaps"
MANUAL_URL: str = "https://www.kaggle.com/datasets/behrad3d/nasa-cmaps"


def check(raw_dir: Path = RAW_DIR) -> bool:
    """Verify all expected data files are present in raw_dir.

    Args:
        raw_dir: Directory to check.

    Returns:
        True if all 12 expected files are present, False otherwise.
    """
    all_present = True
    for fname in EXPECTED_FILES:
        path = raw_dir / fname
        if path.exists():
            size_kb = path.stat().st_size // 1024
            logger.info("[OK] %s (%d KB)", fname, size_kb)
        else:
            logger.info("[MISSING] %s", fname)
            all_present = False
    return all_present


def download_kaggle(raw_dir: Path = RAW_DIR) -> bool:
    """Download the dataset via the Kaggle API.

    Args:
        raw_dir: Directory to download files into.

    Returns:
        True when the download command succeeds, False otherwise.
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.warning("Kaggle API key not found at ~/.kaggle/kaggle.json")
        logger.warning("Manual download instructions:")
        logger.warning("  1. Go to %s", MANUAL_URL)
        logger.warning("  2. Click 'Download' to get the dataset zip")
        logger.warning("  3. Extract all .txt files into: %s/", raw_dir)
        return False

    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s into %s ...", KAGGLE_DATASET, raw_dir)
    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "--unzip",
            "-p",
            str(raw_dir),
        ],
        check=False,
    )
    if result.returncode != 0:
        logger.error(
            "Download failed. Check your Kaggle API credentials and connection."
        )
        return False

    # Flatten directory structure if files extracted to subdirectory
    cmaps_dir = raw_dir / "CMaps"
    if cmaps_dir.exists():
        for txt_file in cmaps_dir.glob("*.txt"):
            shutil.move(str(txt_file), str(raw_dir / txt_file.name))
        shutil.rmtree(cmaps_dir)

    logger.info("Download complete.")
    return True


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the download CLI.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Manage the NASA C-MAPSS turbofan dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--kaggle",
        action="store_true",
        help="Download dataset via Kaggle API",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Verify all expected files are present",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging verbosity (falls back to the LOG_LEVEL env var or INFO).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the download script.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    if args.check:
        logger.info("Checking data files in %s ...", RAW_DIR)
        all_present = check(RAW_DIR)
        return 0 if all_present else 1

    if args.kaggle:
        if not download_kaggle(RAW_DIR):
            return 1
        logger.info("Verifying downloaded files ...")
        return 0 if check(RAW_DIR) else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
