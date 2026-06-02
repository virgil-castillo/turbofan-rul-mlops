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
import sys
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


def download_kaggle(raw_dir: Path = RAW_DIR) -> None:
    """Download the dataset via the Kaggle API.

    Prints manual instructions and exits with code 1 if the Kaggle API
    key is not configured.

    Args:
        raw_dir: Directory to download files into.
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.warning("Kaggle API key not found at ~/.kaggle/kaggle.json")
        logger.warning("Manual download instructions:")
        logger.warning("  1. Go to %s", MANUAL_URL)
        logger.warning("  2. Click 'Download' to get the dataset zip")
        logger.warning("  3. Extract all .txt files into: %s/", raw_dir)
        sys.exit(1)

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
        sys.exit(1)

    # Flatten directory structure if files extracted to subdirectory
    cmaps_dir = raw_dir / "CMaps"
    if cmaps_dir.exists():
        for txt_file in cmaps_dir.glob("*.txt"):
            shutil.move(str(txt_file), str(raw_dir / txt_file.name))
        shutil.rmtree(cmaps_dir)

    logger.info("Download complete.")


def main() -> None:
    """Entry point for the download script."""
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
    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.check:
        logger.info("Checking data files in %s ...", RAW_DIR)
        all_present = check(RAW_DIR)
        sys.exit(0 if all_present else 1)

    if args.kaggle:
        download_kaggle(RAW_DIR)
        logger.info("Verifying downloaded files ...")
        check(RAW_DIR)


if __name__ == "__main__":
    main()
