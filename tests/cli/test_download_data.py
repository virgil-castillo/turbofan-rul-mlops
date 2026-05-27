"""Tests for the turbofan-download-data command."""
from __future__ import annotations

from pathlib import Path

import pytest

from turbofan.cli import download_data


def test_check_uses_repo_data_raw_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check command defaults to the same raw data path as training config."""
    captured: list[Path] = []

    def fake_check(raw_dir: Path = Path("wrong-default")) -> bool:
        """Capture the raw directory passed by the CLI."""
        captured.append(raw_dir)
        return True

    monkeypatch.setattr(download_data, "check", fake_check)
    monkeypatch.setattr("sys.argv", ["turbofan-download-data", "--check"])

    with pytest.raises(SystemExit) as exc_info:
        download_data.main()

    assert exc_info.value.code == 0
    assert captured == [Path("data/raw")]


def test_kaggle_uses_repo_data_raw_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Kaggle command downloads and verifies the default raw data directory."""
    downloaded: list[Path] = []
    checked: list[Path] = []

    def fake_download_kaggle(raw_dir: Path = Path("wrong-default")) -> None:
        """Capture the download target passed by the CLI."""
        downloaded.append(raw_dir)

    def fake_check(raw_dir: Path = Path("wrong-default")) -> bool:
        """Capture the verification target passed by the CLI."""
        checked.append(raw_dir)
        return True

    monkeypatch.setattr(download_data, "download_kaggle", fake_download_kaggle)
    monkeypatch.setattr(download_data, "check", fake_check)
    monkeypatch.setattr("sys.argv", ["turbofan-download-data", "--kaggle"])

    download_data.main()

    assert downloaded == [Path("data/raw")]
    assert checked == [Path("data/raw")]


def test_raw_dir_argument_is_not_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command keeps a fixed config-aligned raw directory."""
    monkeypatch.setattr(
        "sys.argv",
        ["turbofan-download-data", "--check", "--raw-dir", "custom/raw"],
    )

    with pytest.raises(SystemExit) as exc_info:
        download_data.main()

    assert exc_info.value.code == 2
