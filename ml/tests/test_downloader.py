"""Unit tests for src.downloader.DatasetDownloader."""

from src.config import Config
from src.downloader import DatasetDownloader


def test_download_category_skips_existing_file(tmp_path, caplog):
    config = Config(ml_root=tmp_path, categories=["cat"])
    config.data_raw_dir.mkdir(parents=True)
    (config.data_raw_dir / "cat.npy").write_bytes(b"fake-data")

    downloader = DatasetDownloader(config)
    with caplog.at_level("INFO"):
        downloader.download_category("cat")

    assert "already downloaded" in caplog.text
