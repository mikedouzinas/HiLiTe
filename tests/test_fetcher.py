# tests/test_fetcher.py
import os
import pytest
from src.api.fetcher import SoccerNetFetcher

# @pytest.mark.skip(reason="Integration test that requires network access.")
def test_soccernet_fetcher(tmp_path):
    """
    Simple test to check if SoccerNetFetcher initializes correctly 
    and can be used to download the dataset.
    """
    config_path = "config.yaml"  # or create a temporary config file
    fetcher = SoccerNetFetcher(config_path=config_path)
    
    # Overwrite local directory to a temp path for testing
    fetcher.local_directory = tmp_path / "soccernet"
    fetcher.downloader.LocalDirectory = str(fetcher.local_directory)

    # Try a small dataset or partial download if supported
    fetcher.download_dataset("v1")
    
    # After download, check that new files or directories exist
    assert (fetcher.local_directory / "Labels-v1").exists()
