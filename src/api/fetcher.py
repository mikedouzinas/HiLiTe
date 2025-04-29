# src/api/fetcher.py
import os
from pathlib import Path
import yaml
from typing import Optional, List, Dict
import logging
from tqdm import tqdm

# SoccerNet modules
import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Loads configuration from a YAML file.

    Args:
        config_path: Path to the config file

    Returns:
        Dict containing the configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing config file: {e}")

    return config or {}


class SoccerNetFetcher:
    """
    A class to handle downloading datasets from SoccerNet.
    """
    SUPPORTED_DATASETS = {
        "v1": {
            "files": [
                "Labels-v1.json",
                "1_ResNET_TF2_PCA512.npy",
                "1_224p.mkv"
            ]
        },
        "v2": {
            "files": [
                "Labels-v2.json",
                "2_ResNET_TF2_PCA512.npy",
                "2_224p.mkv"
            ]
        }
    }

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the SoccerNetFetcher.

        Args:
            config_path: Path to the config file
        """
        self.config = load_config(config_path)

        # Get configuration with defaults
        self.local_directory = self.config.get(
            "soccernet_local_dir", "data/raw/soccernet")
        self.password = self.config.get("soccernet_password")

        # Ensure that directory exists
        Path(self.local_directory).mkdir(parents=True, exist_ok=True)

        # Create the SoccerNetDownloader object
        self.downloader = SoccerNetDownloader(
            LocalDirectory=self.local_directory
        )

        self.downloader.password = self.password

        # Track download progress
        self.downloaded_tasks = set()

    def get_supported_datasets(self) -> List[str]:
        """
        Get list of supported dataset versions.

        Returns:
            List of supported dataset versions
        """
        return list(self.SUPPORTED_DATASETS.keys())

    def get_dataset_tasks(self, dataset_type: str) -> Dict[str, str]:
        """
        Get available tasks for a specific dataset version.

        Args:
            dataset_type: Version of the dataset (e.g., "v1", "v2")

        Returns:
            Dictionary mapping task names to their identifiers

        Raises:
            ValueError: If dataset_type is not supported
        """
        if dataset_type not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset type '{dataset_type}' not supported. "
                f"Supported types: {self.get_supported_datasets()}"
            )
        return self.SUPPORTED_DATASETS[dataset_type]

    def is_task_downloaded(self, task: str) -> bool:
        """
        Check if a specific task has been downloaded.

        Args:
            task: Task identifier

        Returns:
            True if task is downloaded, False otherwise
        """
        return task in self.downloaded_tasks

    def download_dataset(self, dataset_type: str = "v1", splits: Optional[List[str]] = None) -> None:
        """
        Downloads the specified SoccerNet dataset.

        Args:
        dataset_type: Version of the dataset (e.g., "v1", "v2")
        splits: List of data splits to download. Default is all.
        """
        if dataset_type not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset type '{dataset_type}' not supported. "
                f"Supported types: {self.get_supported_datasets()}"
            )

        files = self.SUPPORTED_DATASETS[dataset_type]["files"]
        if not splits:
            splits = ["train", "valid", "test"]

        logger.info(
            f"Starting download of SoccerNet {dataset_type} into {self.local_directory} ...")
        try:
            self.downloader.downloadGames(files=files, split=splits)
            logger.info("Download completed successfully.")
        except Exception as e:
            logger.error(f"Failed to download files: {e}")
            raise

    def download_custom_files(self, files: List[str], split: List[str], task: Optional[str] = None):
        """
        Download specific files using SoccerNetDownloader.downloadGames.

        Args:
            files: List of file names (e.g., ["1_224p.mkv", "Labels-v2.json"])
            split: List of splits (e.g., ["train", "valid"])
            task: Optional task (used for things like "frames")
        """
        logger.info(
            f"Downloading custom files: {files} for splits: {split} task: {task}")
        try:
            self.downloader.downloadGames(
                files=files,
                split=split,
                task=task
            )
            logger.info("Custom file download completed.")
        except Exception as e:
            logger.error(f"Error downloading custom files: {e}")
            raise


def main():
    """Simple CLI / usage example"""
    try:
        fetcher = SoccerNetFetcher(config_path="config.yaml")

        # Print supported datasets
        print("Supported datasets:", fetcher.get_supported_datasets())

        # Download v1 dataset
        fetcher.download_dataset("v1")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
