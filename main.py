# main.py
from venv import logger
from src.api.fetcher import SoccerNetFetcher

def main():
    try:
        fetcher = SoccerNetFetcher(config_path="config.yaml")
        
        # Show available datasets
        print("Supported datasets:", fetcher.get_supported_datasets())
        
        # 1. Download dense captions (text captions for training NLP model)
        fetcher.downloader.downloadDataTask(task="caption-2023", split=["train", "valid", "test", "challenge"])

        # 2. Download action spotting labels
        fetcher.download_custom_files(
            files=["Labels-v2.json"],
            split=["train", "valid", "test"]
        )

        # 3. Download light video features for fast prototyping
        fetcher.download_custom_files(
            files=["1_ResNET_TF2_PCA512.npy"],
            split=["train", "valid"]
        )

        # 4. (Later) Download full videos — only after you receive your NDA password
        fetcher.downloader.downloadGames(
            files=["1_224p.mkv"],
            split=["train", "valid"],
            task="frames"
        )

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
