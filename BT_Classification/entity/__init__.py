from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for Data Ingestion"""
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    train_data_dir: Path
    test_data_dir: Path
