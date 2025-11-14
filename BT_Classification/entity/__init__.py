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

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """Configuration for Base Model Preparation"""
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


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


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """Configuration for Base Model Preparation"""
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for Model Training"""
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_learning_rate: float 

@dataclass(frozen=True)
class MLflowConfig:
    """Configuration for MLflow Tracking"""
    experiment_name: str
    run_name_prefix: str
    tracking_uri: str
    registered_model_name: str
    tags: dict     