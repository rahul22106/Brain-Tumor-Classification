from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
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
    experiment_name: str
    run_name_prefix: str
    tracking_uri: str
    registered_model_name: str
    dagshub_repo_owner: str 
    dagshub_repo_name: str   
    tags: dict

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    path_of_model: Path
    test_data_dir: Path
    metric_file_name: str
    params_image_size: list
    params_batch_size: int