from BT_Classification.constants import *
from BT_Classification.utils.common import read_yaml, create_directories
from BT_Classification.entity import (
    DataIngestionConfig,PrepareBaseModelConfig,MLflowConfig,TrainingConfig,EvaluationConfig
)


class ConfigurationManager:
    """
    Configuration Manager to read and manage all configurations
    """
    
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        # Create artifacts root directory
        create_directories([self.config.artifacts_root])
    
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get Data Ingestion configuration
        
        Returns:
            DataIngestionConfig: Configuration object for data ingestion
        """
        config = self.config.data_ingestion
        
        # Create root directory for data ingestion
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
            train_data_dir=Path(config.train_data_dir),
            test_data_dir=Path(config.test_data_dir)
        )
        
        return data_ingestion_config


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Get Prepare Base Model configuration
        
        Returns:
            PrepareBaseModelConfig: Configuration object for base model preparation
        """
        config = self.config.prepare_base_model
        params = self.params

        create_directories([Path(config.root_dir)])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_include_top=params.INCLUDE_TOP,
            params_weights=params.WEIGHTS,
            params_classes=params.CLASSES
        )

        return prepare_base_model_config
    
    
    def get_training_config(self) -> TrainingConfig:
        """
        Get Training configuration
        
        Returns:
            TrainingConfig: Configuration object for model training
        """
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = self.config.data_ingestion.train_data_dir
        
        create_directories([Path(training.root_dir)])
        
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE
        )
        
        return training_config
    
    
    def get_mlflow_config(self) -> MLflowConfig:
        """Get MLflow configuration"""
        config = self.config.mlflow
        
        mlflow_config = MLflowConfig(
            experiment_name=config.experiment_name,
            run_name_prefix=config.run_name_prefix,
            tracking_uri=config.tracking_uri,
            registered_model_name=config.registered_model_name,
            dagshub_repo_owner=config.dagshub_repo_owner, 
            dagshub_repo_name=config.dagshub_repo_name,   
            tags=config.tags
        )
        
        return mlflow_config

    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration"""
        config = self.config.evaluation
        params = self.params
        
        create_directories([config.root_dir])
        
        evaluation_config = EvaluationConfig(
            root_dir=Path(config.root_dir),
            path_of_model=Path(config.path_of_model),
            test_data_dir=Path(config.test_data_dir),
            metric_file_name=config.metric_file_name,
            params_image_size=params.IMAGE_SIZE,
            params_batch_size=params.BATCH_SIZE
        )
        
        return evaluation_config