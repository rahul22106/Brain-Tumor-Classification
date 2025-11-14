from BT_Classification.constants import *
from BT_Classification.utils.common import read_yaml, create_directories
from BT_Classification.entity import (
    DataIngestionConfig,PrepareBaseModelConfig,MLflowConfig,TrainingConfig
)


class ConfigurationManager:
    """
    Configuration Manager to read and manage all configurations
    """
    
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH
    ):
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
        Get Base Model Preparation configuration
        
        Returns:
            PrepareBaseModelConfig: Configuration object for base model
        """
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])
        
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
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
        """
        Get MLflow configuration
        
        Returns:
            MLflowConfig: Configuration object for MLflow tracking
        """
        mlflow_config_data = self.config.mlflow
        
        mlflow_config = MLflowConfig(
            experiment_name=mlflow_config_data.experiment_name,
            run_name_prefix=mlflow_config_data.run_name_prefix,
            tracking_uri=mlflow_config_data.tracking_uri if mlflow_config_data.tracking_uri else "",
            registered_model_name=mlflow_config_data.registered_model_name,
            tags=dict(mlflow_config_data.tags) if hasattr(mlflow_config_data, 'tags') else {}
        )
        
        return mlflow_config
        