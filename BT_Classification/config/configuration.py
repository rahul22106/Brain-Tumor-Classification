from BT_Classification.constants import *
from BT_Classification.utils.common import read_yaml, create_directories
from BT_Classification.entity import (
    DataIngestionConfig
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