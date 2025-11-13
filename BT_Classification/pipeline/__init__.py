from BT_Classification.config.configuration import ConfigurationManager
from BT_Classification.components.stage_00_data_ingestion import DataIngestion
from BT_Classification import logger


STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    """
    Pipeline for Data Ingestion Stage
    """
    def __init__(self):
        pass
    
    def main(self):
        """Execute data ingestion pipeline"""
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.initiate_data_ingestion()


if __name__ == '__main__':
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        logger.info(f"{'='*70}\n")
        
        obj = DataIngestionTrainingPipeline()
        obj.main()
        
        logger.info(f"\n{'='*70}")
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
        logger.info(f"{'='*70}\n")
        
    except Exception as e:
        logger.exception(e)
        raise e