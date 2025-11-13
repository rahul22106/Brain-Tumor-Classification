from BT_Classification import logger
from BT_Classification.pipeline import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f"\n{'='*70}")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    logger.info(f"{'='*70}\n")
    
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    
    logger.info(f"\n{'='*70}")
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    logger.info(f"{'='*70}\n")
    
except Exception as e:
    logger.exception(e)
    raise e