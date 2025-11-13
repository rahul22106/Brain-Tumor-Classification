from BT_Classification import logger
from BT_Classification.pipeline import DataIngestionTrainingPipeline
from BT_Classification.pipeline import PrepareBaseModelTrainingPipeline


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


STAGE_NAME = "Prepare Base Model Stage"
try:
    logger.info(f"\n{'='*70}")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    logger.info(f"{'='*70}\n")
    
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    
    logger.info(f"\n{'='*70}")
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    logger.info(f"{'='*70}\n\n")
    
except Exception as e:
    logger.exception(e)
    raise e