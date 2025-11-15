from BT_Classification.config.configuration import ConfigurationManager
from BT_Classification.components.stage_00_data_ingestion import DataIngestion
from BT_Classification.components.stage_01_prepare_base_model import PrepareBaseModel
from BT_Classification.components.stage_02_model_trainer import Training
from BT_Classification.components.stage_03_model_evaluation import Evaluation
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


STAGE_NAME = "Prepare Base Model Stage"


class PrepareBaseModelTrainingPipeline:
    """
    Pipeline for Base Model Preparation Stage
    """
    def __init__(self):
        pass
    
    def main(self):
        """Execute base model preparation pipeline"""
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.initiate_base_model_preparation()


if __name__ == '__main__':
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        logger.info(f"{'='*70}\n")
        
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        
        logger.info(f"\n{'='*70}")
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
        logger.info(f"{'='*70}\n")
        
    except Exception as e:
        logger.exception(e)
        raise e    
    
STAGE_NAME = "Model Training Stage"


class ModelTrainingPipeline:
    """
    Pipeline for Model Training Stage
    """
    def __init__(self):
        pass
    
    def main(self):
        """Execute model training pipeline"""
        config = ConfigurationManager()
        training_config = config.get_training_config()
        mlflow_config = config.get_mlflow_config()
        training = Training(config=training_config, mlflow_config=mlflow_config)
        training.initiate_model_training()


if __name__ == '__main__':
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        logger.info(f"{'='*70}\n")
        
        obj = ModelTrainingPipeline()
        obj.main()
        
        logger.info(f"\n{'='*70}")
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
        logger.info(f"{'='*70}\n")
        
    except Exception as e:
        logger.exception(e)
        raise e    
    
STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    """
    Pipeline for Model Evaluation Stage
    """
    def __init__(self):
        pass
    
    def main(self):
        """Execute model evaluation pipeline"""
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        mlflow_config = config.get_mlflow_config()
        evaluation = Evaluation(config=evaluation_config, mlflow_config=mlflow_config)
        evaluation.evaluation()


if __name__ == '__main__':
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        logger.info(f"{'='*70}\n")
        
        obj = ModelEvaluationPipeline()
        obj.main()
        
        logger.info(f"\n{'='*70}")
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
        logger.info(f"{'='*70}\n")
        
    except Exception as e:
        logger.exception(e)
        raise e    