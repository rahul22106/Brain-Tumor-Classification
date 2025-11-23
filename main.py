import os
import random
import numpy as np
import tensorflow as tf

# Set fixed seed for reproducibility
SEED = 42

print("\n" + "="*70)
print("REPRODUCIBLE TRAINING MODE")
print("="*70)

# Python random
random.seed(SEED)
print(f"✓ Python random seed: {SEED}")

# Numpy random
np.random.seed(SEED)
print(f"✓ NumPy random seed: {SEED}")

# TensorFlow random
tf.random.set_seed(SEED)
print(f"✓ TensorFlow random seed: {SEED}")

# Python hash seed (for dictionary ordering)
os.environ['PYTHONHASHSEED'] = str(SEED)
print(f"✓ Python hash seed: {SEED}")

# TensorFlow deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'
print("✓ TensorFlow deterministic operations enabled")



print("✓ All random seeds configured")
print("✓ Training will now produce consistent results")
print("="*70 + "\n")

# ============================================================================
# NOW IMPORT YOUR MODULES (AFTER SEEDS ARE SET)
# ============================================================================

from BT_Classification import logger
from BT_Classification.pipeline import DataIngestionTrainingPipeline
from BT_Classification.pipeline import PrepareBaseModelTrainingPipeline
from BT_Classification.pipeline import ModelTrainingPipeline
from BT_Classification.pipeline import ModelEvaluationPipeline

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


STAGE_NAME = "Model Training Stage"

try:
    logger.info(f"\n{'='*70}")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    logger.info(f"{'='*70}\n")
    
    model_training = ModelTrainingPipeline()
    model_training.main()
    
    logger.info(f"\n{'='*70}")
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    logger.info(f"{'='*70}\n\n")
    
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info(f"\n{'='*70}")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    logger.info(f"{'='*70}\n")
    
    model_evaluation = ModelEvaluationPipeline()
    model_evaluation.main()
    
    logger.info(f"\n{'='*70}")
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    logger.info(f"{'='*70}\n\n")
    
except Exception as e:
    logger.exception(e)
    raise e




print("\n" + "="*70)
print("ALL STAGES COMPLETED SUCCESSFULLY")
print("="*70)
print("\nCheck your results:")
print("  📊 Metrics: artifacts/evaluation/scores.json")
print("  📈 Confusion Matrix: artifacts/evaluation/confusion_matrix.png")
print("  📉 ROC Curves: artifacts/evaluation/roc_curves.png")
print("  📝 Training Log: artifacts/training/training_log_final_v4.csv")
print("\nExpected improvements:")
print("  ✓ Glioma recall: 55-65% (was 34%)")
print("  ✓ No Tumor recall: 85-90% (was 99% - overpredicting)")
print("  ✓ Overall accuracy: 77-80%")
print("  ✓ Consistent results across runs")
print("="*70 + "\n")