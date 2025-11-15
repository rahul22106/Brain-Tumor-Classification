import os
import json
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import dagshub
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from BT_Classification import logger
from BT_Classification.entity import EvaluationConfig, MLflowConfig
from BT_Classification.utils.common import save_json


class Evaluation:
    """
    Model Evaluation component with comprehensive metrics and MLflow logging
    """
    
    def __init__(self, config: EvaluationConfig, mlflow_config: MLflowConfig):
        self.config = config
        self.mlflow_config = mlflow_config
        self.model = None
        self.test_generator = None
        self.class_names = None
        
    def load_model(self):
        """Load the trained model"""
        try:
            logger.info(f"Loading trained model from: {self.config.path_of_model}")
            
            self.model = tf.keras.models.load_model(
                str(self.config.path_of_model)
            )
            
            logger.info("Model loaded successfully")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    def load_test_data(self):
        """Load and prepare test data"""
        try:
            logger.info(f"Loading test data from: {self.config.test_data_dir}")
            
            # Create test data generator (no augmentation)
            test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255
            )
            
            self.test_generator = test_datagen.flow_from_directory(
                directory=str(self.config.test_data_dir),
                target_size=self.config.params_image_size[:-1],
                batch_size=self.config.params_batch_size,
                class_mode='categorical',
                shuffle=False,  # Important for evaluation
                seed=42
            )
            
            self.class_names = list(self.test_generator.class_indices.keys())
            
            logger.info("="*70)
            logger.info("TEST DATASET INFORMATION")
            logger.info("="*70)
            logger.info(f"Test samples: {self.test_generator.samples}")
            logger.info(f"Number of classes: {self.test_generator.num_classes}")
            logger.info(f"Class names: {self.class_names}")
            logger.info(f"Class indices: {self.test_generator.class_indices}")
            logger.info(f"Batch size: {self.config.params_batch_size}")
            logger.info("="*70)
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    def evaluate_model(self):
        """Evaluate model and compute metrics"""
        try:
            logger.info("\n" + "="*70)
            logger.info("STARTING MODEL EVALUATION")
            logger.info("="*70)
            
            # Calculate steps - FIXED: Convert to int
            steps = int(np.ceil(self.test_generator.samples / self.test_generator.batch_size))
            
            logger.info(f"Total test steps: {steps}")
            
            # Get predictions
            logger.info("Generating predictions...")
            predictions = self.model.predict(
                self.test_generator,
                steps=steps,
                verbose=1
            )
            
            # Get true labels - FIXED: Ensure correct length
            true_labels = self.test_generator.classes[:len(predictions)]
            predicted_labels = np.argmax(predictions, axis=1)
            
            logger.info(f"Predictions shape: {predictions.shape}")
            logger.info(f"True labels length: {len(true_labels)}")
            logger.info(f"Predicted labels length: {len(predicted_labels)}")
            
            # Calculate basic metrics
            test_loss, test_accuracy = self.model.evaluate(
                self.test_generator,
                steps=steps,
                verbose=1
            )
            
            logger.info("\n" + "="*70)
            logger.info("EVALUATION METRICS")
            logger.info("="*70)
            logger.info(f"Test Loss: {test_loss:.4f}")
            logger.info(f"Test Accuracy: {test_accuracy*100:.2f}%")
            
            # Classification Report
            logger.info("\n" + "="*70)
            logger.info("CLASSIFICATION REPORT")
            logger.info("="*70)
            
            report = classification_report(
                true_labels,
                predicted_labels,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            
            # Print detailed report
            print(classification_report(
                true_labels,
                predicted_labels,
                target_names=self.class_names,
                zero_division=0
            ))
            
            # Confusion Matrix
            logger.info("\n" + "="*70)
            logger.info("CONFUSION MATRIX")
            logger.info("="*70)
            
            cm = confusion_matrix(true_labels, predicted_labels)
            logger.info(f"\n{cm}")
            
            # Calculate per-class accuracy
            per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
            logger.info("\nPer-Class Accuracy:")
            for i, class_name in enumerate(self.class_names):
                logger.info(f"  {class_name}: {per_class_accuracy[i]*100:.2f}%")
            
            # Save confusion matrix plot
            self.plot_confusion_matrix(cm, self.class_names)
            
            # Calculate AUC-ROC (if applicable)
            try:
                # Convert to one-hot encoding for AUC calculation
                n_classes = len(self.class_names)
                true_labels_onehot = np.eye(n_classes)[true_labels]
                
                # Make sure predictions match true_labels length
                predictions_aligned = predictions[:len(true_labels)]
                
                # Calculate AUC for each class
                auc_scores = {}
                for i, class_name in enumerate(self.class_names):
                    auc = roc_auc_score(
                        true_labels_onehot[:, i],
                        predictions_aligned[:, i]
                    )
                    auc_scores[class_name] = auc
                    logger.info(f"AUC-ROC for {class_name}: {auc:.4f}")
                
                # Calculate macro-average AUC
                macro_auc = np.mean(list(auc_scores.values()))
                logger.info(f"\nMacro-Average AUC-ROC: {macro_auc:.4f}")
                
                # Plot ROC curves
                self.plot_roc_curves(true_labels_onehot, predictions_aligned, self.class_names)
                
            except Exception as e:
                logger.warning(f"Could not calculate AUC-ROC: {str(e)}")
                auc_scores = {}
                macro_auc = None
            
            # Prepare metrics dictionary
            metrics = {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "macro_avg_precision": float(report['macro avg']['precision']),
                "macro_avg_recall": float(report['macro avg']['recall']),
                "macro_avg_f1_score": float(report['macro avg']['f1-score']),
                "weighted_avg_precision": float(report['weighted avg']['precision']),
                "weighted_avg_recall": float(report['weighted avg']['recall']),
                "weighted_avg_f1_score": float(report['weighted avg']['f1-score']),
            }
            
            # Add per-class metrics
            for class_name in self.class_names:
                metrics[f"{class_name}_precision"] = float(report[class_name]['precision'])
                metrics[f"{class_name}_recall"] = float(report[class_name]['recall'])
                metrics[f"{class_name}_f1_score"] = float(report[class_name]['f1-score'])
                metrics[f"{class_name}_accuracy"] = float(per_class_accuracy[self.class_names.index(class_name)])
            
            # Add AUC scores if available
            if auc_scores:
                for class_name, auc in auc_scores.items():
                    metrics[f"{class_name}_auc_roc"] = float(auc)
                if macro_auc:
                    metrics["macro_avg_auc_roc"] = float(macro_auc)
            
            logger.info("\n" + "="*70)
            logger.info("EVALUATION COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            
            return metrics, cm, report
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot and save confusion matrix"""
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'}
            )
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            
            # Save plot
            cm_path = Path(self.config.root_dir) / 'confusion_matrix.png'
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved at: {cm_path}")
            
        except Exception as e:
            logger.warning(f"Could not save confusion matrix plot: {str(e)}")
    
    def plot_roc_curves(self, true_labels_onehot, predictions, class_names):
        """Plot and save ROC curves for all classes"""
        try:
            plt.figure(figsize=(12, 8))
            
            for i, class_name in enumerate(class_names):
                fpr, tpr, _ = roc_curve(true_labels_onehot[:, i], predictions[:, i])
                auc = roc_auc_score(true_labels_onehot[:, i], predictions[:, i])
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})', linewidth=2)
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curves - Multi-Class Classification', fontsize=16, fontweight='bold')
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            roc_path = Path(self.config.root_dir) / 'roc_curves.png'
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"ROC curves saved at: {roc_path}")
            
        except Exception as e:
            logger.warning(f"Could not save ROC curves plot: {str(e)}")
    
    def save_evaluation_results(self, metrics):
        """Save evaluation metrics to JSON file"""
        try:
            scores_path = Path(self.config.root_dir) / self.config.metric_file_name
            save_json(path=scores_path, data=metrics)
            logger.info(f"Evaluation metrics saved at: {scores_path}")
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    def log_to_mlflow(self, metrics, cm, report):
        """Log evaluation results to MLflow"""
        try:
            logger.info("\n" + "="*70)
            logger.info("LOGGING TO MLFLOW")
            logger.info("="*70)
            
            # Initialize DagsHub (hardcoded from config.yaml)
            dagshub.init(
                repo_owner='rahul22106',
                repo_name='Brain-Tumor-Classification',
                mlflow=True
            )
            
            if self.mlflow_config.tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
            
            mlflow.set_experiment(self.mlflow_config.experiment_name)
            
            run_name = f"{self.mlflow_config.run_name_prefix}_evaluation"
            
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                logger.info(f"MLflow Run ID: {run_id}")
                
                # Log all metrics
                logger.info("Logging metrics...")
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model parameters
                mlflow.log_param("model_path", str(self.config.path_of_model))
                mlflow.log_param("test_samples", self.test_generator.samples)
                mlflow.log_param("num_classes", len(self.class_names))
                mlflow.log_param("class_names", str(self.class_names))
                
                # Set tags
                for tag_key, tag_value in self.mlflow_config.tags.items():
                    mlflow.set_tag(tag_key, tag_value)
                mlflow.set_tag("stage", "evaluation")
                
                # Log artifacts
                logger.info("Logging artifacts...")
                
                # Log confusion matrix
                cm_path = Path(self.config.root_dir) / 'confusion_matrix.png'
                if cm_path.exists():
                    mlflow.log_artifact(str(cm_path))
                
                # Log ROC curves
                roc_path = Path(self.config.root_dir) / 'roc_curves.png'
                if roc_path.exists():
                    mlflow.log_artifact(str(roc_path))
                
                # Log metrics JSON
                scores_path = Path(self.config.root_dir) / self.config.metric_file_name
                if scores_path.exists():
                    mlflow.log_artifact(str(scores_path))
                
                # Log classification report as JSON
                report_path = Path(self.config.root_dir) / 'classification_report.json'
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=4)
                mlflow.log_artifact(str(report_path))
                
                logger.info(f"✓ All artifacts logged to MLflow")
                logger.info(f"✓ MLflow Run ID: {run_id}")
                
        except Exception as e:
            logger.exception(e)
            raise e
    
    def evaluation(self):
        """Main evaluation pipeline"""
        try:
            logger.info("="*70)
            logger.info("MODEL EVALUATION PIPELINE")
            logger.info("="*70)
            
            logger.info("\n>>> Step 1: Load Trained Model")
            self.load_model()
            
            logger.info("\n>>> Step 2: Load Test Data")
            self.load_test_data()
            
            logger.info("\n>>> Step 3: Evaluate Model")
            metrics, cm, report = self.evaluate_model()
            
            logger.info("\n>>> Step 4: Save Evaluation Results")
            self.save_evaluation_results(metrics)
            
            logger.info("\n>>> Step 5: Log to MLflow")
            self.log_to_mlflow(metrics, cm, report)
            
            logger.info("\n" + "="*70)
            logger.info("EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            
            # Print final summary
            logger.info("\n" + "="*70)
            logger.info("FINAL EVALUATION SUMMARY")
            logger.info("="*70)
            logger.info(f"Test Accuracy: {metrics['test_accuracy']*100:.2f}%")
            logger.info(f"Test Loss: {metrics['test_loss']:.4f}")
            logger.info(f"Macro F1-Score: {metrics['macro_avg_f1_score']:.4f}")
            logger.info(f"Weighted F1-Score: {metrics['weighted_avg_f1_score']:.4f}")
            
            if 'macro_avg_auc_roc' in metrics:
                logger.info(f"Macro AUC-ROC: {metrics['macro_avg_auc_roc']:.4f}")
            
            logger.info("\nPer-Class Performance:")
            for class_name in self.class_names:
                acc = metrics[f"{class_name}_accuracy"]
                f1 = metrics[f"{class_name}_f1_score"]
                logger.info(f"  {class_name}: Accuracy={acc*100:.2f}%, F1-Score={f1:.4f}")
            
            logger.info("="*70)
            
            return metrics
            
        except Exception as e:
            logger.exception(e)
            raise e