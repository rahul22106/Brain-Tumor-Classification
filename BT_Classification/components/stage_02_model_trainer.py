import os
import time
from pathlib import Path
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import dagshub
from sklearn.utils import class_weight
import numpy as np
from BT_Classification import logger
from BT_Classification.entity import TrainingConfig, MLflowConfig


class Training:
    """
    FINAL OPTIMIZED Training - Balanced approach based on best results
    """
    
    def __init__(self, config: TrainingConfig, mlflow_config: MLflowConfig):
        self.config = config
        self.mlflow_config = mlflow_config
    
    def get_base_model(self):
        """Load the prepared base model"""
        try:
            logger.info(f"Loading improved model from: {self.config.updated_base_model_path}")
            
            self.model = tf.keras.models.load_model(
                str(self.config.updated_base_model_path)
            )
            
            logger.info("Improved model loaded successfully")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    def train_valid_generator(self):
        """Create OPTIMIZED data generators"""
        try:
            logger.info("Setting up OPTIMIZED data generators...")
            
            if self.config.params_is_augmentation:
                logger.info("Optimized augmentation enabled - BALANCED approach")
                train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=30,
                    width_shift_range=0.3,
                    height_shift_range=0.3,
                    shear_range=0.3,
                    zoom_range=0.35,
                    horizontal_flip=True,
                    vertical_flip=True,
                    brightness_range=[0.7, 1.3],
                    fill_mode='nearest',
                    validation_split=0.2
                )
            else:
                train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255,
                    validation_split=0.2
                )
            
            # Training generator
            self.train_generator = train_datagen.flow_from_directory(
                directory=str(self.config.training_data),
                target_size=self.config.params_image_size[:-1],
                batch_size=self.config.params_batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True,
                seed=42
            )
            
            # Validation generator (no augmentation)
            self.validation_generator = train_datagen.flow_from_directory(
                directory=str(self.config.training_data),
                target_size=self.config.params_image_size[:-1],
                batch_size=self.config.params_batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False,
                seed=42
            )
            
            logger.info("="*70)
            logger.info("DATASET INFORMATION")
            logger.info("="*70)
            logger.info(f"Training samples: {self.train_generator.samples}")
            logger.info(f"Validation samples: {self.validation_generator.samples}")
            logger.info(f"Number of classes: {self.train_generator.num_classes}")
            logger.info(f"Class indices: {self.train_generator.class_indices}")
            logger.info(f"Batch size: {self.config.params_batch_size}")
            logger.info("="*70)
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save the trained model"""
        try:
            model.save(str(path))
            logger.info(f"Model saved at: {path}")
        except Exception as e:
            logger.exception(e)
            raise e
    
    def train(self):
        """Train with FINAL OPTIMIZED class weights"""
        try:
            logger.info("="*70)
            logger.info("STARTING FINAL OPTIMIZED TRAINING")
            logger.info("="*70)
            
            steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
            validation_steps = self.validation_generator.samples // self.validation_generator.batch_size
            
            logger.info(f"\nTraining Configuration:")
            logger.info(f"  Epochs: {self.config.params_epochs}")
            logger.info(f"  Batch Size: {self.config.params_batch_size}")
            logger.info(f"  Steps per Epoch: {steps_per_epoch}")
            logger.info(f"  Validation Steps: {validation_steps}")
            
        # ==================== FINAL OPTIMIZED CLASS WEIGHTS ====================
            logger.info("\n" + "="*70)
            logger.info("CALCULATING FINAL OPTIMIZED CLASS WEIGHTS")
            logger.info("="*70)

            # Get base balanced weights
            class_weights_array = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(self.train_generator.classes),
                y=self.train_generator.classes
            )

            class_weight_dict = dict(enumerate(class_weights_array))

            # FINAL OPTIMIZED: Based on Run 1 results (which worked best)
            # Analysis: 3x was too weak for Glioma, 6x hurt other classes
            # Optimal: 4.2x Glioma, reduce No Tumor more aggressively
            performance_boost_factors = {
                0: 4.2,  # Glioma: Optimal boost (between 3x and 6x)
                1: 1.0,  # Meningioma: Keep balanced (was perfect at 96%)
                2: 0.45, # No Tumor: Reduce more (was still over-predicting)
                3: 1.6   # Pituitary: Slight increase (86% → target 90%)
            }

            # Apply performance adjustments
            logger.info("Base balanced weights + FINAL OPTIMIZED adjustments:")
            for class_idx, weight in class_weight_dict.items():
                class_name = list(self.train_generator.class_indices.keys())[
                    list(self.train_generator.class_indices.values()).index(class_idx)
                ]
                
                original_weight = weight
                boost_factor = performance_boost_factors.get(class_idx, 1.0)
                adjusted_weight = original_weight * boost_factor
                class_weight_dict[class_idx] = adjusted_weight
                
                logger.info(f"  {class_name} (class {class_idx}): {original_weight:.4f} → {adjusted_weight:.4f} (x{boost_factor})")

            logger.info("="*70)
            logger.info("STRATEGY: Balanced optimization - 4.2x Glioma, control No Tumor")
        # =================================================================
            
            # Initialize MLflow
            logger.info("\n" + "="*70)
            logger.info("MLFLOW: Starting Experiment")
            logger.info("="*70)
            
            dagshub.init(
                repo_owner='rahul22106',
                repo_name='Brain-Tumor-Classification',
                mlflow=True
            )
            
            if self.mlflow_config.tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
            
            mlflow.set_experiment(self.mlflow_config.experiment_name)
            
            run_name = f"{self.mlflow_config.run_name_prefix}_final_optimized_v4_batch{self.config.params_batch_size}"
            
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                logger.info(f"MLflow Run ID: {run_id}")
                logger.info(f"Strategy: FINAL OPTIMIZED - BALANCED PERFORMANCE")
                
                # Log Parameters
                mlflow.log_param("strategy", "final_optimized_balanced_v4")
                mlflow.log_param("model_architecture", "MobileNetV2_Improved")
                mlflow.log_param("batch_size", self.config.params_batch_size)
                mlflow.log_param("epochs", self.config.params_epochs)
                mlflow.log_param("learning_rate", self.config.params_learning_rate)
                mlflow.log_param("dropout", "0.35_0.35_0.3")
                mlflow.log_param("l2_reg", "0.008_0.008_0.005")
                mlflow.log_param("frozen_layers", 100)
                mlflow.log_param("gradient_clipping", 1.0)
                mlflow.log_param("class_weights", str(class_weight_dict))
                mlflow.log_param("augmentation", "optimized_balanced")
                mlflow.log_param("glioma_boost", "4.2x_optimal")
                
                # Set Tags
                for tag_key, tag_value in self.mlflow_config.tags.items():
                    mlflow.set_tag(tag_key, tag_value)
                mlflow.set_tag("training_strategy", "final_optimized_v4")
                mlflow.set_tag("focus", "balanced_all_classes")
                
                # Callbacks
                logger.info("\nSetting up OPTIMIZED callbacks...")

                checkpoint_path = str(self.config.trained_model_path).replace('.keras', '_best.keras')
                checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                )

                # OPTIMIZED: Early stopping to prevent overfitting
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',  
                    patience=10,
                    restore_best_weights=True,
                    mode='max',
                    verbose=1,
                    min_delta=0.001
                )
                logger.info("EarlyStopping: monitor=val_accuracy, patience=10, min_delta=0.001")

                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.3,
                    patience=3,
                    min_lr=1e-7,
                    mode='min',
                    verbose=1
                )
                logger.info("ReduceLROnPlateau: factor=0.3, patience=3")

                csv_logger = tf.keras.callbacks.CSVLogger(
                    filename=str(self.config.root_dir / 'training_log_final_v4.csv'),
                    append=True
                )

                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=str(self.config.root_dir / 'tensorboard_logs_final_v4'),
                    histogram_freq=1
                )
                
                # Balanced performance monitor
                class BalancedMonitor(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        train_acc = logs.get('accuracy', 0)
                        val_acc = logs.get('val_accuracy', 0)
                        gap = train_acc - val_acc
                        
                        if gap > 0.15:
                            logger.warning(f"⚠️ OVERFITTING: Train-Val gap = {gap*100:.1f}%")
                        elif gap < -0.05:
                            logger.warning(f"⚠️ UNDERFITTING: Val > Train by {abs(gap)*100:.1f}%")
                        else:
                            logger.info(f"✓ Good balance")
                        
                        logger.info(f"Epoch {epoch+1}: Train={train_acc*100:.2f}% | Val={val_acc*100:.2f}% | Gap={gap*100:.1f}%")
                        logger.info(f"  Strategy: 4.2x Glioma (optimal), 0.45x No Tumor (controlled)")
                
                balanced_monitor = BalancedMonitor()
                
                # Start Training
                logger.info("\n" + "="*70)
                logger.info("TRAINING STARTED - FINAL OPTIMIZED MODE")
                logger.info("="*70)
                
                start_time = time.time()
                
                history = self.model.fit(
                    self.train_generator,
                    epochs=self.config.params_epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=self.validation_generator,
                    validation_steps=validation_steps,
                    class_weight=class_weight_dict,
                    callbacks=[checkpoint, early_stop, reduce_lr, csv_logger, tensorboard, balanced_monitor],
                    verbose=1
                )
                
                end_time = time.time()
                training_time = end_time - start_time
                
                logger.info("\n" + "="*70)
                logger.info("TRAINING COMPLETED")
                logger.info("="*70)
                logger.info(f"Total training time: {training_time/60:.2f} minutes")
                
                # Log Final Metrics
                final_train_acc = history.history['accuracy'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                final_train_loss = history.history['loss'][-1]
                final_val_loss = history.history['val_loss'][-1]
                
                best_val_acc = max(history.history['val_accuracy'])
                best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
                
                train_val_gap = final_train_acc - final_val_acc
                
                mlflow.log_metric("final_train_accuracy", final_train_acc)
                mlflow.log_metric("final_val_accuracy", final_val_acc)
                mlflow.log_metric("final_train_loss", final_train_loss)
                mlflow.log_metric("final_val_loss", final_val_loss)
                mlflow.log_metric("best_val_accuracy", best_val_acc)
                mlflow.log_metric("train_val_gap", train_val_gap)
                mlflow.log_metric("training_time_minutes", training_time/60)
                
                # Training Summary
                logger.info("\n" + "="*70)
                logger.info("FINAL TRAINING SUMMARY")
                logger.info("="*70)
                logger.info(f"Final Training Accuracy: {final_train_acc*100:.2f}%")
                logger.info(f"Final Validation Accuracy: {final_val_acc*100:.2f}%")
                logger.info(f"Train-Val Gap: {train_val_gap*100:.1f}%")
                logger.info(f"Best Validation Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
                logger.info(f"Class Weights: Glioma=4.2x, Meningioma=1.0x, NoTumor=0.45x, Pituitary=1.6x")
                
                if train_val_gap > 0.1:
                    logger.warning("⚠️ OVERFITTING DETECTED - Consider more regularization")
                elif train_val_gap < -0.05:
                    logger.warning("⚠️ UNDERFITTING - Model can learn more")
                else:
                    logger.info("✓ Good generalization achieved")
                
                # Save Model
                self.save_model(
                    path=self.config.trained_model_path,
                    model=self.model
                )
                
                logger.info(f"\n✓ Best model saved: {checkpoint_path}")
                logger.info(f"✓ Final model saved: {self.config.trained_model_path}")
                
            return history
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    def initiate_model_training(self):
        """Main training execution"""
        try:
            logger.info("="*70)
            logger.info("FINAL OPTIMIZED BALANCED TRAINING PIPELINE")
            logger.info("="*70)
            
            logger.info("\n>>> Step 1: Load Improved Model")
            self.get_base_model()
            
            logger.info("\n>>> Step 2: Setup Optimized Augmentation")
            self.train_valid_generator()
            
            logger.info("\n>>> Step 3: Train with Optimal Weights (4.2x Glioma)")
            history = self.train()
            
            logger.info("\n" + "="*70)
            logger.info("TRAINING COMPLETED - FINAL OPTIMIZED VERSION")
            logger.info("="*70)
            
            return history
            
        except Exception as e:
            logger.exception(e)
            raise e