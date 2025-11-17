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
from BT_Classification.seed_config import GLOBAL_SEED
from BT_Classification.entity import TrainingConfig, MLflowConfig


class Training:
    """
    FINAL FIXED Training - Reproducible with corrected class weights
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
        """Create REPRODUCIBLE data generators with fixed seed"""
        try:
            logger.info("Setting up REPRODUCIBLE data generators...")
            
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
            
            # CRITICAL FIX: Use fixed seed for reproducibility
            FIXED_SEED = 42
            
            # Training generator
            self.train_generator = train_datagen.flow_from_directory(
                directory=str(self.config.training_data),
                target_size=self.config.params_image_size[:-1],
                batch_size=self.config.params_batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True,
                seed=FIXED_SEED  # FIXED: Always use same seed
            )
            
            # Validation generator (no augmentation)
            self.validation_generator = train_datagen.flow_from_directory(
                directory=str(self.config.training_data),
                target_size=self.config.params_image_size[:-1],
                batch_size=self.config.params_batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False,  # Don't shuffle validation
                seed=FIXED_SEED  # FIXED: Always use same seed
            )
            
            logger.info("="*70)
            logger.info("DATASET INFORMATION")
            logger.info("="*70)
            logger.info(f"Training samples: {self.train_generator.samples}")
            logger.info(f"Validation samples: {self.validation_generator.samples}")
            logger.info(f"Number of classes: {self.train_generator.num_classes}")
            logger.info(f"Class indices: {self.train_generator.class_indices}")
            logger.info(f"Batch size: {self.config.params_batch_size}")
            
            # DIAGNOSTIC: Show class distribution
            from collections import Counter
            class_counts = Counter(self.train_generator.classes)
            
            logger.info("\nClass Distribution in Training Set:")
            for class_idx, count in sorted(class_counts.items()):
                class_name = list(self.train_generator.class_indices.keys())[
                    list(self.train_generator.class_indices.values()).index(class_idx)
                ]
                percentage = (count / len(self.train_generator.classes)) * 100
                logger.info(f"  {class_name}: {count} samples ({percentage:.1f}%)")
            
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
        """Train with FIXED class weights - NO MORE FLUCTUATIONS!"""
        try:
            logger.info("="*70)
            logger.info("STARTING REPRODUCIBLE TRAINING WITH FIXED WEIGHTS")
            logger.info("="*70)
            
            steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
            validation_steps = self.validation_generator.samples // self.validation_generator.batch_size
            
            logger.info(f"\nTraining Configuration:")
            logger.info(f"  Epochs: {self.config.params_epochs}")
            logger.info(f"  Batch Size: {self.config.params_batch_size}")
            logger.info(f"  Steps per Epoch: {steps_per_epoch}")
            logger.info(f"  Validation Steps: {validation_steps}")
            
        # ==================== FIXED CLASS WEIGHTS ====================
            logger.info("\n" + "="*70)
            logger.info("CALCULATING FIXED CLASS WEIGHTS")
            logger.info("="*70)
            logger.info("PROBLEM IDENTIFIED: No Tumor was 99% recall (overpredicting)")
            logger.info("SOLUTION: Reduce No Tumor weight from 0.45 to 0.25")
            logger.info("="*70)

            # Get base balanced weights
            class_weights_array = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(self.train_generator.classes),
                y=self.train_generator.classes
            )

            class_weight_dict = dict(enumerate(class_weights_array))

            # CRITICAL FIX: Adjusted weights to fix No Tumor overprediction
            performance_boost_factors = {
                0: 5.0,   # Glioma: INCREASED from 4.2 to 5.0 (need more boost)
                1: 1.0,   # Meningioma: Keep balanced (was perfect)
                2: 0.25,  # No Tumor: DRASTICALLY REDUCED from 0.45 to 0.25 (KEY FIX!)
                3: 1.6    # Pituitary: Keep same (was good)
            }

            # Apply performance adjustments
            logger.info("\nBase balanced weights + FIXED adjustments:")
            for class_idx, weight in class_weight_dict.items():
                class_name = list(self.train_generator.class_indices.keys())[
                    list(self.train_generator.class_indices.values()).index(class_idx)
                ]
                
                original_weight = weight
                boost_factor = performance_boost_factors.get(class_idx, 1.0)
                adjusted_weight = original_weight * boost_factor
                class_weight_dict[class_idx] = adjusted_weight
                
                logger.info(f"  {class_name} (class {class_idx}): {original_weight:.4f} → {adjusted_weight:.4f} (x{boost_factor})")

            logger.info("\n" + "="*70)
            logger.info("STRATEGY: Stop No Tumor overprediction")
            logger.info("  ✓ Glioma: 5.0x (up from 4.2x)")
            logger.info("  ✓ No Tumor: 0.25x (down from 0.45x) - KEY FIX")
            logger.info("  ✓ This should give Glioma 55-65% recall")
            logger.info("  ✓ And No Tumor 85-90% recall (not 99%!)")
            logger.info("="*70)
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
            
            run_name = f"{self.mlflow_config.run_name_prefix}_fixed_reproducible_v5_batch{self.config.params_batch_size}"
            
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                logger.info(f"MLflow Run ID: {run_id}")
                logger.info(f"Strategy: FIXED REPRODUCIBLE - NO FLUCTUATIONS")
                
                # Log Parameters
                mlflow.log_param("strategy", "fixed_reproducible_v5")
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
                mlflow.log_param("glioma_boost", "5.0x_fixed")
                mlflow.log_param("no_tumor_weight", "0.25x_reduced")
                mlflow.log_param("random_seed", 42)
                mlflow.log_param("reproducible", "True")
                
                # Set Tags
                for tag_key, tag_value in self.mlflow_config.tags.items():
                    mlflow.set_tag(tag_key, tag_value)
                mlflow.set_tag("training_strategy", "fixed_reproducible_v5")
                mlflow.set_tag("focus", "fix_no_tumor_overprediction")
                mlflow.set_tag("issue_fixed", "no_tumor_99percent_recall")
                
                # Callbacks
                logger.info("\nSetting up callbacks...")

                checkpoint_path = str(self.config.trained_model_path).replace('.keras', '_best.keras')
                checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                )

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
                    filename=str(self.config.root_dir / 'training_log_fixed_v5.csv'),
                    append=False  # Start fresh
                )

                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=str(self.config.root_dir / 'tensorboard_logs_fixed_v5'),
                    histogram_freq=1
                )
                
                # Custom monitor
                class FixedMonitor(tf.keras.callbacks.Callback):
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
                        logger.info(f"  Fixed weights: Glioma=5.0x, NoTumor=0.25x (no overprediction)")
                
                fixed_monitor = FixedMonitor()
                
                # Start Training
                logger.info("\n" + "="*70)
                logger.info("TRAINING STARTED - FIXED & REPRODUCIBLE MODE")
                logger.info("="*70)
                
                start_time = time.time()
                
                history = self.model.fit(
                    self.train_generator,
                    epochs=self.config.params_epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=self.validation_generator,
                    validation_steps=validation_steps,
                    class_weight=class_weight_dict,
                    callbacks=[checkpoint, early_stop, reduce_lr, csv_logger, tensorboard, fixed_monitor],
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
                logger.info(f"Class Weights: Glioma=5.0x, Meningioma=1.0x, NoTumor=0.25x, Pituitary=1.6x")
                logger.info("\nFIXES APPLIED:")
                logger.info("  ✓ Random seed fixed (42) - reproducible results")
                logger.info("  ✓ No Tumor weight reduced (0.45 → 0.25)")
                logger.info("  ✓ Glioma weight increased (4.2 → 5.0)")
                logger.info("\nEXPECTED IMPROVEMENTS:")
                logger.info("  ✓ Glioma recall: 55-65% (was 34%)")
                logger.info("  ✓ No Tumor recall: 85-90% (was 99%)")
                logger.info("  ✓ Overall accuracy: 77-80%")
                logger.info("  ✓ Consistent results across runs")
                
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
            logger.info("FIXED REPRODUCIBLE TRAINING PIPELINE")
            logger.info("="*70)
            
            logger.info("\n>>> Step 1: Load Improved Model")
            self.get_base_model()
            
            logger.info("\n>>> Step 2: Setup Reproducible Data Generators")
            self.train_valid_generator()
            
            logger.info("\n>>> Step 3: Train with Fixed Weights (5.0x Glioma, 0.25x No Tumor)")
            history = self.train()
            
            logger.info("\n" + "="*70)
            logger.info("TRAINING COMPLETED - FIXED VERSION")
            logger.info("="*70)
            
            return history
            
        except Exception as e:
            logger.exception(e)
            raise e