import os
import time
from pathlib import Path
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import dagshub
from BT_Classification import logger
from BT_Classification.entity import TrainingConfig, MLflowConfig


class Training:
    """
    Component for training the brain tumor classification model with MLflow tracking
    """
    
    def __init__(self, config: TrainingConfig, mlflow_config: MLflowConfig):
        """
        Initialize Training component
        
        Args:
            config (TrainingConfig): Configuration for model training
            mlflow_config (MLflowConfig): Configuration for MLflow tracking
        """
        self.config = config
        self.mlflow_config = mlflow_config
    
    
    def get_base_model(self):
        """
        Load the prepared base model with custom head
        """
        try:
            logger.info(f"Loading base model from: {self.config.updated_base_model_path}")
            
            self.model = tf.keras.models.load_model(
                str(self.config.updated_base_model_path)
            )
            
            logger.info("✓ Model loaded successfully")
            logger.info(f"✓ Model input shape: {self.model.input_shape}")
            logger.info(f"✓ Model output shape: {self.model.output_shape}")
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    
    def train_valid_generator(self):
        """
        Create data generators for training and validation
        """
        try:
            logger.info("Setting up data generators...")
            
            # Data augmentation configuration for training
            if self.config.params_is_augmentation:
                logger.info("✓ Data augmentation enabled")
                train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.3,
                    horizontal_flip=True,
                    vertical_flip=True,
                    brightness_range=[0.8,1.2],
                    fill_mode='nearest',
                    validation_split=0.2  # 20% for validation
                )
            else:
                logger.info("✓ No data augmentation")
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
            
            # Log dataset information
            logger.info("="*70)
            logger.info("DATASET INFORMATION")
            logger.info("="*70)
            logger.info(f"✓ Training samples: {self.train_generator.samples}")
            logger.info(f"✓ Validation samples: {self.validation_generator.samples}")
            logger.info(f"✓ Number of classes: {self.train_generator.num_classes}")
            logger.info(f"✓ Class indices: {self.train_generator.class_indices}")
            logger.info(f"✓ Batch size: {self.config.params_batch_size}")
            logger.info(f"✓ Steps per epoch: {self.train_generator.samples // self.config.params_batch_size}")
            logger.info(f"✓ Validation steps: {self.validation_generator.samples // self.config.params_batch_size}")
            logger.info("="*70)
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the trained model
        
        Args:
            path (Path): Path to save the model
            model (tf.keras.Model): Trained model
        """
        try:
            model.save(str(path))
            logger.info(f"✓ Model saved at: {path}")
        except Exception as e:
            logger.exception(e)
            raise e
    
    
    def train(self):
        """
        Train the model with MLflow tracking
        """
        try:
            logger.info("="*70)
            logger.info("STARTING MODEL TRAINING WITH MLFLOW")
            logger.info("="*70)
            
            # Calculate steps
            steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
            validation_steps = self.validation_generator.samples // self.validation_generator.batch_size
            
            logger.info(f"\nTraining Configuration:")
            logger.info(f"  → Epochs: {self.config.params_epochs}")
            logger.info(f"  → Batch Size: {self.config.params_batch_size}")
            logger.info(f"  → Steps per Epoch: {steps_per_epoch}")
            logger.info(f"  → Validation Steps: {validation_steps}")
            logger.info(f"  → Training Samples: {self.train_generator.samples}")
            logger.info(f"  → Validation Samples: {self.validation_generator.samples}")
            
            # ============================================================
            # MLFLOW: Start Run with DagsHub
            # ============================================================
            logger.info("\n" + "="*70)
            logger.info("MLFLOW: Starting Experiment Tracking with DagsHub")
            logger.info("="*70)
            
            # Initialize DagsHub
            dagshub.init(
                repo_owner='rahul22106',
                repo_name='Brain-Tumor-Classification',
                mlflow=True
            )
            logger.info("✓ DagsHub initialized successfully")
            
            # Set MLflow tracking URI (if provided)
            if self.mlflow_config.tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
                logger.info(f"✓ MLflow Tracking URI: {self.mlflow_config.tracking_uri}")
            else:
                logger.info("✓ MLflow Tracking: Local (mlruns folder)")
            
            # Set experiment name
            mlflow.set_experiment(self.mlflow_config.experiment_name)
            
            # Generate run name
            run_name = f"{self.mlflow_config.run_name_prefix}_batch{self.config.params_batch_size}_epoch{self.config.params_epochs}"
            
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                logger.info(f"✓ MLflow Run ID: {run_id}")
                logger.info(f"✓ Run Name: {run_name}")
                logger.info(f"✓ Experiment: {self.mlflow_config.experiment_name}")
                logger.info(f"✓ View on DagsHub: https://dagshub.com/rahul22106/Brain-Tumor-Classification/experiments")
                
                # ============================================================
                # MLFLOW: Log Parameters
                # ============================================================
                logger.info("\nLogging Parameters to MLflow...")

                mlflow.log_param("model_architecture", "MobileNetV2")
                mlflow.log_param("image_size", f"{self.config.params_image_size[0]}x{self.config.params_image_size[1]}")
                mlflow.log_param("batch_size", self.config.params_batch_size)
                mlflow.log_param("epochs", self.config.params_epochs)

                # ✅ FIX: Use config learning rate instead of model.optimizer
                mlflow.log_param("learning_rate", self.config.params_learning_rate)

                mlflow.log_param("augmentation", str(self.config.params_is_augmentation))
                mlflow.log_param("optimizer", "Adam")
                mlflow.log_param("loss_function", "categorical_crossentropy")
                mlflow.log_param("train_samples", int(self.train_generator.samples))
                mlflow.log_param("val_samples", int(self.validation_generator.samples))
                mlflow.log_param("num_classes", int(self.train_generator.num_classes))
                # Convert list to comma-separated string
                mlflow.log_param("class_names", ",".join(self.train_generator.class_indices.keys()))

                logger.info("✓ Parameters logged to MLflow")
                                
                # ============================================================
                # MLFLOW: Set Tags
                # ============================================================
                # Set tags from config
                for tag_key, tag_value in self.mlflow_config.tags.items():
                                    mlflow.set_tag(tag_key, tag_value)
                                
                logger.info("✓ Tags set in MLflow")
                                
                # ============================================================
                # Callbacks Setup
                # ============================================================
                logger.info("\nSetting up enhanced callbacks...")

                # Model checkpoint - save best model
                checkpoint_path = str(self.config.trained_model_path).replace('.keras', '_best.keras')
                checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_accuracy',  # Track accuracy
                    save_best_only=True,
                    mode='max',
                    verbose=1
                )
                logger.info(f"✓ ModelCheckpoint: {checkpoint_path}")

                # Early stopping with more patience
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',  # Monitor accuracy instead of loss
                    patience=8,            # Increased patience
                    restore_best_weights=True,
                    mode='max',
                    verbose=1
                )
                logger.info("✓ EarlyStopping: patience=12 (monitoring val_accuracy)")

                # Reduce learning rate on plateau with better settings
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy',  # Monitor accuracy
                    factor=0.5,             # Less aggressive reduction
                    patience=5,             # More patience
                    min_lr=1e-7,
                    mode='max',
                    verbose=1
                )
                logger.info("✓ ReduceLROnPlateau: factor=0.5, patience=5")

                # CSV Logger
                csv_logger = tf.keras.callbacks.CSVLogger(
                    filename=str(self.config.root_dir / 'training_log.csv'),
                    append=True
                )
                logger.info(f"✓ CSVLogger: {self.config.root_dir / 'training_log.csv'}")

                # TensorBoard
                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=str(self.config.root_dir / 'tensorboard_logs'),
                    histogram_freq=1,
                    update_freq='epoch'
                )
                logger.info(f"✓ TensorBoard: {self.config.root_dir / 'tensorboard_logs'}")
                
                # ============================================================
                # Start Training
                # ============================================================
                logger.info("\n" + "="*70)
                logger.info("TRAINING STARTED")
                logger.info("="*70)
                
                start_time = time.time()
                
                history = self.model.fit(
                    self.train_generator,
                    epochs=self.config.params_epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=self.validation_generator,
                    validation_steps=validation_steps,
                    callbacks=[checkpoint, early_stop, reduce_lr, csv_logger, tensorboard],
                    verbose=1
                )
                
                end_time = time.time()
                training_time = end_time - start_time
                
                logger.info("\n" + "="*70)
                logger.info("TRAINING COMPLETED")
                logger.info("="*70)
                logger.info(f"✓ Total training time: {training_time/60:.2f} minutes")
                
                # ============================================================
                # MLFLOW: Log Final Metrics
                # ============================================================
                logger.info("\n" + "="*70)
                logger.info("LOGGING FINAL METRICS TO MLFLOW")
                logger.info("="*70)
                
                final_train_acc = history.history['accuracy'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                final_train_loss = history.history['loss'][-1]
                final_val_loss = history.history['val_loss'][-1]
                
                best_val_acc = max(history.history['val_accuracy'])
                best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
                
                # Log final metrics - use simple metric logging without steps
                mlflow.log_metric("final_train_accuracy", final_train_acc)
                mlflow.log_metric("final_val_accuracy", final_val_acc)
                mlflow.log_metric("final_train_loss", final_train_loss)
                mlflow.log_metric("final_val_loss", final_val_loss)
                mlflow.log_metric("best_val_accuracy", best_val_acc)
                mlflow.log_metric("training_time_minutes", training_time/60)
                
                logger.info("✓ Final metrics logged to MLflow")
                
                # ============================================================
                # Training Summary
                # ============================================================
                logger.info("\n" + "="*70)
                logger.info("TRAINING SUMMARY")
                logger.info("="*70)
                
                logger.info(f"Final Training Accuracy: {final_train_acc*100:.2f}%")
                logger.info(f"Final Validation Accuracy: {final_val_acc*100:.2f}%")
                logger.info(f"Final Training Loss: {final_train_loss:.4f}")
                logger.info(f"Final Validation Loss: {final_val_loss:.4f}")
                logger.info(f"\nBest Validation Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
                
                # ============================================================
                # Save Model
                # ============================================================
                logger.info("\n" + "="*70)
                logger.info("SAVING MODEL")
                logger.info("="*70)
                
                self.save_model(
                    path=self.config.trained_model_path,
                    model=self.model
                )
                logger.info(f"✓ Final model saved")
                
                # ============================================================
                # MLFLOW: Log Model Artifacts (Optional - DagsHub might not support all operations)
                # ============================================================
                logger.info("\n" + "="*70)
                logger.info("LOGGING ARTIFACTS TO MLFLOW")
                logger.info("="*70)
                
                try:
                    # Log the model (simplified approach)
                    mlflow.tensorflow.log_model(
                        self.model,
                        artifact_path="model"
                    )
                    logger.info("✓ Model logged to MLflow")
                except Exception as e:
                    logger.warning(f"Could not log model to MLflow: {e}")
                    logger.info("Continuing without model logging...")
                
                try:
                    # Log training history CSV
                    mlflow.log_artifact(str(self.config.root_dir / 'training_log.csv'))
                    logger.info("✓ Training log CSV logged to MLflow")
                except Exception as e:
                    logger.warning(f"Could not log training log: {e}")
                
                # ============================================================
                # MLFLOW: End Run
                # ============================================================
                logger.info("\n" + "="*70)
                logger.info("MLFLOW: Experiment Tracking Completed")
                logger.info("="*70)
                logger.info(f"✓ Run ID: {run_id}")
                logger.info(f"✓ View on DagsHub: https://dagshub.com/rahul22106/Brain-Tumor-Classification/experiments")
                logger.info(f"✓ Or MLflow UI: mlflow ui → http://localhost:5000")
                
            return history
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    
    def initiate_model_training(self):
        """
        Main method to execute model training with MLflow
        """
        try:
            logger.info("="*70)
            logger.info("INITIALIZING MODEL TRAINING WITH MLFLOW")
            logger.info("="*70)
            
            # Step 1: Load base model
            logger.info("\n>>> Step 1: Load Base Model")
            self.get_base_model()
            
            # Step 2: Setup data generators
            logger.info("\n>>> Step 2: Setup Data Generators")
            self.train_valid_generator()
            
            # Step 3: Train model with MLflow tracking
            logger.info("\n>>> Step 3: Train Model with MLflow Tracking")
            history = self.train()
            
            logger.info("\n" + "="*70)
            logger.info("✓ MODEL TRAINING COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            
            logger.info("\nNext Steps:")
            logger.info("  1. View MLflow UI: mlflow ui")
            logger.info("  2. Check training logs: artifacts/training/training_log.csv")
            logger.info("  3. View TensorBoard: tensorboard --logdir artifacts/training/tensorboard_logs")
            logger.info("  4. Evaluate model on test set (Stage 04)")
            logger.info("  5. Best model: " + str(self.config.trained_model_path).replace('.keras', '_best.keras'))
            
            return history
            
        except Exception as e:
            logger.exception(e)
            raise e