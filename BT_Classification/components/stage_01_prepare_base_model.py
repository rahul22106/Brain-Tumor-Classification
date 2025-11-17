import os
from pathlib import Path
import tensorflow as tf
from BT_Classification.seed_config import GLOBAL_SEED
from BT_Classification import logger
from BT_Classification.entity import PrepareBaseModelConfig


class PrepareBaseModel:
    """
    Component for preparing base model (MobileNetV2) with custom classification head
    OPTIMIZED FOR BETTER PERFORMANCE - BALANCED APPROACH
    """
    
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        """Download and save MobileNetV2 base model with ImageNet weights"""
        try:
            logger.info("Loading MobileNetV2 base model...")
            
            self.model = tf.keras.applications.MobileNetV2(
                input_shape=self.config.params_image_size,
                include_top=self.config.params_include_top,
                weights=self.config.params_weights
            )
            
            self.model.save(str(self.config.base_model_path))
            logger.info(f"Base model saved at: {self.config.base_model_path}")
            logger.info(f"Model: MobileNetV2")
            logger.info(f"Input shape: {self.config.params_image_size}")
            logger.info(f"Total layers: {len(self.model.layers)}")
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        IMPROVED: Better balance between regularization and learning capacity
        Key changes:
        - Reduced frozen layers (100 instead of 130) for more trainable parameters
        - Moderate dropout (0.35, 0.35, 0.3) instead of aggressive (0.45, 0.45, 0.4)
        - Lighter L2 regularization (0.008) for better learning
        - Higher learning rate (0.0005) for faster convergence
        """
        try:
            # Strategy: Unfreeze more layers for better learning
            if freeze_all:
                logger.info("Freezing all base model layers...")
                for layer in model.layers:
                    layer.trainable = False
            elif freeze_till is not None and freeze_till > 0:
                logger.info(f"Freezing first {freeze_till} layers...")
                for layer in model.layers[:freeze_till]:
                    layer.trainable = False
            
            trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            non_trainable_count = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
            
            logger.info(f"Trainable parameters: {trainable_count:,}")
            logger.info(f"Non-trainable parameters: {non_trainable_count:,}")
            logger.info("Adding IMPROVED custom classification head...")

            # Global Average Pooling
            x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(model.output)
            
            # IMPROVED: Moderate dropout (reduced from 0.45 to 0.35)
            x = tf.keras.layers.Dropout(0.35, name='dropout_gap')(x)

            # First Dense - with moderate regularization
            x = tf.keras.layers.Dense(
                units=256,
                kernel_regularizer=tf.keras.regularizers.l2(0.008),  # Reduced from 0.01
                bias_regularizer=tf.keras.regularizers.l2(0.008),
                name='dense_1'
            )(x)
            x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
            x = tf.keras.layers.Activation('relu', name='relu_1')(x)
            x = tf.keras.layers.Dropout(0.35, name='dropout_1')(x)  # Reduced from 0.45

            # Second Dense - lighter regularization
            x = tf.keras.layers.Dense(
                units=128,
                kernel_regularizer=tf.keras.regularizers.l2(0.008),
                bias_regularizer=tf.keras.regularizers.l2(0.008),
                name='dense_2'
            )(x)
            x = tf.keras.layers.BatchNormalization(name='bn_2')(x)
            x = tf.keras.layers.Activation('relu', name='relu_2')(x)
            x = tf.keras.layers.Dropout(0.3, name='dropout_2')(x)  # Reduced from 0.4

            # Output layer with light regularization
            prediction = tf.keras.layers.Dense(
                units=classes,
                activation='softmax',
                kernel_regularizer=tf.keras.regularizers.l2(0.005),  # Reduced from 0.01
                name='output_layer'
            )(x)

            # Create full model
            full_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=prediction
            )
            
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipnorm=1.0
            )
            
            full_model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Model compiled with IMPROVED BALANCED configuration")
            logger.info(f"Learning rate: {learning_rate}")
            logger.info(f"Dropout rates: 0.35, 0.35, 0.3 (reduced for more learning)")
            logger.info(f"L2 regularization: 0.008, 0.008, 0.005 (lighter)")
            logger.info(f"Gradient clipping: 1.0")
            
            return full_model
            
        except Exception as e:
            logger.exception(f"Error in model preparation: {str(e)}")
            raise e
    
    def update_base_model(self):
        """Update base model with IMPROVED BALANCED custom head"""
        try:
            logger.info("Creating IMPROVED BALANCED model...")
            
            # CRITICAL IMPROVEMENT: Freeze fewer layers (100 instead of 130)
            # This allows 30 more layers to be trainable, giving model more capacity
            self.full_model = self._prepare_full_model(
                model=self.model,
                classes=self.config.params_classes,
                freeze_all=False,
                freeze_till=100,  # CHANGED: Reduced from 130
                learning_rate= 0.0005  
            )
            
            trainable_count = sum([tf.keras.backend.count_params(w) for w in self.full_model.trainable_weights])
            non_trainable_count = sum([tf.keras.backend.count_params(w) for w in self.full_model.non_trainable_weights])
            
            logger.info(f"Final model - Trainable: {trainable_count:,}")
            logger.info(f"Final model - Non-trainable: {non_trainable_count:,}")
            
            self.full_model.save(str(self.config.updated_base_model_path))
            logger.info(f"Improved model saved at: {self.config.updated_base_model_path}")
            
            logger.info("\n" + "="*70)
            logger.info("IMPROVED BALANCED MODEL ARCHITECTURE")
            logger.info("="*70)
            self.full_model.summary(print_fn=logger.info)
            logger.info("="*70)
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    def initiate_base_model_preparation(self):
        """Main method to execute base model preparation"""
        try:
            logger.info("="*70)
            logger.info("STARTING IMPROVED BALANCED BASE MODEL PREPARATION")
            logger.info("="*70)
            
            logger.info("\n>>> Step 1: Load Base Model (MobileNetV2)")
            self.get_base_model()
            
            logger.info("\n>>> Step 2: Add IMPROVED BALANCED Classification Head")
            self.update_base_model()
            
            logger.info("\n" + "="*70)
            logger.info("IMPROVED BALANCED MODEL READY")
            logger.info("="*70)
            
            logger.info("\nConfiguration Summary:")
            logger.info(f"  Architecture: MobileNetV2 + Improved Head")
            logger.info(f"  Frozen layers: First 100 layers (REDUCED from 130)")
            logger.info(f"  Trainable layers: ~55 layers (INCREASED from ~25)")
            logger.info(f"  Dropout: 0.35, 0.35, 0.3 (REDUCED from 0.45, 0.45, 0.4)")
            logger.info(f"  L2 regularization: 0.008, 0.008, 0.005 (LIGHTER)")
            logger.info(f"  Learning rate: 0.0005 (INCREASED from 0.0005)")
            logger.info(f"  Gradient clipping: 1.0")
            logger.info(f"  Strategy: BALANCED REGULARIZATION + ENHANCED LEARNING CAPACITY")
            
            logger.info("\nExpected Improvements:")
            logger.info("  ✓ Better learning capacity (30 more trainable layers)")
            logger.info("  ✓ Faster convergence (2x higher learning rate)")
            logger.info("  ✓ Less aggressive regularization (better balance)")
            logger.info("  ✓ Target: 70-80% overall accuracy")
            logger.info("  ✓ Target: 65-75% Glioma recall (from 34%)")
            
            return str(self.config.updated_base_model_path)
            
        except Exception as e:
            logger.exception(e)
            raise e