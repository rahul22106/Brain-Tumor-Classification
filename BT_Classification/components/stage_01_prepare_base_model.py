import os
from pathlib import Path
import tensorflow as tf
from BT_Classification import logger
from BT_Classification.entity import PrepareBaseModelConfig


class PrepareBaseModel:
    """
    Component for preparing base model (MobileNetV2) with custom classification head
    """
    
    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initialize Base Model Preparation component
        
        Args:
            config (PrepareBaseModelConfig): Configuration for base model
        """
        self.config = config
    
    
    def get_base_model(self):
        """
        Download and save MobileNetV2 base model with ImageNet weights
        """
        try:
            logger.info("Loading MobileNetV2 base model...")
            
            # Load MobileNetV2 with ImageNet weights
            self.model = tf.keras.applications.MobileNetV2(
                input_shape=self.config.params_image_size,
                include_top=self.config.params_include_top,
                weights=self.config.params_weights
            )
            
            # Save base model
            self.model.save(str(self.config.base_model_path))
            logger.info(f"✓ Base model saved at: {self.config.base_model_path}")
            
            # Model summary
            logger.info(f"✓ Model: MobileNetV2")
            logger.info(f"✓ Input shape: {self.config.params_image_size}")
            logger.info(f"✓ Total layers: {len(self.model.layers)}")
            logger.info(f"✓ Trainable params: {self.model.count_params():,}")
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Add custom classification head to base model
        
        Args:
            model: Base MobileNetV2 model
            classes: Number of output classes
            freeze_all: Whether to freeze all base layers
            freeze_till: Number of layers to freeze (if freeze_all is False)
            learning_rate: Learning rate for optimizer
            
        Returns:
            Updated model with custom head
        """
        try:
            # Freeze layers if specified
            if freeze_all:
                logger.info("Freezing all base model layers...")
                for layer in model.layers:
                    layer.trainable = False
            elif freeze_till is not None and freeze_till > 0:
                logger.info(f"Freezing first {freeze_till} layers...")
                for layer in model.layers[:freeze_till]:
                    layer.trainable = False
            
            # Count trainable and non-trainable parameters
            trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            non_trainable_count = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
            
            logger.info(f"✓ Trainable parameters: {trainable_count:,}")
            logger.info(f"✓ Non-trainable parameters: {non_trainable_count:,}")
            
            # Add custom classification head
            logger.info("Adding custom classification head...")
            
            # Global Average Pooling
            flatten_in = tf.keras.layers.GlobalAveragePooling2D()(model.output)
            
            # Dropout for regularization
            dropout = tf.keras.layers.Dropout(0.3)(flatten_in)
            
            # Dense layer with ReLU activation
            dense = tf.keras.layers.Dense(
                units=128,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            )(dropout)
            
            # Another dropout
            dropout2 = tf.keras.layers.Dropout(0.2)(dense)
            
            # Output layer for 4 tumor classes
            prediction = tf.keras.layers.Dense(
                units=classes,
                activation='softmax',
                name='output_layer'
            )(dropout2)
            
            # Create full model
            full_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=prediction
            )
            
            # Compile model
            logger.info("Compiling model...")
            full_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            logger.info("✓ Model compiled successfully")
            logger.info(f"✓ Optimizer: Adam (lr={learning_rate})")
            logger.info(f"✓ Loss: Categorical Crossentropy")
            logger.info(f"✓ Metrics: Accuracy, Precision, Recall")
            
            return full_model
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    
    def update_base_model(self):
        """
        Update base model with custom classification head and save
        """
        try:
            logger.info("Updating base model with custom head...")
            
            # Prepare full model with custom head
            self.full_model = self._prepare_full_model(
                model=self.model,
                classes=self.config.params_classes,
                freeze_all=True,  # Freeze all for transfer learning
                freeze_till=None,
                learning_rate=self.config.params_learning_rate
            )
            
            # Save updated model
            self.full_model.save(str(self.config.updated_base_model_path))
            logger.info(f"✓ Updated model saved at: {self.config.updated_base_model_path}")
            
            # Print model summary
            logger.info("\n" + "="*70)
            logger.info("MODEL ARCHITECTURE SUMMARY")
            logger.info("="*70)
            self.full_model.summary(print_fn=logger.info)
            logger.info("="*70)
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    
    def initiate_base_model_preparation(self):
        """
        Main method to execute base model preparation
        """
        try:
            logger.info("="*70)
            logger.info("STARTING BASE MODEL PREPARATION")
            logger.info("="*70)
            
            # Step 1: Get base MobileNetV2 model
            logger.info("\n>>> Step 1: Load Base Model (MobileNetV2)")
            self.get_base_model()
            
            # Step 2: Update with custom classification head
            logger.info("\n>>> Step 2: Add Custom Classification Head")
            self.update_base_model()
            
            logger.info("\n" + "="*70)
            logger.info("✓ BASE MODEL PREPARATION COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            
            logger.info("\nModel Details:")
            logger.info(f"  → Architecture: MobileNetV2 + Custom Head")
            logger.info(f"  → Input Size: {self.config.params_image_size}")
            logger.info(f"  → Number of Classes: {self.config.params_classes}")
            logger.info(f"  → Class Names: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor")
            logger.info(f"  → Base Model: Frozen (Transfer Learning)")
            logger.info(f"  → Learning Rate: {self.config.params_learning_rate}")
            logger.info(f"  → Model Ready for Training!")
            
            return str(self.config.updated_base_model_path)
            
        except Exception as e:
            logger.exception(e)
            raise e