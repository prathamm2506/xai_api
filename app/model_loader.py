"""
Model loader module for DenseNet121 lung cancer classification model.
"""
import os
import tensorflow as tf
from tensorflow.keras.models import load_model


def categorical_focal_loss(alpha, gamma=2.0):
    """
    Custom focal loss function matching training configuration.
    
    Args:
        alpha: Class weights array
        gamma: Focusing parameter for focal loss
    
    Returns:
        Loss function
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1.0 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    
    return loss


class ModelLoader:
    """Singleton model loader for the lung cancer classification model."""
    
    _instance = None
    _model = None
    
    # Class mapping based on training
    CLASS_INDICES = {
        "Benign": 0,
        "Malignant": 1,
        "Normal": 2
    }
    
    # Inverse mapping
    INDICES_TO_CLASS = {v: k for k, v in CLASS_INDICES.items()}
    
    # Focal loss alpha values used in training
    FOCAL_LOSS_ALPHA = [2.5, 1.0, 1.0]
    FOCAL_LOSS_GAMMA = 2.0
    
    # Grad-CAM layer name for DenseNet121
    LAST_CONV_LAYER = "conv5_block16_concat"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path=None):
        """
        Load the trained model if not already loaded.
        
        Args:
            model_path: Path to the .keras model file
            
        Returns:
            Loaded Keras model
        """
        if self._model is not None:
            return self._model
        
        if model_path is None:
            # Default path
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "densenet_focal_phase3.keras")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        self._model = load_model(
            model_path,
            custom_objects={
                "loss": categorical_focal_loss(
                    alpha=self.FOCAL_LOSS_ALPHA,
                    gamma=self.FOCAL_LOSS_GAMMA
                )
            },
            compile=False
        )
        
        print("Model loaded successfully!")
        print(f"Input shape: {self._model.input_shape}")
        print(f"Output classes: {self.INDICES_TO_CLASS}")
        model.trainalbe = False  # Set model to inference mode
        
        return self._model
    
    def get_model(self):
        """Get the loaded model instance."""
        if self._model is None:
            return self.load_model()
        return self._model
    
    def get_classes(self):
        """Get list of class names."""
        return list(self.CLASS_INDICES.keys())
    
    def get_class_indices(self):
        """Get class to index mapping."""
        return self.CLASS_INDICES
    
    def get_last_conv_layer(self):
        """Get the last convolutional layer name for Grad-CAM."""
        return self.LAST_CONV_LAYER


# Global instance
model_loader = ModelLoader()

