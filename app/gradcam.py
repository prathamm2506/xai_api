"""
Grad-CAM visualization module for explainable AI.
"""
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import base64


def make_gradcam_heatmap(model, img_array, layer_name, class_idx=None):
    """
    Generate Grad-CAM heatmap for a given image.
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array (1, 224, 224, 3)
        layer_name: Name of the last convolutional layer
        class_idx: Class index for which to generate heatmap.
                   If None, uses the predicted class.
    
    Returns:
        Grad-CAM heatmap (224, 224)
    """
    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ]
    )
    
    # If no class specified, use predicted class
    if class_idx is None:
        predictions = model.predict(img_array, verbose=0)
        
        # Handle different output formats from model.predict
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]
        
        # Ensure predictions is a numpy array
        predictions = np.array(predictions)
        
        # If still a batch (2D), take the first sample
        if len(predictions.shape) > 1:
            predictions = predictions[0]
        
        class_idx = int(np.argmax(predictions))
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)

    # If model outputs tuple
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

    # predictions shape: (1, num_classes)
        loss = predictions[:, class_idx]

    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Compute CAM
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    
    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy(), class_idx


def apply_lung_mask(heatmap, image_rgb):
    """
    Apply lung mask to focus Grad-CAM on lung regions only.
    
    Args:
        heatmap: Grad-CAM heatmap (normalized 0-1), shape (224, 224)
        image_rgb: Original RGB image (0-255), shape (224, 224, 3)
    
    Returns:
        Masked heatmap with lung regions preserved
    """
    from .preprocessing import create_lung_mask

    # Resize heatmap to image size FIRST
    heatmap_resized = cv2.resize(
        heatmap,
        (image_rgb.shape[1], image_rgb.shape[0])
    )
    
    # Create lung mask
    mask = np.array(create_lung_mask(image_rgb), dtype=np.float32)
    
    # Normalize mask to 0-1 range
    mask_normalized = mask / 255.0
    
    # Apply mask - preserve lung regions, suppress background
    # Using multiplication to focus on lung areas while keeping gradient values
    heatmap_masked = heatmap_resized * mask_normalized
    
    # Normalize result to 0-1 range (important for proper colormap)
    if heatmap_masked.max() > 0:
        heatmap_masked = heatmap_masked / heatmap_masked.max()
    
    return heatmap_masked





def create_overlay(heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):

    if original_image.dtype != np.uint8:
        original_image = np.uint8(original_image)

    # Resize heatmap
    heatmap_resized = cv2.resize(
        heatmap,
        (original_image.shape[1], original_image.shape[0])
    )

    # Normalize once
    heatmap_min = heatmap_resized.min()
    heatmap_max = heatmap_resized.max()

    if heatmap_max > heatmap_min:
        heatmap_normalized = (heatmap_resized - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap_normalized = np.zeros_like(heatmap_resized)

    # 🔥 Smooth
    heatmap_normalized = cv2.GaussianBlur(heatmap_normalized, (5, 5), 0)
    # 🔥 Suppress weak noise
    heatmap_normalized[heatmap_normalized < 0.15] = 0


    # 🔥 Enhance contrast
    heatmap_uint8 = np.uint8(255 * (heatmap_normalized ** 0.7))

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend properly
    overlay = cv2.addWeighted(
        original_image,
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )
    # heatmap_normalized = np.power(heatmap_normalized, 1.5) # Enhance contrast further


    return overlay



def generate_visualizations(original_image, model, layer_name):
    """
    Generate all visualization images.
    
    Args:
        original_image: Original RGB image (PIL Image or numpy array)
        model: Trained Keras model
        layer_name: Name of last convolutional layer
    
    Returns:
        Dictionary containing base64 encoded images
    """
    from .preprocessing import load_and_preprocess_image, IMG_SIZE
    
    # Convert PIL to numpy if needed
    if hasattr(original_image, 'convert'):
        original_image = original_image.resize(IMG_SIZE)
        original_image = np.array(original_image)
    
    # Preprocess for model
    img_array = load_and_preprocess_image(original_image)
    
    # Get predictions
    predictions = model.predict(img_array, verbose=0)
    
    # Handle different output formats from model.predict
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    
    # Ensure predictions is a numpy array
    predictions = np.array(predictions)
    
    # If still a batch (2D), take the first sample
    if len(predictions.shape) > 1:
        predictions = predictions[0]
    
    predicted_class = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    
    # Generate Grad-CAM
    heatmap, _ = make_gradcam_heatmap(
        model, img_array, layer_name, predicted_class
    )
    
    # Apply lung mask
    heatmap_masked = apply_lung_mask(heatmap, original_image)
    
    # Create overlays
    # gradcam: overlay WITH lung mask (focused on lung regions)
    overlay_masked = create_overlay(heatmap_masked, original_image, alpha=0.4)
    # overlay: overlay WITHOUT lung mask (raw heatmap on original)
    heatmap_resized = cv2.resize(
        heatmap,
        (original_image.shape[1], original_image.shape[0])
    )

    overlay_raw = create_overlay(heatmap_resized, original_image, alpha=0.4)

    
    # Convert to base64
    def array_to_base64(img_arr):
        """Convert numpy array to base64 string."""
        # Ensure proper normalization to 0-1 range
        if img_arr.dtype != np.uint8:
            img_arr = np.uint8(255 * np.clip(img_arr, 0, 1))

        
        if img_arr.dtype != np.uint8:
            img_arr = np.uint8(255 * img_arr)
        
        img_pil = Image.fromarray(img_arr)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    
    # ========================================
    # FIXED: Return colorful heatmap with COLORMAP_JET
    # Following exact pipeline from TODO.md
    # ========================================
    
    # STEP 1: Normalize the masked heatmap (0-1 range)
    heatmap_normalized = np.maximum(heatmap_masked, 0)
    if heatmap_normalized.max() > 0:
        heatmap_normalized = heatmap_normalized / (heatmap_normalized.max() + 1e-8)
    
    # STEP 2: Convert to colorful heatmap (JET colormap)
    heatmap_uint8 = np.uint8(255 * heatmap_normalized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    return {
        'original': array_to_base64(original_image),
        'heatmap': array_to_base64(heatmap_color),  # Return colorful heatmap
        'gradcam': array_to_base64(overlay_masked),
        'overlay': array_to_base64(overlay_raw),
        'predicted_class': predicted_class,
        'confidence': confidence
    }


def get_attention_regions(heatmap, threshold=0.5):
    """
    Extract high-attention regions from Grad-CAM heatmap.
    
    Args:
        heatmap: Grad-CAM heatmap (normalized 0-1)
        threshold: Threshold for region detection
    
    Returns:
        List of regions with coordinates and intensity
    """
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Find high-intensity regions
    _, binary = cv2.threshold(
        np.uint8(255 * heatmap),
        int(255 * threshold),
        255,
        cv2.THRESH_BINARY
    )
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    h, w = heatmap.shape
    
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        # Calculate average intensity in region
        roi = heatmap[y:y+ch, x:x+cw]
        intensity = float(np.mean(roi))
        
        regions.append({
            'x': float(x / w * 100),  # Normalize to percentage
            'y': float(y / h * 100),
            'width': float(cw / w * 100),
            'height': float(ch / h * 100),
            'intensity': intensity,
            'center_x': float((x + cw/2) / w * 100),
            'center_y': float((y + ch/2) / h * 100)
        })
    
    return regions

