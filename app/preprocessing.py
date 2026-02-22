"""
Image preprocessing module for lung cancer classification.
"""
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.applications.densenet import preprocess_input


IMG_SIZE = (224, 224)


def load_and_preprocess_image(image_file, target_size=IMG_SIZE):
    """
    Load and preprocess an image for model prediction.

    Accepts:
    - UploadFile
    - file path (str)
    - PIL Image
    - numpy array
    """

    # ----------------------------
    # 1️⃣ Handle numpy array directly
    # ----------------------------
    if isinstance(image_file, np.ndarray):
        image = Image.fromarray(image_file)

    # PIL image
    elif isinstance(image_file, Image.Image):
        image = image_file.copy()

    # File-like object
    elif hasattr(image_file, "read"):
        image = Image.open(image_file).copy()

    # File path
    else:
        image = Image.open(image_file).copy()

    # ----------------------------
    # 2️⃣ Force RGB
    # ----------------------------
    if image.mode != "RGB":
        image = image.convert("RGB")

    # ----------------------------
    # 3️⃣ Deterministic resize
    # ----------------------------
    image = image.resize(target_size, Image.BILINEAR)

    # ----------------------------
    # 4️⃣ Convert to float32
    # ----------------------------
    img_array = np.array(image, dtype=np.float32)

    # ----------------------------
    # 5️⃣ DenseNet preprocessing
    # ----------------------------
    img_array = preprocess_input(img_array)

    # ----------------------------
    # 6️⃣ Add batch dimension
    # ----------------------------
    img_array = np.expand_dims(img_array, axis=0)

    return img_array




def load_image_for_visualization(image_file, target_size=IMG_SIZE):
    """
    Load image for visualization (without DenseNet preprocessing).
    
    Args:
        image_file: Uploaded file object or file path
        target_size: Target image size
    
    Returns:
        Numpy array of the image (0-255 range)
    """
    if hasattr(image_file, 'read'):
        image = Image.open(image_file)
    else:
        image = Image.open(image_file)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.uint8)
    
    return img_array


def decode_base64_image(base64_string):
    """
    Decode base64 encoded image string.
    
    Args:
        base64_string: Base64 encoded image string
    
    Returns:
        PIL Image object
    """
    import base64
    from io import BytesIO
    
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode
    img_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(img_data))
    
    return image


def preprocess_for_gradcam(image_array):
    """
    Preprocess image array for Grad-CAM generation.
    Reverses DenseNet preprocessing.
    
    Args:
        image_array: Preprocessed image array
    
    Returns:
        Original scale image array
    """
    # Reverse the preprocessing
    img = image_array.copy()
    img = img + 1.0  # Reverse the -1 to 1 shift
    img = img * 127.5  # Reverse the /127.5 scaling
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def create_lung_mask(image_rgb):
    """
    Create a binary lung mask for the image.
    
    Args:
        image_rgb: RGB image array
    
    Returns:
        Binary mask (0-255)
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's thresholding
    _, mask = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def resize_and_normalize(img_array, target_size=IMG_SIZE):
    """
    Resize image and normalize for preprocessing.
    
    Args:
        img_array: Input image array
        target_size: Target size tuple
    
    Returns:
        Preprocessed image array
    """
    if isinstance(img_array, Image.Image):
        img_array = img_array.resize(target_size)
        img_array = np.array(img_array, dtype=np.float32)
    
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

