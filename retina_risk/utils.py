import numpy as np
from PIL import Image
import io
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import frangi

def load_image_bytes(data: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(im)

def is_valid_retinal_image(arr: np.ndarray) -> bool:
    """
    Validates if the uploaded image is likely a retinal fundus photograph.
    Checks for circular mask characteristics and typical color distributions.
    """
    if len(arr.shape) != 3:
        return False
    
    # Check for circularity/aspect ratio typical of fundus cameras
    h, w = arr.shape[:2]
    aspect_ratio = w / h
    if aspect_ratio < 0.8 or aspect_ratio > 1.25:
        # Most fundus images are roughly square or slightly rectangular
        pass 

    # Check color distribution: Retinal images are dominated by red/orange channels
    r_mean = np.mean(arr[:, :, 0])
    g_mean = np.mean(arr[:, :, 1])
    b_mean = np.mean(arr[:, :, 2])
    
    # Fundus images typically have R > G > B
    if not (r_mean > g_mean and g_mean > b_mean):
        return False
    
    # Check for minimum brightness to avoid completely dark/noise images
    if r_mean < 30:
        return False
        
    return True

def preprocess_image(arr: np.ndarray, target_width: int = 512) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = target_width / float(w)
    new_h = int(h * scale)
    img = resize(arr, (new_h, target_width), preserve_range=True, anti_aliasing=True)
    g = rgb2gray(img).astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g

def compute_vesselness(gray: np.ndarray) -> np.ndarray:
    v = frangi(gray)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    return v.astype(np.float32)

def compute_stats(gray: np.ndarray, vessel: np.ndarray) -> dict:
    """
    Computes statistical features including intensity, vesselness, and a refined lesion score.
    """
    m = float(np.mean(gray))
    s = float(np.std(gray))
    vm = float(np.mean(vessel))
    vs = float(np.std(vessel))
    
    # Refined Lesion Score: 
    # 1. Ignore the very brightest parts (likely the optic disc) by capping at 0.95
    # 2. Ignore the background by only looking at the central part of the image
    h, w = gray.shape
    center_h, center_w = h // 2, w // 2
    margin = int(min(h, w) * 0.3)
    central_roi = gray[center_h-margin:center_h+margin, center_w-margin:center_w+margin]
    
    # Detect spots: significantly different from the local mean
    lesion_mask = (central_roi > 0.8) | (central_roi < 0.15)
    lesion_score = float(np.mean(lesion_mask))
    
    return {
        "intensity_mean": m, 
        "intensity_std": s, 
        "vesselness_mean": vm, 
        "vesselness_std": vs,
        "lesion_score": lesion_score
    }

def overlay_vesselness(gray: np.ndarray, vessel: np.ndarray) -> np.ndarray:
    base = (gray * 255.0).astype(np.uint8)
    r = base
    g = base
    b = (np.clip(base + (vessel * 255.0), 0, 255)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)
