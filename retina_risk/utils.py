import numpy as np
from PIL import Image
import io
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import frangi

def load_image_bytes(data: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(im)

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
    m = float(np.mean(gray))
    s = float(np.std(gray))
    vm = float(np.mean(vessel))
    vs = float(np.std(vessel))
    return {"intensity_mean": m, "intensity_std": s, "vesselness_mean": vm, "vesselness_std": vs}

def overlay_vesselness(gray: np.ndarray, vessel: np.ndarray) -> np.ndarray:
    base = (gray * 255.0).astype(np.uint8)
    r = base
    g = base
    b = (np.clip(base + (vessel * 255.0), 0, 255)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)
