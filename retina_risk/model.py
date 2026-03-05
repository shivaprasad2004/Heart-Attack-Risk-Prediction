import math
import os
import pickle
import numpy as np

_PIPELINE = None
_USE_TRAINED = False
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

def _load_pipeline():
    global _PIPELINE, _USE_TRAINED
    if _PIPELINE is None and os.path.exists(_MODEL_PATH):
        try:
            with open(_MODEL_PATH, "rb") as f:
                _PIPELINE = pickle.load(f)
                _USE_TRAINED = True
        except Exception:
            _PIPELINE = None
            _USE_TRAINED = False

def has_trained_model() -> bool:
    _load_pipeline()
    return _PIPELINE is not None

def set_use_trained(use: bool):
    global _USE_TRAINED
    _load_pipeline()
    _USE_TRAINED = bool(use) and (_PIPELINE is not None)

def _heuristic_score(stats: dict) -> float:
    """
    Computes a balanced risk score to avoid constant High or constant Low results.
    """
    x1 = stats["intensity_mean"]
    x2 = stats["intensity_std"]
    x3 = stats["vesselness_mean"]
    x4 = stats["vesselness_std"]
    x5 = stats.get("lesion_score", 0.0)
    
    # Balanced weights to allow full range of scores (0-100)
    # Vessel Component: Reduced weight to avoid over-sensitivity
    vessel_comp = (15.0 * x3) + (5.0 * x4)
    
    # Lesion Component: Key differentiator for High risk, but tuned
    lesion_comp = (8.0 * x5)
    
    # Contrast Component: Helps with overall image quality
    contrast_comp = (1.5 * x2)
    
    # Final combined score Z with a more neutral bias
    z = vessel_comp + lesion_comp + contrast_comp - 1.2
    
    # Stepped Sigmoid for better category separation
    p = 1.0 / (1.0 + math.exp(-3.5 * z))
    return float(p * 100.0)

def predict_risk(stats: dict) -> float:
    _load_pipeline()
    if _PIPELINE is not None and _USE_TRAINED:
        x = np.array([[stats["intensity_mean"], stats["intensity_std"], stats["vesselness_mean"], stats["vesselness_std"]]], dtype=np.float32)
        try:
            if hasattr(_PIPELINE, "predict_proba"):
                p = float(_PIPELINE.predict_proba(x)[0, 1])
            else:
                p = float(_PIPELINE.predict(x)[0])
                p = max(0.0, min(1.0, p))
            return p * 100.0
        except Exception:
            return _heuristic_score(stats)
    return _heuristic_score(stats)
