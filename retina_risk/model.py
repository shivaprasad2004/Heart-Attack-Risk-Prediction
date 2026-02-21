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
    x1 = stats["intensity_mean"]
    x2 = stats["intensity_std"]
    x3 = stats["vesselness_mean"]
    x4 = stats["vesselness_std"]
    z = 0.8 * (1.0 - x1) + 1.2 * x2 + 2.0 * x3 + 0.7 * x4 - 0.5
    p = 1.0 / (1.0 + math.exp(-5.0 * z))
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
