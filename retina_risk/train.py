import io
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from .utils import preprocess_image, compute_vesselness, compute_stats

def compute_features_from_image_bytes(data: bytes) -> dict:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(img)
    g = preprocess_image(arr)
    v = compute_vesselness(g)
    return compute_stats(g, v)

def stats_to_vector(stats: dict) -> np.ndarray:
    return np.array([
        stats["intensity_mean"],
        stats["intensity_std"],
        stats["vesselness_mean"],
        stats["vesselness_std"],
    ], dtype=np.float32)

def fit_pipeline(X: np.ndarray, y: np.ndarray) -> Pipeline:
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500)),
    ])
    pipe.fit(X, y)
    return pipe

def cross_validate_features(X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    y_true = []
    y_prob = []
    y_pred = []
    for train_idx, val_idx in skf.split(X, y):
        pipe = Pipeline(steps=[("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
        pipe.fit(X[train_idx], y[train_idx])
        if hasattr(pipe, "predict_proba"):
            prob = pipe.predict_proba(X[val_idx])[:, 1]
        else:
            prob = pipe.predict(X[val_idx]).astype(float)
        pred = (prob >= 0.5).astype(int)
        y_true.extend(y[val_idx].tolist())
        y_prob.extend(prob.tolist())
        y_pred.extend(pred.tolist())
    y_true_arr = np.array(y_true, dtype=np.int32)
    y_prob_arr = np.array(y_prob, dtype=np.float32)
    y_pred_arr = np.array(y_pred, dtype=np.int32)
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, average="binary", zero_division=0)
    try:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    fpr, tpr, thr = roc_curve(y_true_arr, y_prob_arr)
    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()},
    }

def train_from_uploads(images: list, labels_csv: bytes, save_path: str) -> dict:
    df = pd.read_csv(io.BytesIO(labels_csv))
    df["filename"] = df["filename"].astype(str)
    label_map = {row["filename"]: int(row["label"]) for _, row in df.iterrows()}
    feats = []
    ys = []
    used = 0
    for f in images:
        name = os.path.basename(f.name)
        if name in label_map:
            stats = compute_features_from_image_bytes(f.read())
            feats.append(stats_to_vector(stats))
            ys.append(label_map[name])
            used += 1
    if used == 0:
        return {"success": False, "message": "No images matched labels by filename"}
    X = np.stack(feats, axis=0)
    y = np.array(ys, dtype=np.int32)
    cv_metrics = cross_validate_features(X, y, cv=5)
    pipe = fit_pipeline(X, y)
    y_pred = pipe.predict(X)
    acc = float(accuracy_score(y, y_pred))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(pipe, f)
    return {"success": True, "accuracy": acc, "count": int(len(y)), "cv": cv_metrics}

def validate_from_uploads(images: list, labels_csv: bytes, model_path: str) -> dict:
    if not os.path.exists(model_path):
        return {"success": False, "message": "No trained model found"}
    with open(model_path, "rb") as f:
        pipe = pickle.load(f)
    df = pd.read_csv(io.BytesIO(labels_csv))
    df["filename"] = df["filename"].astype(str)
    label_map = {row["filename"]: int(row["label"]) for _, row in df.iterrows()}
    feats = []
    ys = []
    used = 0
    for f in images:
        name = os.path.basename(f.name)
        if name in label_map:
            stats = compute_features_from_image_bytes(f.read())
            feats.append(stats_to_vector(stats))
            ys.append(label_map[name])
            used += 1
    if used == 0:
        return {"success": False, "message": "No images matched labels by filename"}
    X = np.stack(feats, axis=0)
    y = np.array(ys, dtype=np.int32)
    if hasattr(pipe, "predict_proba"):
        prob = pipe.predict_proba(X)[:, 1]
    else:
        prob = pipe.predict(X).astype(float)
    pred = (prob >= 0.5).astype(int)
    acc = float(accuracy_score(y, pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    try:
        auc = float(roc_auc_score(y, prob))
        fpr, tpr, thr = roc_curve(y, prob)
    except Exception:
        auc = float("nan")
        fpr, tpr, thr = np.array([]), np.array([]), np.array([])
    cm = confusion_matrix(y, pred)
    return {
        "success": True,
        "count": int(len(y)),
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()},
    }
