# Retinal Image Heart Attack Risk Estimator

A lightweight Streamlit application that estimates heart attack risk from retinal fundus images. It preprocesses images, highlights vasculature with a vesselness filter, extracts simple features, and produces a 0–100 risk score. A validation view reports medical-style metrics on labeled datasets. A basic scikit‑learn training pipeline is included for optional model fitting.

## Quick Start

- Windows (PowerShell):
  - `python -m pip install -r requirements.txt`
  - `streamlit run app.py`
- Open in browser:
  - http://localhost:8501

## What You Can Do

- Upload JPG/PNG/BMP retinal images
- See original vs. vesselness overlay
- View image feature metrics (intensity, vesselness)
- Get a 0–100 risk score with color-coded level
- Read a simple‑English explanation (Details tab)
- Download a JSON report
- Validate model accuracy on labeled data (Report → Open validation)

## How It Works

- Preprocessing:
  - RGB → grayscale, proportional resize, intensity normalization
- Vesselness:
  - Frangi filter highlights tubular structures (arteries/veins)
- Feature Extraction:
  - Intensity mean and standard deviation
  - Vesselness mean and standard deviation
- Risk Estimation:
  - Uses a trained logistic regression model if available
  - Falls back to a heuristic logistic mapping otherwise

## App Structure

- Main UI and flow: [app.py](file:///c:/VamshiMajorProject/app.py)
- Heuristic/trained model logic: [retina_risk/model.py](file:///c:/VamshiMajorProject/retina_risk/model.py)
- Image utilities: [retina_risk/utils.py](file:///c:/VamshiMajorProject/retina_risk/utils.py)
- Training and validation helpers: [retina_risk/train.py](file:///c:/VamshiMajorProject/retina_risk/train.py)
- Dependencies: [requirements.txt](file:///c:/VamshiMajorProject/requirements.txt)

## UI Guide

- Sidebar:
  - About and legend cards
  - Optional toggle to enable a trained model (appears only if a model file exists)
  - Sidebar image source is set in a constant inside app.py
- Main:
  - Upload widget
  - Two‑column layout with image tabs and risk summary
  - Tabs: Analysis (charts), Report (JSON + validation link), Details (plain‑English interpretation)

## Using the Validation View (Medical Accuracy)

- From the Report tab, click “Open validation”
- Provide:
  - Validation labels CSV with columns: `filename,label` where `label ∈ {0,1}`
  - Corresponding images (filenames must match)
- Reported metrics:
  - Accuracy, Precision, Recall, F1, AUC
  - ROC curve and Confusion matrix

## Training a Model (Optional)

There is a small training pipeline that fits a logistic regression using the four image features and persists it to `retina_risk/model.pkl`.

### In‑App (when a training entry point is present)

- If a “Train Model” page is available, upload a training CSV and images as for validation. The app saves the model and exposes a sidebar toggle to use it.

### Programmatic (script example)

```python
from pathlib import Path
import pandas as pd
from retina_risk.train import train_from_uploads

# Prepare files: list of paths to images, and CSV bytes with columns filename,label
image_paths = [Path("data/train/img1.jpg"), Path("data/train/img2.jpg")]
class FakeUpload:
    def __init__(self, p): self.name=str(Path(p).name); self._b=open(p,"rb").read()
    def read(self): return self._b
images = [FakeUpload(str(p)) for p in image_paths]

df = pd.DataFrame([{"filename": p.name, "label": 0} for p in image_paths])
csv_bytes = df.to_csv(index=False).encode("utf-8")

res = train_from_uploads(images, csv_bytes, "retina_risk/model.pkl")
print(res)
```

After training, start/reload the app and use the sidebar toggle to enable the trained model.

## Configuration and Theming

- The application currently uses a dark UI with pink/teal accents.
- To change the sidebar illustration:
  - Edit the `SIDEBAR_IMAGE_URL` constant near the top of [app.py](file:///c:/VamshiMajorProject/app.py)
  - The app fetches and caches the image; if it cannot be fetched, it falls back to a placeholder
- To force a global light theme (optional):
  - Create `.streamlit/config.toml` with:
    ```
    [theme]
    base = "light"
    primaryColor = "#16a34a"
    backgroundColor = "#ffffff"
    secondaryBackgroundColor = "#f0fdf4"
    textColor = "#0b0f19"
    ```
  - Remove or edit this file to return to the default/dark look

## Dataset Requirements

- Retinal fundus photographs (clear, focused)
- Labeled CSV uses binary labels (0/1). Adapt code if you use multi‑class labels.
- Filenames in CSV must exactly match uploaded image names.

## Limitations and Notes

- The heuristic score is not a medical diagnosis.
- Validation depends on the quality and representativeness of labeled data.
- Small training sets may overfit; consider cross‑validation and external test sets.
- Vesselness features are simplistic; performance can improve with richer features or CNNs.

## Extending the Project

- Add additional features:
  - Vessel caliber statistics
  - Optic disc segmentation and proximity‑aware metrics
- Replace the model:
  - RandomForest/Gradient Boosting for tabular features
  - Lightweight CNN for image‑level learning
- Packaging:
  - PyInstaller for a Windows executable
  - Dockerfile for consistent deployment

## Troubleshooting

- “Image not visible”:
  - The app now fetches the sidebar image server‑side and uses a fallback. Replace `SIDEBAR_IMAGE_URL` with a direct image URL (PNG/JPG).
- “Import errors / missing packages”:
  - Run `python -m pip install -r requirements.txt`
- “App not loading”:
  - Restart Streamlit: `streamlit run app.py`
  - Open http://localhost:8501 and hard refresh the browser

## Security and Privacy

- Do not upload sensitive patient data.
- The app runs locally and does not transmit images unless the user modifies it to do so.

## Credits

- Built with Streamlit, scikit‑image, scikit‑learn, Altair, Pillow, NumPy, Pandas.

