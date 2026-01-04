# Eye Disease Classification

This repository contains an AI project for classifying eye diseases from retinal images. It includes data preprocessing, model training notebooks, trained models, and a small Flask web app for inference.

## Quick Start

- Create a Python virtual environment and activate it.
- Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

- Run the web app (from project root):

```powershell
python .\web_app\app.py
```

Open `http://127.0.0.1:5000/` in your browser.

## Results

The `results/eye_disease` folder contains analysis artifacts and visualizations produced during training and evaluation. Preview of key images:

- Class distribution
	![Class Distribution](results/eye_disease/class_distribution.png)

- Confusion matrix
	![Confusion Matrix](results/eye_disease/confusion_matrix.png)

- Normalized confusion matrix
	![Confusion Matrix Normalized](results/eye_disease/confusion_matrix_normalized.png)

- ROC curves
	![ROC Curves](results/eye_disease/roc_curves.png)

- Grad-CAM visualizations
	![Grad-CAM Visualizations](results/eye_disease/grad_cam_visualizations.png)

- Sample images
	![Sample Images](results/eye_disease/sample_images.png)

- Training history (ViT)
	![ViT Training History](results/eye_disease/vit_training_history.png)

- Augmented images overview
	![Augmented Images](results/eye_disease/augmented_images.png)

- Performance summary (CSV): `results/eye_disease/performance_summary.csv`
- Classification report (CSV): `results/eye_disease/classification_report.csv`

## Notes
- If the web app shows "Model: Unavailable", check `web_app/app.py` logs or visit `http://127.0.0.1:5000/status` for the model load error details.
- The project expects the dataset under `data/` as organized in the repo. See notebooks for preprocessing and training steps.

If you want, I can add automatic thumbnails of these images into the docs, or generate an HTML gallery page under `results/`.

