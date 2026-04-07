# Cattle Breed Recognition AI

## Overview

This project classifies cattle images into four breeds using transfer learning on MobileNetV2.

Target classes:

- Gir
- Red_Sindhi
- Sahiwal
- Tharparkar

## Current Pipeline

1. Train in Colab with `colab_train_cattle_breed.ipynb`.
2. Export artifacts:
	 - `models/cattle_breed_mobilenetv2.keras`
	 - `models/labels.txt`
3. Run Flask backend for prediction API.
4. Use the frontend to upload an image and view predicted breed + score.

## Project Structure

```text
cattle-breed-recognition-ai/
	backend/
		app.py
	data/
		raw/                  # local dataset (ignored in git)
		test/                 # uploaded test images
	frontend/
		index.html
		script.js
	ml/
		train.py              # local training script
		predict.py            # inference used by backend
	colab_train_cattle_breed.ipynb
	models/                 # generated model + labels (ignored in git)
	README.md
	requirements.txt
```

## Local Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Training (Recommended: Colab)

Use `colab_train_cattle_breed.ipynb` to train with:

- pretrained MobileNetV2
- class weights for imbalance
- fine-tuning stage
- model and labels export

After training, keep these files in `models/` locally:

- `cattle_breed_mobilenetv2.keras`
- `labels.txt`

## Run Prediction API

```bash
python backend/app.py
```

API endpoint:

- `POST /predict` with multipart field `image`

Response includes:

- `predicted_class`
- `score` / `score_percent`
- `top_predictions`

## Run Frontend

Open `frontend/index.html` in a browser.

The frontend calls backend endpoint:

- default: `http://127.0.0.1:5000/predict`
- override by setting browser localStorage key `API_BASE_URL`

Example in browser console:

```js
localStorage.setItem("API_BASE_URL", "https://your-backend-url")
```

## GitHub Push Checklist

Safe to push:

- source code
- notebook
- docs

Do not push:

- dataset (`data/raw`)
- trained models (`models/*.keras`, etc.)
- local env/secrets (`.env`, virtual env folders)

## Deployment Note

GitHub Pages is static-only.

- Frontend can be hosted on GitHub Pages.
- Flask + TensorFlow backend must be hosted separately (Render/Railway/Azure/etc.).
