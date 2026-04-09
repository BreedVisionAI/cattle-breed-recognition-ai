# Cattle Breed Recognition using Deep Learning 🐄

## Project Overview
This academic project aims to accurately identify various cattle breeds given a raw image. Accurate breed identification is essential for livestock management, targeted breeding, and veterinary applications. We utilize a Deep Learning pipeline focused on high efficiency and deployment flexibility.

## Model Used
We used **MobileNetV2** with transfer learning to efficiently train our model on a limited dataset and deployed it using Streamlit for real-time prediction. MobileNetV2 is specifically designed for mobile and resource-constrained environments, making it incredibly fast without sacrificing substantial accuracy.

## Workflow
All major technical components for model building are unified in the root directory:
1. **Preprocessing**: Standardizes input image size to 224x224, pixels between -1 and 1, and applies Data Augmentation (Flips, Rotation, Zoom) to prevent overfitting.
2. **Splitting and Loading**: Automates loading pre-split training vs. validation sets and sets up prefetching for rapid GPU handoff.
3. **Training**: Loads a pre-trained ImageNet `MobileNetV2`, fine-tunes the top layers, and trains a custom 128-node brain.
4. **Prediction / Inference Check**: Performs sanity checks testing single images against the exported `.h5` model.
5. **Deployment (`app.py`)**: A Streamlit application built for a user-friendly, real-time interface with a built-in Keras compatibility patch.

## How to Run

### Google Colab Workflow (Model Training)
1. Upload the `model_training_pipeline.ipynb` to Google Colab.
2. Mount your drive and ensure your `data` folder contains `train`, `val`, and `test` splits.
3. Run the cells in `model_training_pipeline.ipynb`.
4. Download the generated `cattle_breed_mobilenetv2.h5` model to your local machine, placing it in the root of this project.

### Local Streamlit Workflow (Inference)
1. Ensure you have installed the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Verify `cattle_breed_mobilenetv2.h5` is in the root directory alongside `app.py`.
3. Launch Streamlit:
   ```bash
   streamlit run app.py
   ```

## Team Members
* Person 1 - Problem Statement & Dataset
* Person 2 - Preprocessing & Dataset Loading Module
* Person 3 - Training & MobileNetV2 Focus
* Person 4 - Inference Engine & Streamlit Wrapper
* Person 5 - Results, Metrics, & Conclusion
