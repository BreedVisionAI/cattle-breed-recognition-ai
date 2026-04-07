import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "cattle_breed_mobilenetv2.keras"
LABELS_PATH = PROJECT_ROOT / "models" / "labels.txt"
IMG_SIZE = (224, 224)


def load_labels(path):
	with open(path, "r", encoding="utf-8") as f:
		return [line.strip() for line in f.readlines() if line.strip()]


def _load_resources(model_path=MODEL_PATH, labels_path=LABELS_PATH):
	model_file = Path(model_path)
	labels_file = Path(labels_path)

	if not model_file.exists():
		raise FileNotFoundError(f"Model file not found: {model_file}")
	if not labels_file.exists():
		raise FileNotFoundError(f"Labels file not found: {labels_file}")

	custom_objects = {"preprocess_input": preprocess_input}
	try:
		model = tf.keras.models.load_model(
			model_file,
			custom_objects=custom_objects,
			compile=False,
		)
	except TypeError:
		# Fallback for models saved with Lambda layers in stricter Keras environments.
		model = tf.keras.models.load_model(
			model_file,
			custom_objects=custom_objects,
			compile=False,
			safe_mode=False,
		)
	class_names = load_labels(labels_file)
	if not class_names:
		raise ValueError("Labels file is empty")

	return model, class_names


MODEL, CLASS_NAMES = _load_resources()


def predict_image(image_path, model_path=MODEL_PATH, labels_path=LABELS_PATH):
	# Allow optional override for CLI usage while keeping API calls fast with cached resources.
	if Path(model_path) == MODEL_PATH and Path(labels_path) == LABELS_PATH:
		model = MODEL
		class_names = CLASS_NAMES
	else:
		model, class_names = _load_resources(model_path, labels_path)

	img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
	img_array = tf.keras.utils.img_to_array(img)
	img_array = np.expand_dims(img_array, axis=0)

	probs = model.predict(img_array, verbose=0)[0]
	idx = int(np.argmax(probs))
	confidence = float(probs[idx])

	top_k = min(3, len(class_names))
	top_indices = np.argsort(probs)[::-1][:top_k]
	top_predictions = [
		{
			"class": class_names[int(i)],
			"score": float(probs[int(i)]),
		}
		for i in top_indices
	]

	return {
		"predicted_class": class_names[idx],
		"confidence": confidence,
		"score": confidence,
		"score_percent": round(confidence * 100, 2),
		"top_predictions": top_predictions,
	}


def main():
	parser = argparse.ArgumentParser(description="Predict cattle breed from image")
	parser.add_argument("--image", required=True, help="Path to input image")
	args = parser.parse_args()

	result = predict_image(args.image)
	print(f"Predicted: {result['predicted_class']}")
	print(f"Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
	main()
