import argparse
import numpy as np
import tensorflow as tf


MODEL_PATH = "models/cattle_breed_mobilenetv2.keras"
LABELS_PATH = "models/labels.txt"
IMG_SIZE = (224, 224)


def load_labels(path):
	with open(path, "r", encoding="utf-8") as f:
		return [line.strip() for line in f.readlines() if line.strip()]


def predict_image(image_path, model_path=MODEL_PATH, labels_path=LABELS_PATH):
	model = tf.keras.models.load_model(model_path)
	class_names = load_labels(labels_path)

	img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
	img_array = tf.keras.utils.img_to_array(img)
	img_array = img_array / 255.0
	img_array = np.expand_dims(img_array, axis=0)

	probs = model.predict(img_array, verbose=0)[0]
	idx = int(np.argmax(probs))

	return {
		"predicted_class": class_names[idx],
		"confidence": float(probs[idx]),
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
