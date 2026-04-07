from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "cattle_breed_mobilenetv2.keras"
LABELS_SAVE_PATH = PROJECT_ROOT / "models" / "labels.txt"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42


def build_datasets(dataset_dir, batch_size):
	if not dataset_dir.exists():
		raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

	train_ds = tf.keras.utils.image_dataset_from_directory(
		dataset_dir,
		validation_split=0.2,
		subset="training",
		seed=SEED,
		image_size=IMG_SIZE,
		batch_size=batch_size,
	)

	val_ds = tf.keras.utils.image_dataset_from_directory(
		dataset_dir,
		validation_split=0.2,
		subset="validation",
		seed=SEED,
		image_size=IMG_SIZE,
		batch_size=batch_size,
	)

	class_names = train_ds.class_names
	train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
	val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

	return train_ds, val_ds, class_names


def build_model(num_classes):
	augmentation = tf.keras.Sequential(
		[
			layers.RandomFlip("horizontal"),
			layers.RandomRotation(0.1),
			layers.RandomZoom(0.1),
		],
		name="augmentation",
	)

	base_model = tf.keras.applications.MobileNetV2(
		input_shape=IMG_SIZE + (3,),
		include_top=False,
		weights="imagenet",
	)
	base_model.trainable = False

	inputs = layers.Input(shape=IMG_SIZE + (3,))
	x = augmentation(inputs)
	x = layers.Rescaling(1.0 / 255)(x)
	x = base_model(x, training=False)
	x = layers.GlobalAveragePooling2D()(x)
	x = layers.Dropout(0.2)(x)
	outputs = layers.Dense(num_classes, activation="softmax")(x)

	model = models.Model(inputs, outputs)
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=["accuracy"],
	)
	return model


def save_labels(class_names, labels_path):
	labels_path.parent.mkdir(parents=True, exist_ok=True)
	with labels_path.open("w", encoding="utf-8") as label_file:
		for name in class_names:
			label_file.write(f"{name}\n")


def main():
	MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

	train_ds, val_ds, class_names = build_datasets(DATASET_DIR, BATCH_SIZE)
	model = build_model(num_classes=len(class_names))

	model.summary()
	model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
	model.save(MODEL_SAVE_PATH)
	save_labels(class_names, LABELS_SAVE_PATH)

	print(f"Model saved to: {MODEL_SAVE_PATH}")
	print(f"Labels saved to: {LABELS_SAVE_PATH}")


if __name__ == "__main__":
	main()
