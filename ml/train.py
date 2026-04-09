import os
import tensorflow as tf
from tensorflow.keras import layers, models

DATASET_DIR = "data/train"
MODEL_SAVE_PATH = "models/cattle_breed_mobilenetv2.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5


def build_datasets(dataset_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "data/val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    class_names = train_ds.class_names
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model(num_classes):
    data_augmentation = tf.keras.Sequential(
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
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    os.makedirs("models", exist_ok=True)
    train_ds, val_ds, class_names = build_datasets(DATASET_DIR)
    model = build_model(num_classes=len(class_names))
    
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    model.save(MODEL_SAVE_PATH)

    labels_path = "models/labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Labels saved to: {labels_path}")
