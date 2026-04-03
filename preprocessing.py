import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# 1. SETUP CONSTANTS
# ==========================================
# Point this to your local folder containing the breed subfolders
DATA_DIR = "C:\\Users\\sulav\\Desktop\\Developer\\IP_project\\cattle-breed-recognition-ai\\datasets"

IMG_SIZE = (224, 224) # Strict requirement for MobileNetV2

BATCH_SIZE = 16       

# ==========================================
# 2. LOAD AND SPLIT DATASET
# ==========================================
print("Loading training data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("Loading validation data...")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Extract the class names (the breed names) for later use
class_names = train_dataset.class_names
print(f"Breeds found: {class_names}")

# ==========================================
# 3. DEFINE DATA AUGMENTATION
# ==========================================
# This applies random transformations to prevent overfitting.
# It runs directly on your GPU for maximum speed.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"), # Flips left/right
    tf.keras.layers.RandomRotation(0.15),     # Tilts up to 15%
    tf.keras.layers.RandomZoom(0.1),          # Zooms in/out 10%
])

# ==========================================
# 4. APPLY NORMALIZATION & AUGMENTATION
# ==========================================
# MobileNetV2 requires pixels to be scaled between -1 and 1.
# This function applies Keras's built-in MobileNetV2 preprocessor.
def format_image(image, label):
    image = preprocess_input(image)
    return image, label

# Apply augmentation ONLY to the training data
train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Apply the strict MobileNetV2 normalization to BOTH datasets
train_dataset = train_dataset.map(format_image, num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.map(format_image, num_parallel_calls=tf.data.AUTOTUNE)

# ==========================================
# 5. OPTIMIZE FOR PERFORMANCE
# ==========================================
# prefetch() overlaps data preprocessing and model execution while training.
# This ensures your RTX 3050 never has to wait for the next batch of images.
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

print("Data pipeline ready! You can now pass train_dataset into model.fit()")
# ==========================================
# 6. BUILD THE MODEL (TRANSFER LEARNING)
# ==========================================
print("Downloading MobileNetV2 base model...")

# We calculate the number of breeds from the previous step
num_classes = len(class_names) 

# Load the base model. 
# include_top=False means we chop off the original 1000-class output layer.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model so we don't destroy its pre-trained knowledge
base_model.trainable = False

# Create the new classification head for your cattle breeds
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False) # Keep base model in inference mode
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x) # Helps prevent overfitting

# The final layer matches your number of cattle breeds
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# ==========================================
# 7. COMPILE AND TRAIN
# ==========================================
print("Compiling model...")
# Since our labels are integers (0, 1, 2...), we use SparseCategoricalCrossentropy
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary() # This will print a nice breakdown of your model's architecture

# Let's set it to train for 10 epochs (passes over the entire dataset) to start.
# Your RTX 3050 should handle this quite quickly!
INITIAL_EPOCHS = 10

print("Starting training! Keep an eye on the validation accuracy...")
history = model.fit(
    train_dataset,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_dataset
)

# Save the trained model to your local hard drive
model.save("cattle_breed_mobilenetv2.keras")
print("Model saved successfully!")