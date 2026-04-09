import streamlit as st
from PIL import Image
import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Keras 3.x -> 2.x Compatibility Patch ---
# If your model was exported in Colab (Keras 3) but loaded locally (Keras 2),
# it throws a "quantization" configuration error. This code safely bypasses that!
class MyDense(tf.keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return super().from_config(config)

class MyDropout(tf.keras.layers.Dropout):
    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return super().from_config(config)
# --------------------------------------------

st.set_page_config(page_title="Cattle Breed Recognition", page_icon="🐄")

@st.cache_resource
def load_predictor():
    try:
        model = tf.keras.models.load_model(
            'cattle_breed_mobilenetv2.h5', 
            custom_objects={'Dense': MyDense, 'Dropout': MyDropout}, 
            compile=False
        )
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_predictor()
class_names = ['gir', 'red_sindhi', 'sahiwal', 'tharparkar']

st.title("🐄 Cattle Breed Recognition")
st.write("Upload an image of cattle, and the Deep Learning model (MobileNetV2) will predict its breed!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_disp = Image.open(uploaded_file)
    st.image(img_disp, caption="Uploaded Image", use_container_width=True)
    st.write("")
    
    if model is None:
        st.error("Model file 'cattle_breed_mobilenetv2.h5' not found or failed to load. Ensure it is in the root directory.")
    else:
        with st.spinner("Extracting visual features via MobileNetV2..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf_file:
                tf_file.write(uploaded_file.getbuffer())
                temp_path = tf_file.name
            
            # Predict Logic
            img = image.load_img(temp_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = float(np.max(predictions))
            
            os.remove(temp_path)
            
        st.success(f"**Prediction:** {predicted_class.replace('_', ' ').title()}")
        st.info(f"**Confidence:** {confidence * 100:.2f}%")
