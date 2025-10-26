import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import string
from PIL import Image
import pandas as pd
from streamlit_drawable_canvas import st_canvas

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="EMNIST Character Recognition",
    page_icon="✍️",
)
st.title("✍️ EMNIST Character Recognizer")
st.markdown("Draw a digit (0-9), uppercase letter (A-Z), or lowercase letter (a-z) on the canvas below.")

# --- 2. Load Model and Labels ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_emnist_model():
    try:
        model = load_model('emnist_data_augmentation.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model 'emnist_data_augmentation.keras'. Make sure the file is in the same directory. Error: {e}")
        return None

model = load_emnist_model()
emnist_byclass_labels = list('0123456789' + string.ascii_uppercase + string.ascii_lowercase)
num_classes = 62

# --- 3. Create the Drawing Canvas ---
# We set the background to black and stroke to white, just like our EMNIST data!
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",  # Black background
    stroke_width=15,               # Brush width
    stroke_color="#FFFFFF",        # White stroke
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# --- 4. Prediction Logic ---
if st.button("Predict") and model is not None:
    if canvas_result.image_data is not None:
        # Get the drawing data (it's a 4-channel RGBA numpy array)
        img_data = canvas_result.image_data
        
        # Convert to PIL Image and then to grayscale
        # The canvas gives us RGBA, we just need the 'L' (grayscale)
        img_pil = Image.fromarray(img_data.astype('uint8')).convert('L')
        
        # --- Preprocessing ---
        # Resize to 28x28
        img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_np = np.array(img_resized)
        # img_np = np.flip(img_np, axis=0) # Flip vertically to match EMNIST orientation
        
        # **IMPORTANT**: Because we drew white-on-black, we DO NOT need to invert.
        # We just normalize.
        img_np = img_np / 255.0
        
        # Reshape for the model (1, 28, 28, 1)
        img_batch = img_np.reshape(1, 28, 28, 1)

        # --- Make Prediction ---
        prediction = model.predict(img_batch)
        pred_index = np.argmax(prediction)
        pred_char = emnist_byclass_labels[pred_index]
        confidence = prediction[0][pred_index] * 100

        st.success(f"Prediction: **{pred_char}** (Confidence: {confidence:.2f}%)")
        
        # --- Show Confidences ---
        st.write("Top 5 Predictions:")
        
        # Create a DataFrame for the bar chart
        top_indices = np.argsort(prediction[0])[-5:][::-1]
        top_chars = [emnist_byclass_labels[i] for i in top_indices]
        top_confs = [prediction[0][i] for i in top_indices]
        
        df = pd.DataFrame({
            'Character': top_chars,
            'Confidence': top_confs
        })
        
        st.bar_chart(df.set_index('Character'))

    else:
        st.warning("Please draw a character on the canvas first!")