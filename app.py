import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import operator

# --- Page Configuration ---
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✍️",
)

# --- Model Loading ---
@st.cache_resource
def load_keras_model():
    """Loads the pre-trained Keras model from an .h5 file."""
    try:
        model = load_model('digit_model.h5')
        return model
    except (FileNotFoundError, IOError) as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure the 'digit_model.h5' file is in the same directory as this app.")
        return None

model = load_keras_model()

# --- App Title and Description ---
st.title("✍️ Handwritten Digit Recognizer")
st.markdown("Draw a digit (0-9) in the box below, and the AI will try to guess what it is!")

# --- Drawing Canvas ---
col1, col2 = st.columns([3, 1]) # Create columns for canvas and controls

with col1:
    st.subheader("Drawing Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=20, # Pen thickness
        stroke_color='#FFFFFF', # Pen color (white)
        background_color='#000000', # Background color (black)
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

# --- Control Buttons ---
with col2:
    st.subheader("Controls")
    predict_button = st.button("Predict Digit")
    clear_button = st.button("Clear Canvas")

# --- Prediction Logic ---
if canvas_result.image_data is not None and predict_button:
    if model is not None:
        # Get the drawn image data
        img = canvas_result.image_data.astype('uint8')
        
        # Check if the canvas is empty (all black)
        if np.sum(img) == 0:
            st.warning("Please draw a digit first!")
        else:
            # Preprocess the image to match the model's input requirements
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (28, 28))
            img_reshaped = img_resized.reshape(1, 28, 28, 1)
            img_normalized = img_reshaped / 255.0

            # Make a prediction
            with st.spinner("Analyzing your drawing..."):
                prediction = model.predict(img_normalized)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)

            # Display the result
            st.success(f"I'm **{confidence:.2%}** sure you drew a...")
            st.markdown(f"<p style='font-size: 100px; text-align: center; font-weight: bold;'>{predicted_digit}</p>", unsafe_allow_html=True)
            
    else:
        st.error("Model is not loaded. Cannot make a prediction.")

if clear_button:
    # A trick to clear the canvas is to re-render it with a new key or just let Streamlit's state handling do its thing.
    # Often, you don't need to do anything here if the button press causes a rerun.
    # If it doesn't clear, you can manage the canvas state more explicitly.
    st.rerun()