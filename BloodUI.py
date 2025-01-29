import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model (Ensure the model is saved as 'blood_group_fingerprint_model.h5')
model = load_model(r'blood_group_fingerprint_model2.h5')

# Define the blood group classes (ensure these match the classes the model was trained on)
blood_group_classes = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Streamlit UI
st.title("Blood Group Prediction from Fingerprint")

# Accept image file upload (allowing BMP, PNG, JPG, and JPEG formats)
uploaded_file = st.file_uploader("Upload Fingerprint Image (BMP, PNG, JPG, JPEG)", type=["bmp", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read and process the uploaded BMP image (or other accepted formats)
    img = load_img(uploaded_file, target_size=(128, 128))  # Resize the image to 128x128
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    # Predict the blood group using the trained model
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_blood_group = blood_group_classes[predicted_class_index]

    # Display the result
    st.image(img, caption="Uploaded Fingerprint Image", use_column_width=True)
    st.write(f"Predicted Blood Group: {predicted_blood_group}")
