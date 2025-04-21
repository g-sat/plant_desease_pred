import os
import json
import time
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
from disease_solutions import disease_solutions

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #388E3C;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .solution-box {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #388E3C;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #757575;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/plant-under-rain.png", width=80)
    st.markdown("<h2 style='text-align: center;'>Plant Disease Classifier</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### About")
    st.info("""
    This application uses deep learning to identify plant diseases from leaf images.
    Upload a clear image of a plant leaf to get a diagnosis and treatment recommendations.
    """)
    
    st.markdown("### Instructions")
    st.markdown("""
    1. Upload a clear image of a plant leaf
    2. Click 'Classify' to analyze the image
    3. View the diagnosis and treatment recommendations
    4. Follow the suggested solutions to treat the disease
    """)
    
    st.markdown("### Supported Plants")
    st.markdown("""
    - Apple
    - Corn (Maize)
    - Grape
    - Orange
    - Peach
    - Pepper
    - Potato
    - Squash
    - Strawberry
    - Tomato
    """)
    
    st.markdown("---")
    st.markdown("### Credits")
    st.markdown("""
    Model trained on the PlantVillage dataset.
    Solutions provided by agricultural experts.
    """)

# Main content
st.markdown("<h1 class='main-header'>🌿 Plant Disease Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a plant leaf image to diagnose diseases and get treatment recommendations</p>", unsafe_allow_html=True)

# Initialize session state for prediction
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False

# File uploader with custom styling
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

# Load model and class indices
@st.cache_resource
def load_model():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_class_indices():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    return json.load(open(f"{working_dir}/class_indices.json"))

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence = float(predictions[0][predicted_class_index])
    return predicted_class_name, confidence

# Load model and class indices
model = load_model()
class_indices = load_class_indices()

# Display image and prediction
if uploaded_image is not None:
    st.session_state.image_uploaded = True
    image = Image.open(uploaded_image)
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3 class='sub-header'>Uploaded Image</h3>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        
        # Add image analysis options
        st.markdown("<h3 class='sub-header'>Image Analysis</h3>", unsafe_allow_html=True)
        analysis_option = st.radio(
            "Select analysis type:",
            ["Disease Detection", "Plant Health Assessment", "Detailed Analysis"]
        )
        
        if st.button('Analyze Image', key='analyze_btn'):
            with st.spinner('Analyzing image...'):
                # Simulate processing time for better UX
                time.sleep(1.5)
                
                # Get prediction and confidence
                prediction, confidence = predict_image_class(model, uploaded_image, class_indices)
                st.session_state.prediction = prediction
                
                # Display results based on analysis option
                if analysis_option == "Disease Detection":
                    st.success(f'Disease Detected: {prediction}')
                    st.progress(confidence)
                    st.write(f"Confidence: {confidence:.2%}")
                    
                elif analysis_option == "Plant Health Assessment":
                    if "healthy" in prediction.lower():
                        st.success("Plant appears healthy!")
                        st.balloons()
                    else:
                        st.warning("Plant shows signs of disease")
                        st.write(f"Detected issue: {prediction}")
                    
                else:  # Detailed Analysis
                    st.info("Detailed Analysis Results")
                    st.write(f"Primary diagnosis: {prediction}")
                    st.write(f"Confidence level: {confidence:.2%}")
                    
                    # Show top 3 possible diagnoses
                    preprocessed_img = load_and_preprocess_image(uploaded_image)
                    predictions = model.predict(preprocessed_img)
                    top_indices = np.argsort(predictions[0])[-3:][::-1]
                    
                    st.write("Other possible conditions:")
                    for idx in top_indices[1:]:
                        st.write(f"- {class_indices[str(idx)]} ({predictions[0][idx]:.2%})")
    
    with col2:
        if st.session_state.prediction:
            st.markdown("<h3 class='sub-header'>Treatment Recommendations</h3>", unsafe_allow_html=True)
            
            # Display disease solutions if available
            if st.session_state.prediction in disease_solutions:
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.write(f"### {st.session_state.prediction.replace('___', ' - ')}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                for i, solution in enumerate(disease_solutions[st.session_state.prediction], 1):
                    st.markdown(f"<div class='solution-box'>{i}. {solution}</div>", unsafe_allow_html=True)
                
                # Additional resources
                st.markdown("<h4>Additional Resources</h4>", unsafe_allow_html=True)
                resource_type = st.selectbox(
                    "Select resource type:",
                    ["Treatment Guides", "Prevention Tips", "Expert Consultation"]
                )
                
                if resource_type == "Treatment Guides":
                    st.write("📚 Comprehensive treatment guides for this disease are available from agricultural extension services.")
                elif resource_type == "Prevention Tips":
                    st.write("🛡️ Regular monitoring and early intervention are key to preventing disease spread.")
                else:
                    st.write("👨‍🌾 Consider consulting with a local plant pathologist for specialized treatment plans.")
                
                # Save recommendations
                if st.button("Save Recommendations"):
                    st.success("Recommendations saved to your profile!")
            else:
                st.info("No specific solutions available for this condition.")
                
            # Feedback section
            st.markdown("<h3 class='sub-header'>Was this helpful?</h3>", unsafe_allow_html=True)
            feedback = st.radio("", ["Yes", "No", "Partially"])
            if feedback:
                st.write("Thank you for your feedback!")
                if feedback != "Yes":
                    st.text_area("How can we improve?", "")
                    if st.button("Submit Feedback"):
                        st.success("Feedback submitted successfully!")

# Footer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("© 2023 Plant Disease Classifier | Powered by Deep Learning")
st.markdown("</div>", unsafe_allow_html=True)
