import streamlit as st
import requests
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define API endpoint
API_ENDPOINT = "http://127.0.0.1:8000/prediction/"

def load_dicom_image(dicom_file):
    """
    Load and convert DICOM file to displayable image.
    """
    try:
        dicom_data = pydicom.dcmread(dicom_file)
        image_array = dicom_data.pixel_array
        image_normalized = ((image_array - image_array.min()) /
                            (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        return image_normalized
    except Exception as e:
        st.error(f"Error processing DICOM file: {e}")
        return None

def display_dicom_image(image_array):
    """
    Display the DICOM image using Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_array, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

def get_prediction_from_api(file):
    """
    Send the file to the FastAPI backend and get the prediction.
    """
    try:
        response = requests.post(
            API_ENDPOINT,
            files={"file": file}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Brain CT Scan Analyzer",
        page_icon=":brain:",
        layout="wide"
    )
    st.title("🧠 Brain CT Scan Analyzer")

    # Sidebar instructions
    st.sidebar.header("📋 Instructions")
    st.sidebar.markdown("""
    1. Upload a Brain CT Scan (.dcm file)
    2. View the medical image
    3. Get AI-powered analysis
    """)

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload DICOM Brain CT Scan",
        type=['dcm'],
        help="Upload a DICOM format brain CT scan"
    )

    # Display the uploaded image and prediction
    if uploaded_file is not None:
        # Display the DICOM image
        dicom_image = load_dicom_image(uploaded_file)
        if dicom_image is not None:
            st.subheader("📸 Brain CT Scan")
            display_dicom_image(dicom_image)

            # Send to API for prediction
            st.subheader("🤖 AI Analysis")
            if st.button("Analyze Scan"):
                with st.spinner("Analyzing the scan..."):
                    uploaded_file.seek(0)  # Reset file pointer
                    prediction = get_prediction_from_api(uploaded_file)
                if prediction:
                    st.success(f"Prediction: {prediction.get('injury_probability', 'N/A') * 100:.2f}% chance of injury.")

    # Footer
    st.markdown("---")
    st.markdown("""
    ### ℹ️ About This Tool
    - This tool is a demonstration of medical image analysis.
    - For accurate diagnoses, always consult medical professionals.
    """)

if __name__ == "__main__":
    main()
