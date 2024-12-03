import streamlit as st
import requests
from PIL import Image
import numpy as np
import io

# API url in Google cloud
API_URL = "https://hem-detect-98537971591.europe-west1.run.app/prediction/"

def preprocess_image(image):
    """
    Preprocess the uploaded image to be compatible with the API.
    Resizes the image to 150x150 and converts it to grayscale if needed.
    """
    image = image.resize((150, 150))
    return np.array(image)

def send_image_to_api(image):
    """
    Send the preprocessed image to the FastAPI endpoint and get predictions.
    """
    # image_bytes = io.BytesIO()
    # image.save(image_bytes, format="PNG")  # Guardar como PNG
    # image_bytes.seek(0)  # Ir al inicio del flujo

    # # Send request to API
    # files = {"file": (image_bytes, "image/png")}
    # response = requests.post(API_URL, files=files)
    # Open the file and send it as a request
    image_path= "out.png"
    with open(image_path, 'rb') as file:
        images = {'file': (image_path, file, 'image/png')}
        response = requests.post(API_URL, files=images)


    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
        return None

def main():
    st.title("📊 Injury Prediction Tool")

    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. Upload an image file (e.g., JPG, PNG).
    2. Click 'Analyze' to get the AI prediction.
    3. View the results below.
    """)

    # File uploader for image
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        image.save('out.png')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # # Preprocess the image
        # image_array = preprocess_image(image)

        if st.button("Analyze"):
            with st.spinner("Analyzing the image..."):
                # Send the image to the API
                result = send_image_to_api(image)

                if result:
                    # Display the prediction result
                    st.success(f"Prediction Result: {result['injury']:.2f}")
                else:
                    st.error("Failed to get a prediction from the API.")

    st.markdown("---")
    st.markdown("### About")
    st.text("This tool uses a machine learning model to predict injury types from uploaded images.")

if __name__ == "__main__":
    main()
