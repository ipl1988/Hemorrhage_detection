import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import base64

API_URL = "https://hem-detect-98537971591.europe-west1.run.app/prediction/"
def main():
    # # Set the background
    # set_background(background_path)

    # Main container for content
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Title and subtitle
    st.markdown('<h1 class="eheader-title">Hemorrhage Detection Assistant</h1>', unsafe_allow_html=True)
    st.write("This AI-powered assistant helps analyze brain CT scans for signs of hemorrhage.")


    # Insert custom CSS for glowing effect in sidebar image
    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow:
                0 0 5px #2A0033,
                0 0 10px #4B007A,
                0 0 15px #660099,
                0 0 20px #8000CC,
                0 0 25px #9900FF,
                0 0 30px #9933FF,
                0 0 35px #FF66FF;
            position: relative;
            z-index: -1;
            border-radius: 45px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    st.sidebar.markdown("---")


    # Sidebar instructions
    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. Upload an image file (PNG).
    2. Click 'Analyze' to get the AI prediction.
    3. View the results below.
    """)


    # File uploader for brain scans
    uploaded_file = st.file_uploader("Upload a Brain CT Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        try:
        # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Brain CT Image")

        # Button for analysis
            if st.button("Analyze"):
             with st.spinner("Analyzing the image..."):
                # Send the image to the API
                   img = uploaded_file.getvalue()
                   files = {"file": ("image.png", img, "image/png")}
                   response = requests.post(API_URL, files=files)
                   result=response.json()["injury"]
                   if result is not None:
                        if result>0.5 :
                            st.success(f'Positive: high probability({round(result,2)}) to find a hemorrhage in this image.')
                        else:
                            st.success(f'Negative: low probability({round(result,2)}) to find a hemorrhage in this image.')
            else:
                st.error("Failed to retrieve a prediction.")
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")

    st.markdown("---")
    st.markdown("### About")
    st.text("This tool uses a CNN deep learning model to predict injury types from uploaded images.")

if __name__ == "__main__":
    main()
