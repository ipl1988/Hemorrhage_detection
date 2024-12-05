import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import base64

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



# Function to encode an image as base64
def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
        return encoded_string

# # # Function for background to be called at in main function
# background_path= "background.png"
# def set_background(background_path):
#     """
#     Sets a background image for the Streamlit app using CSS.
#     """
#     with open(background_path, "rb") as img_file:
#         encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
# st.markdown(
# f"""
# <style>
# .st-emotion-cache-1r4qj8v {{
#     background-image: url("background.png");
#     background: rgb(188 113 113);
#     background-size: cover;
#     background-repeat: no-repeat;
#     background-attachment: fixed;
# }}
# </style>
# """,
# unsafe_allow_html=True
# )

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as file:
        bin_content = file.read()
    return base64.b64encode(bin_content).decode('utf-8')

bin_str = get_base64_of_bin_file("background.png")

st.markdown(
    f"""
    <style>
    .st-emotion-cache-1yiq2ps {{
        background-image: url("data:image/png;base64,{bin_str}") !important;
        background: rgb(188 113 113);
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Main app function
def main():
    # # Set the background
    # set_background(background_path)

    # Main container for content
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Title and subtitle
    st.markdown('<h1 class="header-title">Hemorrhage Detection Assistant</h1>', unsafe_allow_html=True)
    st.write("This AI-powered assistant helps analyze brain CT scans for signs of hemorrhage.")

    # Load and display sidebar image
    img_path = "brain.png"
    img_base64 = img_to_base64(img_path)
    if img_base64:
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
            unsafe_allow_html=True,
        )

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
                result = send_image_to_api(image)

                if result is not None:
                    if result==1:
                        st.success("Positive: high probability to find a hemorrhage in this image.")
                    else:
                        st.success("Negative: high probability to find a hemorrhage in this image.")
                else:
                    st.error("Failed to retrieve a prediction.")
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")

    st.markdown("---")
    st.markdown("### About")
    st.text("This tool uses a CNN deep learning model to predict injury types from uploaded images.")

if __name__ == "__main__":
    main()
