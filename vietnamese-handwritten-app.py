import streamlit as st
from PIL import Image
import torch
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from load_css import local_css
# Function to load model based on selected version and weights
def load_model(model_version, weights_file):
    # Load the trained model config
    config_path = f'./configs/{model_version}.yml'
    config = Cfg.load_config_from_file(config_path)

    # Load the trained model weights
    weights_path = f'./weights/{weights_file}.pth'
    config['weights'] = weights_path
    config['device'] = 'cpu'  # Set device to CPU

    detector = Predictor(config)
    return detector

device = torch.device("cpu")

def main():
    st.set_page_config(page_title="Vietnamese Handwritten By Line App",layout="wide")
    local_css("./styles.css")

    st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("https://images.unsplash.com/photo-1513708929605-6dd0e1b081bd?q=80&w=1476&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    st.markdown('<link rel="stylesheet" href="styles.css">', unsafe_allow_html=True)
    
    st.write(
        """
        <p style="text-align: center; font-size: 45px;">
            <span class="border-highlight"><strong>
            <span class="title-word title-word-1">Vietnamese</span>
            <span class="title-word title-word-2">Handwritten </span>
            <span class="title-word title-word-3">Recognition ✍🏻</span></strong>
            </span>
        </p> 
        """,
        unsafe_allow_html=True
        )
    st.write(
            """
            <p style="text-align: center; color: #dee2e6; font-size: 20px;"><i>
                Đây là ứng dụng demo cho đề tài nhận diện chữ viết tay cấp độ theo dòng trong môi trường Tiếng Việt.<br>
                <hr style='margin: 1rem'>
                This is a demo application for the project Vietnamese Handwritten Recognition. <br>
                <strong><u>Authors:</strong></u>Võ Khánh Linh <br>
            </i></p> 
            """,
            unsafe_allow_html=True
        )

    # Select box for model version
    model_version = st.selectbox('Hãy chọn phiên bản của mô hình hình(Select Model Version):', ['config_v3', 'config_v1'])
    # Select box for weights
    weights_files = ['transformerocr_v3', 'transformerocr_v1', 'transformerocr_test_wb']
    weights_file = st.selectbox('Hãy chọn trọng số cho mô hình (Select Weights):', weights_files)

    # Streamlit UI components
    uploaded_file = st.file_uploader("Tải ảnh cần nhận diện (Upload an image):", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='', use_column_width=True)

        # Load the selected model and weights
        detector = load_model(model_version, weights_file)

        # Function to perform inference
        #@st.cache_data
        def recognize_text(_image):
            recognized_text = detector.predict(_image, return_prob=False)
            return recognized_text

        # Perform inference on the uploaded image
        recognized_text = recognize_text(image)
        st.write(
            """
            <p style="text-align: left; color: #dee2e6;">
            <strong><u>Chữ viết trên ảnh là (Recognized Text): </strong></u>
            </p> 
            """,
            unsafe_allow_html=True
        )
        st.write(recognized_text)
        

if __name__ == "__main__":
    main()
