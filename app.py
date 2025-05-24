import streamlit as st
from PIL import Image
from handle.process_image import process_image
from handle.vnese_orc import prediction_ocr

st.set_page_config(page_title="Nhận diện chữ viết tay tiếng Việt", layout="centered")

st.title("📷 Nhận diện chữ viết tay tiếng Việt")


# Upload ảnh
uploaded_file = st.file_uploader("Tải ảnh lên", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên', use_container_width=True)

    if st.button("🔍 Nhận diện chữ"):
        processed_img = process_image(image)

        # Hiển thị ảnh đã xử lý
        st.image(processed_img, caption="Ảnh đã xử lý", use_container_width=True, channels="GRAY")

        prediction_txt = prediction_ocr(valid_img=processed_img)
        st.success(prediction_txt)
