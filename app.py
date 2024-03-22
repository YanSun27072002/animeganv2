
import streamlit as st
from PIL import Image
import numpy as np

image_captured = st.camera_input("Capture", key="first_camera")
img_array=np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)  # Mảng ngẫu nhiên kích thước 100x100x3 (RGB)

if image_captured is not None:
    bytes_data = image_captured.getvalue()
    st.image(image_captured)
    img = Image.open(image_captured)
    img_array = np.array(img)
    print("arr",img_array)

    image = Image.fromarray(img_array)

    # Lưu ảnh dưới định dạng JPG
    image.save(r"D:\webpage\taianh\example.jpg")




