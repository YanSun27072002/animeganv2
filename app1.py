

import onnxruntime as ort
import streamlit as st
from PIL import Image
import os, cv2
import numpy as np
pic_form = ['.jpeg','.jpg','.png','.JPEG','.JPG','.PNG']
from glob import glob

def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def process_image(img, x32=True):
    h, w = img.shape[:2]
    if x32: 
        def to_32s(x):
            return 256 if x < 256 else x - x%32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/ 127.5 - 1.0
    return img

def load_test_data(image_path, size):
    img0 = cv2.imread(image_path).astype(np.float32)
    img = process_image(img0, size)
    img = np.expand_dims(img, axis=0)
    return img, img0.shape

def save_images(images, image_path, size):
    images = (np.squeeze(images) + 1.) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    images = cv2.resize(images, size)
    cv2.imwrite(image_path, cv2.cvtColor(images, cv2.COLOR_RGB2BGR))

def Convert(input_imgs_path, output_path, onnx ="model.onnx", img_size=[256,256]):
    result_dir = output_path
    check_folder(result_dir)
    test_files = glob('{}/*.*'.format(input_imgs_path))
    test_files = [ x for x in test_files if os.path.splitext(x)[-1] in pic_form]
    try:
        session = ort.InferenceSession(onnx, None)
    except Exception as e:
        print(f"Error loading the ONNX model: {e}")
    # ------------
    x = session.get_inputs()[0].name
    y = session.get_outputs()[0].name

    print(test_files)
    print(type(test_files))
    for i, sample_file  in enumerate(test_files) :
        sample_image, shape = load_test_data(sample_file, img_size)
        image_path = os.path.join(result_dir,'{0}'.format(os.path.basename(sample_file)))
        fake_img = session.run(None, {x : sample_image})
        save_images(fake_img[0], image_path, (shape[1], shape[0]))
if __name__ == '__main__':
    onnx_file = r"D:\webpage\Shinkai_53.onnx"
    input_imgs_path = r'D:\webpage\input'
    output_path = r'D:\webpage\output'
     
    image_captured = st.camera_input("Capture", key="first_camera")
    if image_captured is not None:
        bytes_data = image_captured.getvalue()
        # st.image(image_captured)
        img = Image.open(image_captured)
        img_array = np.array(img)
        image = Image.fromarray(img_array)
        image.save(r"D:\webpage\input\example.jpg")
        Convert(input_imgs_path, output_path, onnx_file)
        final_output = "D:\webpage\output\example.jpg"
        st.image(final_output, caption='Result', use_column_width=True)


