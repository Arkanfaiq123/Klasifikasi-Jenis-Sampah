import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Load model
model = load_model("garbage_classifier_model.h5")

# Daftar label sesuai dataset Anda
class_names = ['cardboard', 'plastic', 'paper', 'glass', 'metal', 'trash']

# Konfigurasi Streamlit
st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("♻️ Garbage Image Classifier")
st.write("Upload gambar sampah untuk mengklasifikasikannya ke dalam salah satu kategori.")

# Upload gambar
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = load_img(uploaded_file, target_size=(150, 150))
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 150, 150, 3)

    # Prediksi
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]

    # Tampilkan hasil
    st.markdown(f"### 🔍 Hasil Prediksi: `{predicted_label}`")
