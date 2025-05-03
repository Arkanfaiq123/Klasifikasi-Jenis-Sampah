import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load model
model = load_model('garbage_classifier_model.h5')

# Label kelas (urutan harus sesuai dengan flow_from_directory)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Fungsi prediksi gambar
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    print(f"Prediksi: {predicted_class} ({confidence * 100:.2f}%)")

# Contoh penggunaan
if __name__ == "__main__":
    # Ganti dengan path ke gambar uji kamu
    test_image_path = 'test_images/sample.jpg'
    
    if os.path.exists(test_image_path):
        predict_image(test_image_path)
    else:
        print(f"File '{test_image_path}' tidak ditemukan.")
