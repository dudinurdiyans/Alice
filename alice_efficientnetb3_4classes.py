import streamlit as st
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import gdown
from io import BytesIO
import zipfile

# === Konfigurasi halaman ===
st.set_page_config(page_title="Klasifikasi X-ray Paru-paru", layout="centered")

# === Judul Aplikasi ===
st.title("🩻 ALICE: Automated Lung Image Classification and Evaluation ")
st.write("Klasifikasi Penyakit Paru-paru dari Citra X-Ray")

# === Unduh Model dari Google Drive ===
MODEL_PATH = "EffNetB3_CLAHE3.keras"
GDRIVE_FILE_ID = "114NJu53mzUS31njrCWSrlaDW15prNyTZ"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Mengunduh model dari Google Drive..."):
        gdown.download(id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=False)

# === Load Model ===
model = load_model(MODEL_PATH)

# === Label Kelas ===
CLASS_NAMES = ['Corona Virus Disease', 'Normal', 'Pneumonia', 'Tuberculosis']

# === Fungsi Prediksi ===
def predict(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds)) * 100
    return CLASS_NAMES[class_idx], confidence, img_array

# === Sidebar Dokumentasi ===
with st.sidebar:
    st.title("📄 Dokumentasi")
    st.markdown("""
    **Kelas yang dideteksi:**
    - Corona Virus Disease
    - Normal
    - Pneumonia
    - Tuberculosis

    **Cara pakai:**
    1. Unggah gambar X-ray.
    2. Klik tombol **Prediksi**.
    3. Lihat hasil prediksi.

    _Model: EffNetB3_CLAHE (EfficientNetB3 + CLAHE)_
    """)

# === Logging prediksi ===
log_data = []

def log_prediction(filename, label, confidence):
    log_data.append({"filename": filename, "label": label, "confidence": confidence})
    df = pd.DataFrame(log_data)
    df.to_csv('predictions_log.csv', index=False)

# === Upload Gambar Tunggal ===
uploaded_file = st.file_uploader("Unggah gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar X-ray yang diunggah", use_container_width=True)

    if st.button("🔍 Prediksi"):
        label, confidence, _ = predict(img)
        st.success(f"**Hasil Prediksi: {label} ({confidence:.2f}%)**")

        log_prediction(uploaded_file.name, label, confidence)

        # Tampilkan histori
        st.subheader("📋 Histori Prediksi")
        if os.path.exists('predictions_log.csv'):
            log_df = pd.read_csv('predictions_log.csv')
            st.dataframe(log_df)

# === Batch Prediksi (ZIP) ===
st.subheader("📂 Batch Prediksi (ZIP)")
batch_file = st.file_uploader("Unggah file ZIP yang berisi gambar X-ray", type=["zip"])

if batch_file is not None:
    with zipfile.ZipFile(BytesIO(batch_file.read())) as archive:
        image_files = [f for f in archive.namelist() if f.endswith(('jpg', 'jpeg', 'png'))]
        for image_file in image_files:
            with archive.open(image_file) as img_file:
                img = Image.open(img_file).convert("RGB")
                label, confidence, _ = predict(img)
                st.write(f"{image_file}: {label} ({confidence:.2f}%)")
