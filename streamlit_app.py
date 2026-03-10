import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch

# 1. Konfigurasi Halaman
st.set_page_config(page_title="YOLO Detection App", layout="wide")

st.title("🚀 Aplikasi Deteksi Objek YOLO")
st.write("Selamat datang di project pertama saya! Silakan unggah gambar untuk dideteksi.")

# 2. Load Model
# Pastikan file 'best.pt' ada di folder yang sama dengan file app.py ini
@st.cache_resource
def load_model():
    # 1. Daftarkan kelas ultralytics agar dianggap aman oleh PyTorch
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
    # 2. Muat model seperti biasa
    model = YOLO("best.pt")
    return model

try:
    model = load_model()
    st.success("Model 'best.pt' berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'best.pt' ada di folder yang sama. Error: {e}")

# 3. Sidebar untuk Pengaturan
st.sidebar.header("Pengaturan Model")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# 4. Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Konversi file ke format gambar PIL
    image = Image.open(uploaded_file)
    
    # Buat dua kolom untuk membandingkan Original vs Hasil
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Hasil Deteksi")
        
        # Jalankan Prediksi
        with st.spinner('Sedang mendeteksi...'):
            # Convert PIL image ke format yang dimengerti YOLO (numpy array)
            img_array = np.array(image)
            results = model.predict(source=img_array, conf=conf_threshold)

            # Ambil gambar hasil plot (bounding boxes)
            # results[0].plot() mengembalikan array gambar dengan kotak deteksi
            res_plotted = results[0].plot()
            
            # Tampilkan hasil
            st.image(res_plotted, channels="BGR", use_container_width=True)

    # 5. Tampilkan Informasi Tambahan
    st.divider()
    st.subheader("Detail Deteksi")
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            prob = float(box.conf[0])
            st.write(f"- Menemukan **{label}** dengan tingkat keyakinan **{prob:.2f}**")
    else:

        st.write("Tidak ada objek yang terdeteksi.")

