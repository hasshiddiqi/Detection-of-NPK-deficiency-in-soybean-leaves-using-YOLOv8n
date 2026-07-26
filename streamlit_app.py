import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import cv2

from huggingface_hub import hf_hub_download
import os

st.set_page_config(
    page_title="SMART AGRICULTURE VISION SYSTEM",
    layout="wide",
    initial_sidebar_state=None,
)

st.title("SMART AGRICULTURE VISION SYSTEM")



st.markdown("""
<style>
div[data-testid="stTabs"] > div[data-baseweb="tab-list"] {
    display: flex;
    justify-content: center;
    gap: 8px;
    background-color: #f0f2f0;
    padding: 6px;
    border-radius: 12px;
    width: fit-content;
    margin: 0 auto 20px auto;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(0, 0, 0, 0.05);
}

div[data-testid="stTabs"] button[data-baseweb="tab"] {
    height: 42px;
    border-radius: 8px;
    padding: 0 20px;
    background-color: transparent;
    border: none;
    color: #333;
    font-weight: 500;
    transition: all 0.2s ease;
}

div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #1B4332;
    color: white;
}

div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
    display: none;
}

            
</style>
""", unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(["Prediksi", "Karakteristik Visual Defisiensi", "Tentang Model"])
with tab1:
    st.markdown("""
    ### 🌱 Apa itu SAVIS?
    SAVIS membantu Anda memeriksa kesehatan daun kedelai hanya dengan foto.

    **Cara pakai:**
    1. 📷 Ambil atau unggah foto daun kedelai yang jelas
    2. 🔍 Sistem akan otomatis memeriksa tanda kekurangan unsur hara
    3. 💡 Anda akan mendapat penjelasan dan saran pupuk yang sesuai

    Tidak perlu keahlian khusus — cukup foto, dan SAVIS membantu Anda memahami kondisi tanaman.
    """)
    st.divider()
        
    #Kotak unggah foto
    st.markdown("""
    <style>
    [data-testid="stFileUploader"] {
        border: 2px dashed #1B4332;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Seret & lepas foto di sini atau tekan untuk memilih", 
        type=["jpg", "jpeg", "png"],
        max_upload_size=20,
        accept_multiple_files=False
    )

    st.write("atau gunakan kamera")
    
    enable = st.checkbox("Izinkan kamera")
    camera_photo = st.camera_input(
        "Ambil gambar",
        key=None, 
        help=None,
        disabled=not enable
    )

    image_source = uploaded_file or camera_photo
      
    # Pastikan file 'best.pt' ada di folder yang sama dengan file app.py ini
    @st.cache_resource 
    def load_model():
        model_path = hf_hub_download(
            repo_id="NewbieFinalBosss/SAVIS",
            filename="best.onnx")
        model = YOLO(model_path)
        return model
    
    
    model = load_model()
     
    if image_source is not None:
        # Konversi file ke format gambar PIL
        image = Image.open(image_source).convert("RGB")
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
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # konversi ke BGR
                results = model.predict(source=img_array, conf=0.90, imgsz=896, task='detect')
                # Ambil gambar hasil plot (bounding boxes)
                # results[0].plot() mengembalikan array gambar dengan kotak deteksi
                label_translation = {
                "Nutrient Sufficient": "Hara Tercukupi",
                "N Deficiency": "Defisiensi N",
                "P Deficiency": "Defisiensi P",
                "K Deficiency": "Defisiensi K",
                }
                results[0].names = {
                k: label_translation.get(v, v) for k, v in results[0].names.items()
                }
                res_plotted = results[0].plot()
                # Tampilkan hasil
                st.image(res_plotted, channels="BGR", use_container_width=True)

        #Tampilkan Informasi Tambahan
        st.divider()

        info_map = {
    "Nutrient Sufficient": {
        "keterangan": "✅ Tanaman Anda memiliki hara yang tercukupi.",
        "solusi": "Pertahankan pola pemupukan saat ini sesuai dosis anjuran setempat."
    },
    "N Deficiency": {
        "keterangan": "⚠️ Tanaman Anda mengalami kekurangan unsur **Nitrogen (N)**.",
        "solusi": "Gunakan pupuk mengandung nitrogen seperti Urea atau ZA. Konsultasikan dosis dan waktu aplikasi yang tepat dengan penyuluh pertanian setempat, karena kebutuhan bisa berbeda tergantung kondisi lahan."
    },
    "P Deficiency": {
        "keterangan": "⚠️ Tanaman Anda mengalami kekurangan unsur **Fosfor (P)**.",
        "solusi": "Gunakan pupuk mengandung fosfor seperti SP-36 atau TSP. Konsultasikan dosis dan waktu aplikasi yang tepat dengan penyuluh pertanian setempat, karena kebutuhan bisa berbeda tergantung kondisi lahan."
    },
    "K Deficiency": {
        "keterangan": "⚠️ Tanaman Anda mengalami kekurangan unsur **Kalium (K)**.",
        "solusi": "Gunakan pupuk mengandung kalium seperti KCl atau ZK. Konsultasikan dosis dan waktu aplikasi yang tepat dengan penyuluh pertanian setempat, karena kebutuhan bisa berbeda tergantung kondisi lahan."
    },
}

        if len(results[0].boxes) > 0:
            detected_classes = set()
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                detected_classes.add(model.names[class_id])

            # Kalau ada defisiensi terdeteksi bersamaan dengan "Nutrient Sufficient",
            # sembunyikan "Nutrient Sufficient" supaya tidak membingungkan petani
            deficiency_classes = {"N Deficiency", "P Deficiency", "K Deficiency"}
            if detected_classes & deficiency_classes:
                detected_classes.discard("Nutrient Sufficient")

            st.subheader("Hasil Pemeriksaan")
            for label in detected_classes:
                info = info_map.get(label)
                if info:
                    st.warning(info["keterangan"])
                    st.write(f"💡 **Solusi:** {info['solusi']}")
                    st.write("")

with tab2:
    st.header("Karakteristik Visual Daun Hara Tercukupi", divider="blue")
    st.write("Daun yang tercukupi haranya menampilkan warna hijau merata tanpa perubahan warna abnormal")
    st.image("Assets/Hara Tercukupi.jpg", width=600)
    st.write("")
    st.write("")
    st.write("")

    st.header("Karakteristik Visual Daun Defisiensi Nitrogen", divider="blue")
    st.write("Defisiensi nitrogen menyebabkan tanaman tumbuh kerdil dengan daun yang sempit. Gejala klorosis dimulai dari daun tua karena nitrogen diremobilisasi ke daun muda untuk pertumbuhan. Secara visual, tanaman tampak hijau pucat atau kuning (Marschner, 2012).")
    st.image("Assets/Defisiensi N.jpg", width=600)
    st.write("")
    st.write("")
    st.write("")

    st.header("Karakteristik Visual Daun Defisiensi Fosfor", divider="blue")
    st.write("Defisiensi fosfor menghambat pertumbuhan tanaman sehingga daun yang tumbuh relatif sedikit. Gejala visual dimulai dari daun tua, berupa klorosis antartulang daun dan nekrosis yang berkembang menyatu di sepanjang tepi daun (Yara Canada, 2018).")
    st.image("Assets/Defisiensi P.jpg", width=600)
    st.write("")
    st.write("")
    st.write("")

    st.header("Karakteristik Visual Daun Defisiensi Kalium", divider="blue")
    st.write("Defisiensi kalium menyebabkan klorosis antarvena yang dimulai pada daun tua, sementara vena utama tetap berwarna hijau untuk sementara. Klorosis kemudian berlanjut ke arah pangkal daun dan diikuti munculnya nekrosis di bagian tepi daun (Yara Canada, 2018).")
    st.image("Assets/Defisiensi K.jpg", width=600)
    st.write("")
    st.write("")
    st.write("")

    st.header("Referensi")
    st.write("Marschner, P. (2012). Marschner's Mineral Nutrition of Higher Plants (Third Edition). Academic Press. DOI:10.1016/C2009-0-63043-9")
    st.write("Yara Canada. (2018). Nutrient Deficiencies in Soybean. Diakeses 17 Mei 2026, dari https://www.yaracanada.ca/crop-nutrition/soybean/nutrient-deficiencies/")
    
with tab3:
    st.header("Tentang Model", divider="blue")
    st.subheader("Arsitektur dan Konfigurasi")
    st.write("Arsitektur: YOLOv8m")
    st.write("Versi Ultralytics: 8.4.24")
    st.write("Versi PyTorch: 2.12.0+cu132")
    st.write("Perangkat: GPU Laptop NVIDIA® GeForce RTX™ 5070")
    st.write("CUDA: 13.2")
    st.write("cuDNN: 9.20.0.48 ")
    st.write("hiperparameter: imgsz=896, batch=8, epochs=100 (early stopping pada epoch 79), patience=20")
    st.write("")
    st.write("")
    
    st.subheader("Dataset")
    st.write("Kelas: Hara Tercukupi, Defisiensi N, Defisiensi P, dan Defisiensi K")
    st.write("Citra digital tiap kelas: 300")
    st.write("Rasio train/val/test: 70:15:15")
    st.write("")
    st.write("")

    st.subheader("Metrik Performa Validasi")
    st.write(
        pd.DataFrame(
            {
                "Kelas": ["Hara Tercukupi", "Defisiensi N", "Defisiensi P", "Defisiensi K", "Semua"],
                "Citra": [43, 43, 43, 51, 198],
                "Anotasi": [185, 203, 155, 210, 753],
                "Precision": [1.000, 0.969, 0.998, 0.990, 0.989],
                "Recall": [0.962, 0.961, 1.000, 1.000, 0.981],
                "mAP50": [0.995, 0.992, 0.995, 0.995, 0.994],
                "mAP50-95": [0.985, 0.979, 0.995, 0.993, 0.988],
            }
        )
    )
    st.write("")
    st.write("")

    st.subheader("Kurva Precision-Recall")
    st.image("Assets/BoxPR_curve.png", width=600)
    st.write("")
    st.write("")

    st.subheader("Kurva Skor F1")
    st.image("Assets/BoxF1_curve.png", width=600)
    st.write("")
    st.write("")

    st.subheader("Confusion Matrix")
    st.write("Non-normalized")
    st.image("Assets/confusion_matrix.png", width=600)
    st.write("")
    st.write("Normalized")
    st.image("Assets/confusion_matrix_normalized.png", width=600)
            
    





