# 🎈 Object Detection and Classification with Feature Extraction

Program ini melakukan deteksi dan klasifikasi objek berdasarkan fitur warna, bentuk, dan tekstur dari area yang dipilih pada citra menggunakan model machine learning (SVM atau lainnya) yang telah dilatih sebelumnya.

---

## 🔧 Cara Penggunaan 
```
// Cloning peroject nya

git clone https://github.com/hoesnial/Dayeuh-Waste-Classifier.git
```
```
// Install Library-library yang akan digunakan satu persatu

pip install opencv-python
pip install matplotlib
pip install scikit-image
pip install numpy
pip instakk pandas
```
---
## 👩‍👧‍👦 Kelompok C2 
- 152023088 Muhamad Husni 
- 152023114 Rizal Shiddieq
- 152023172 SAEPPUDIN

---

## 🎗️ Fitur Utama
- Ekstraksi fitur warna untuk objek kertas (LAB)

- Ekstraksi fitur bentuk untuk objek logam (laplacian)

- Ekstraksi fitur tekstur untuk objek plastik (glcm)

---

## 🎯 Tujuan Project
Tujuan dari proyek ini adalah untuk membangun sebuah sistem klasifikasi objek pada citra gambar berbasis ekstraksi fitur warna, bentuk, dan tekstur, yang dapat digunakan untuk:

- Mengidentifikasi jenis objek (misalnya: plastik, kertas, organik) dalam sebuah gambar secara otomatis.

- Mengembangkan pipeline sederhana namun efektif untuk sistem klasifikasi citra berbasis pembelajaran mesin.

- Memberikan antarmuka interaktif yang memungkinkan pengguna memilih area objek secara manual (ROI) untuk analisis langsung.
