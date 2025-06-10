import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import matplotlib.pyplot as plt

# ====== Ekstraksi Tekstur GLCM ======
def ekstraksi_fitur_glcm(image_gray):
    image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    glcm = graycomatrix(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    
    fitur = []
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in props:
        nilai = graycoprops(glcm, prop)
        fitur.extend(np.mean(nilai, axis=1))  # rata-rata dari 4 arah

    return np.array(fitur)

# ====== Ekstraksi Tekstur LBP ======
def ekstraksi_fitur_lbp(image_gray, radius=1, n_points=8):
    lbp = local_binary_pattern(image_gray, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                              bins=np.arange(0, n_points + 3),
                              range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return lbp.astype("uint8"), hist

# ====== Visualisasi Proses GLCM ======
def tampilkan_proses_glcm(image_bgr):
    asli = cv2.resize(image_bgr, (300, 300))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (300, 300))

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_resized = cv2.resize(thresh, (300, 300))

    kontur_img = image_bgr.copy()
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(kontur_img, contours, -1, (0, 255, 0), 2)
    kontur_img = cv2.resize(kontur_img, (300, 300))

    atas = np.hstack([asli, cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)])
    bawah = np.hstack([cv2.cvtColor(thresh_resized, cv2.COLOR_GRAY2BGR), kontur_img])
    gabungan = np.vstack([atas, bawah])
    return gabungan

# ====== Visualisasi Proses LBP ======
def tampilkan_proses_lbp(image_bgr, lbp_image):
    asli = cv2.resize(image_bgr, (300, 300))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (300, 300))
    lbp_vis = cv2.resize(lbp_image, (300, 300))

    gabung = np.hstack([
        cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(lbp_vis, cv2.COLOR_GRAY2BGR)
    ])
    return np.vstack([asli, gabung])

# ====== Proses Ekstraksi Dataset ======
def ekstrak_visual_dataset(folder_dataset, output_csv='fitur_tekstur_plastik.csv'):
    data_fitur = []
    folder_plastik = os.path.join(folder_dataset, 'plastik')

    if not os.path.exists(folder_plastik):
        print(f"[!] Folder tidak ditemukan: {folder_plastik}")
        return

    for nama_file in os.listdir(folder_plastik):
        path_gambar = os.path.join(folder_plastik, nama_file)
        img = cv2.imread(path_gambar)

        if img is None:
            print(f"[!] Gagal membaca: {path_gambar}")
            continue

        img = cv2.resize(img, (300, 300))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # === Ekstraksi GLCM ===
        fitur_glcm = ekstraksi_fitur_glcm(gray)

        # === Ekstraksi LBP ===
        lbp_img, fitur_lbp = ekstraksi_fitur_lbp(gray)

        # Gabungkan fitur
        fitur_total = list(fitur_glcm) + list(fitur_lbp) + [0]  # Label plastik = 0
        data_fitur.append(fitur_total)

        # === Tampilkan Visualisasi GLCM ===
        visual_glcm = tampilkan_proses_glcm(img)
        cv2.imshow("Proses GLCM - Sampah Plastik", visual_glcm)

        # === Tampilkan Visualisasi LBP ===
        visual_lbp = tampilkan_proses_lbp(img, lbp_img)
        cv2.imshow("Proses LBP - Sampah Plastik", visual_lbp)

        print(f"[•] {nama_file} — Tekan tombol untuk lanjut, ESC untuk keluar...")
        key = cv2.waitKey(0)
        if key == 27:
            print("[!] Proses dihentikan oleh pengguna.")
            break
        cv2.destroyAllWindows()

    # Simpan ke CSV
    kolom_glcm = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    kolom_lbp = [f'lbp_{i}' for i in range(len(fitur_lbp))]
    kolom_total = kolom_glcm + kolom_lbp + ['label']
    df = pd.DataFrame(data_fitur, columns=kolom_total)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Dataset fitur tekstur plastik disimpan ke: {output_csv}")

# ====== Jalankan Program ======
if __name__ == '__main__':
    ekstrak_visual_dataset('dataset')
