import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# --- GLCM Ekstraksi (peningkatan: banyak jarak & rata-rata global) ---
def ekstraksi_fitur_tekstur_glcm(image_gray):
    image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image_gray, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)

    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    fitur = []
    for prop in props:
        nilai = graycoprops(glcm, prop)
        fitur.append(np.mean(nilai))  # rata-rata seluruh kombinasi
    return np.array(fitur)

# --- LBP Ekstraksi (peningkatan: P=16, R=2, uniform histogram lebih deskriptif) ---
def ekstraksi_fitur_lbp(image_gray):
    P, R = 16, 2
    lbp = local_binary_pattern(image_gray, P, R, method='uniform')
    n_bins = P + 2  # Uniform pattern
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalisasi
    return hist, lbp.astype('uint8')

# --- Visualisasi Proses GLCM ---
def tampilkan_proses_glcm(image_bgr):
    asli = cv2.resize(image_bgr, (300, 300))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (300, 300))

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_resized = cv2.resize(thresh, (300, 300))

    kosong = np.zeros_like(asli)
    atas = np.hstack([asli, cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)])
    bawah = np.hstack([cv2.cvtColor(thresh_resized, cv2.COLOR_GRAY2BGR), kosong])
    return np.vstack([atas, bawah])

# --- Visualisasi Proses LBP ---
def tampilkan_proses_lbp(image_bgr, lbp_image):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (300, 300))
    lbp_resized = cv2.resize(lbp_image, (300, 300))
    asli_resized = cv2.resize(image_bgr, (600, 300))
    gabung = np.hstack([
        cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(lbp_resized, cv2.COLOR_GRAY2BGR)
    ])
    return np.vstack([asli_resized, gabung])

# --- Proses Dataset ---
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

        fitur_glcm = ekstraksi_fitur_tekstur_glcm(gray)
        fitur_lbp, lbp_img = ekstraksi_fitur_lbp(gray)

        fitur_total = np.concatenate([fitur_glcm, fitur_lbp])
        data_fitur.append(list(fitur_total) + [0])  # Label plastik = 0

        visual_glcm = tampilkan_proses_glcm(img)
        visual_lbp = tampilkan_proses_lbp(img, lbp_img)

        cv2.imshow("Proses Ekstraksi Tekstur GLCM", visual_glcm)
        cv2.imshow("Proses Ekstraksi Tekstur LBP", visual_lbp)
        print(f"Tampilkan: {nama_file} — Tekan tombol apa pun untuk lanjut...")
        key = cv2.waitKey(0)
        if key == 27:
            print("[!] Dihentikan oleh pengguna.")
            break
        cv2.destroyAllWindows()

    kolom_glcm = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    kolom_lbp = [f'lbp_{i}' for i in range(18)]
    kolom = kolom_glcm + kolom_lbp + ['label']

    df = pd.DataFrame(data_fitur, columns=kolom)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Dataset fitur tekstur plastik berhasil disimpan ke {output_csv}")

# --- Eksekusi ---
if __name__ == '__main__':
    ekstrak_visual_dataset('dataset')
