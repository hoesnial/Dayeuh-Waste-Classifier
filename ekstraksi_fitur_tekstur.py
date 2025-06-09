import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops
from matplotlib import pyplot as plt


def ekstrak_fitur_lbp(gray):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist, lbp

def ekstrak_fitur_glcm(gray):
    distances = [1]
    angles = [0]
    glcm = greycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    asm = greycoprops(glcm, 'ASM')[0, 0]
    return [contrast, dissimilarity, homogeneity, energy, correlation, asm], glcm

def tampilkan_proses_ekstraksi(img, lbp_img, glcm_matrix):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalisasi nilai GLCM untuk divisualisasikan
    glcm_vis = glcm_matrix[:, :, 0, 0]
    glcm_vis = cv2.normalize(glcm_vis, None, 0, 255, cv2.NORM_MINMAX)
    glcm_vis = glcm_vis.astype(np.uint8)
    glcm_vis = cv2.resize(glcm_vis, (gray.shape[1], gray.shape[0]))

    # Ubah LBP ke format uint8 untuk ditampilkan
    lbp_vis = cv2.normalize(lbp_img, None, 0, 255, cv2.NORM_MINMAX)
    lbp_vis = lbp_vis.astype(np.uint8)

    # Pastikan semua gambar berdimensi sama
    if lbp_vis.shape != gray.shape:
        lbp_vis = cv2.resize(lbp_vis, (gray.shape[1], gray.shape[0]))

    proses_lbp = np.hstack([gray, lbp_vis])
    proses_glcm = np.hstack([gray, glcm_vis])

    semua = np.vstack([
        proses_lbp,
        proses_glcm
    ])
    return semua

def simpan_visualisasi(output_path, visual_img):
    if not os.path.exists('hasil_visual'):
        os.makedirs('hasil_visual')
    cv2.imwrite(os.path.join('hasil_visual', output_path), visual_img)

def ekstrak_visual_dataset(path_dataset):
    for root, dirs, files in os.walk(path_dataset):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, filename)
                img = cv2.imread(filepath)
                if img is None:
                    print(f"[SKIP] Gagal membaca: {filepath}")
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                lbp_feat, lbp_img = ekstrak_fitur_lbp(gray)
                glcm_feat, glcm_mtx = ekstrak_fitur_glcm(gray)

                visual = tampilkan_proses_ekstraksi(img, lbp_img, glcm_mtx)
                simpan_visualisasi(filename, visual)

                print(f'[OK] {filename} berhasil diproses dan disimpan.')

if __name__ == "__main__":
    ekstrak_visual_dataset('dataset')
