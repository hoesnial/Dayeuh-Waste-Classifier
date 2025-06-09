import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops

def ekstrak_fitur_lbp(gray):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist, lbp

def ekstraksi_fitur_tekstur_glcm(image_gray):
    image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(image_gray, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)

    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    fitur = []
    for prop in props:
        nilai = greycoprops(glcm, prop)
        fitur.append(np.mean(nilai))
    return np.array(fitur), glcm

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

def tampilkan_proses_lbp(image_bgr, lbp_img, lbp_hist):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    asli = cv2.resize(image_bgr, (300, 300))
    gray = cv2.resize(gray, (300, 300))
    lbp_vis = cv2.normalize(lbp_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    lbp_vis = cv2.resize(lbp_vis, (300, 300))

    baris_atas = np.hstack([asli, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)])
    baris_bawah = np.hstack([cv2.cvtColor(lbp_vis, cv2.COLOR_GRAY2BGR), np.zeros_like(asli)])

    # Histogram LBP sebagai gambar
    hist_canvas = np.ones((300, 600, 3), dtype=np.uint8) * 255
    hist_scaled = (lbp_hist * 250).astype(np.int32)
    for i in range(len(hist_scaled)):
        cv2.line(hist_canvas, (i*10 + 10, 290), (i*10 + 10, 290 - hist_scaled[i]), (0, 0, 255), 5)

    return np.vstack([baris_atas, baris_bawah]), hist_canvas

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

                # Ekstraksi fitur
                glcm_fitur, glcm_mtx = ekstraksi_fitur_tekstur_glcm(gray)
                lbp_hist, lbp_img = ekstrak_fitur_lbp(gray)

                # Visualisasi proses
                visual_glcm = tampilkan_proses_glcm(img)
                visual_lbp, hist_lbp = tampilkan_proses_lbp(img, lbp_img, lbp_hist)

                # Tampilkan dua jendela
                cv2.imshow("Visualisasi Proses GLCM", visual_glcm)
                cv2.imshow("Visualisasi Proses LBP", visual_lbp)
                cv2.imshow("Histogram LBP", hist_lbp)

                print(f"Tampilkan {filename} - Tekan tombol untuk lanjut, ESC untuk keluar.")
                key = cv2.waitKey(0)
                if key == 27:
                    print("[!] Dihentikan oleh pengguna.")
                    cv2.destroyAllWindows()
                    return
                cv2.destroyAllWindows()

if __name__ == "__main__":
    ekstrak_visual_dataset('dataset')
