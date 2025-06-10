import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

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
    glcm = graycomatrix(image_gray, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)

    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    fitur = []
    for prop in props:
        nilai = graycoprops(glcm, prop)
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

def tampilkan_proses_lbp(image_bgr, lbp_img):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    asli = cv2.resize(image_bgr, (300, 300))
    gray = cv2.resize(gray, (300, 300))
    lbp_vis = cv2.normalize(lbp_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    lbp_vis = cv2.resize(lbp_vis, (300, 300))

    baris_atas = np.hstack([asli, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)])
    baris_bawah = np.hstack([cv2.cvtColor(lbp_vis, cv2.COLOR_GRAY2BGR), np.zeros_like(asli)])

    return np.vstack([baris_atas, baris_bawah])

def tampilkan_histogram_lbp_matplotlib(hist_data, title):
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(hist_data)), hist_data, width=0.8, color='blue')
    plt.title(title)
    plt.xlabel("Bin")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_image(title, image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def ekstrak_visual_dataset(path_dataset):
    output_dir = 'hasil_ekstraksi/plastik'
    os.makedirs(output_dir, exist_ok=True)

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
                visual_lbp = tampilkan_proses_lbp(img, lbp_img)

                # Tampilkan hasil dengan matplotlib
                show_image(f"{filename} - GLCM", visual_glcm)
                show_image(f"{filename} - LBP", visual_lbp)
                tampilkan_histogram_lbp_matplotlib(lbp_hist, f"{filename} - LBP Histogram")

                # Simpan file hasil
                nama_file = os.path.splitext(filename)[0]
                cv2.imwrite(os.path.join(output_dir, f"{nama_file}_glcm.jpg"), visual_glcm)
                cv2.imwrite(os.path.join(output_dir, f"{nama_file}_lbp.jpg"), visual_lbp)

                print(f"[✓] Disimpan dan ditampilkan: {nama_file}_glcm.jpg, _lbp.jpg")

if __name__ == "__main__":
    ekstrak_visual_dataset('dataset')
