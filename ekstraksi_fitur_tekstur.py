import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt

def ekstraksi_tekstur_glcm(image_gray):
    glcm = graycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    fitur = {
        'energi': graycoprops(glcm, 'energy')[0, 0],
        'kontras': graycoprops(glcm, 'contrast')[0, 0],
        'homogenitas': graycoprops(glcm, 'homogeneity')[0, 0],
        'korelasi': graycoprops(glcm, 'correlation')[0, 0],
    }
    return fitur

# ---------- UJI VISUALISASI ----------
if __name__ == '__main__':
    path = 'dataset/plastik/contoh1.jpg'  # Ganti dengan path gambar plastik kamu
    image = cv2.imread(path)

    if image is None:
        print("Gambar tidak ditemukan!")
        exit()

    # Grayscale dan pra-pemrosesan
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Resize agar seragam untuk GLCM
    gray_resized = cv2.resize(blur, (256, 256))

    # Ekstraksi fitur tekstur (GLCM)
    fitur = ekstraksi_tekstur_glcm(gray_resized)

    # Segmentasi threshold untuk deteksi objek
    _, thresh = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Temukan kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Buat salinan gambar untuk menampilkan kontur
    kontur_img = cv2.cvtColor(gray_resized.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(kontur_img, contours, -1, (0, 255, 0), 2)

    # Tampilkan fitur
    print("Fitur Tekstur (GLCM):")
    for k, v in fitur.items():
        print(f"{k.capitalize()}: {v:.4f}")

    # Visualisasi proses
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Gambar Asli")
    axs[0].axis('off')

    axs[1].imshow(gray_resized, cmap='gray')
    axs[1].set_title("Grayscale + Resize")
    axs[1].axis('off')

    axs[2].imshow(cv2.cvtColor(kontur_img, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Deteksi Kontur Objek Plastik")
    axs[2].axis('off')

    plt.suptitle("Ekstraksi Tekstur Plastik + Kontur", fontsize=14)
    plt.tight_layout()
    plt.show()
