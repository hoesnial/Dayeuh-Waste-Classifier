import cv2
import numpy as np
import pandas as pd
import os

def ekstraksi_fitur_logam(image):
    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Laplacian untuk mendeteksi pantulan tajam
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    # 3. Threshold area terang
    _, threshold = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # 4. Statistik intensitas dan pantulan
    fitur = {
        "mean_gray": np.mean(gray),
        "std_gray": np.std(gray),
        "mean_laplacian": np.mean(laplacian_abs),
        "std_laplacian": np.std(laplacian_abs)
    }

    return fitur, gray, laplacian_abs, threshold

def gambar_kontur_logam(image_asli, mask_threshold):
    kontur_img = image_asli.copy()
    contours, _ = cv2.findContours(mask_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]
    cv2.drawContours(kontur_img, contours, -1, (0, 0, 255), 2)
    return kontur_img

# ---------- UJI VISUALISASI & SIMPAN CSV ----------
if __name__ == '__main__':
    img_path = 'dataset/logam/contoh1.jpg'  # Ganti dengan path gambar logam
    image = cv2.imread(img_path)

    if image is None:
        print("Gambar tidak ditemukan!")
        exit()

    fitur, gray, laplacian_abs, threshold = ekstraksi_fitur_logam(image)
    gambar_kontur = gambar_kontur_logam(cv2.resize(image, (300, 300)), cv2.resize(threshold, (300, 300)))

    print("\nFitur Logam Terdeteksi:")
    for k, v in fitur.items():
        print(f"{k}: {v:.2f}")

    # ---------- SIMPAN KE CSV ----------
    output_csv = 'fitur_logam_dataset.csv'
    df_fitur = pd.DataFrame([fitur])
    df_fitur.insert(0, 'nama_file', os.path.basename(img_path))

    if not os.path.isfile(output_csv):
        df_fitur.to_csv(output_csv, index=False)
    else:
        df_fitur.to_csv(output_csv, mode='a', header=False, index=False)

    print(f"Fitur berhasil disimpan ke dalam file: {output_csv}")

    # ---------- VISUALISASI ----------
    img_asli = cv2.resize(image, (300, 300))
    gray = cv2.resize(gray, (300, 300))
    laplacian_vis = cv2.resize(laplacian_abs, (300, 300))
    threshold = cv2.resize(threshold, (300, 300))

    cv2.putText(img_asli, "Gambar Asli", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(gray, "Grayscale", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    cv2.putText(laplacian_vis, "Laplacian (Pantulan)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    cv2.putText(threshold, "Threshold Area Terang", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    cv2.putText(gambar_kontur, "Deteksi Logam (Kontur)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    atas = np.hstack([img_asli, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)])
    tengah = np.hstack([cv2.cvtColor(laplacian_vis, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)])
    bawah = np.hstack([gambar_kontur, np.zeros_like(gambar_kontur)])

    final_vis = np.vstack([atas, tengah, bawah])

    cv2.imshow("Deteksi Sampah Logam - Ekstraksi + Kontur", final_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
