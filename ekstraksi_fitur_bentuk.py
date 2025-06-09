import cv2
import numpy as np
import pandas as pd
import os

def ekstraksi_fitur_logam(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold area terang untuk segmentasi
    _, threshold = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Cari kontur terbesar untuk hitung fitur bentuk
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]

    if contours:
        cnt_terbesar = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt_terbesar)
        hull = cv2.convexHull(cnt_terbesar)
        hull_area = cv2.contourArea(hull)

        if hull_area > 0:
            solidity = float(area) / hull_area
        else:
            solidity = 0

        # Hu Moments
        moments = cv2.moments(cnt_terbesar)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    else:
        solidity = 0
        hu_moments_log = np.zeros(7)

    # Statistik dasar grayscale
    fitur = {
        "mean_gray": np.mean(gray),
        "std_gray": np.std(gray),
        "solidity": solidity,
    }

    for i in range(7):
        fitur[f"hu_moment{i+1}"] = hu_moments_log[i]

    return fitur, gray, threshold

def gambar_kontur_logam(image_asli, mask_threshold):
    kontur_img = image_asli.copy()
    contours, _ = cv2.findContours(mask_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]
    cv2.drawContours(kontur_img, contours, -1, (0, 0, 255), 2)
    return kontur_img

# ---------- UJI VISUALISASI & SIMPAN CSV ----------
if __name__ == '__main__':
    img_path = 'dataset/logam/contoh1.jpg'  # Ganti path sesuai gambar
    image = cv2.imread(img_path)

    if image is None:
        print("Gambar tidak ditemukan!")
        exit()

    fitur, gray, threshold = ekstraksi_fitur_logam(image)
    gambar_kontur = gambar_kontur_logam(cv2.resize(image, (300, 300)), cv2.resize(threshold, (300, 300)))

    print("\nFitur Logam Terdeteksi:")
    for k, v in fitur.items():
        print(f"{k}: {v:.4f}")

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
    threshold = cv2.resize(threshold, (300, 300))

    cv2.putText(img_asli, "Gambar Asli", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(gray, "Grayscale", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    cv2.putText(threshold, "Threshold Area Terang", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    cv2.putText(gambar_kontur, "Deteksi Logam (Kontur)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    atas = np.hstack([img_asli, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)])
    bawah = np.hstack([cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR), gambar_kontur])
    final_vis = np.vstack([atas, bawah])

    cv2.imshow("Deteksi Sampah Logam - Hu Moments & Solidity", final_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
