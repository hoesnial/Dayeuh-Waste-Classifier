import cv2
import numpy as np
import os

def ekstraksi_fitur_logam(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]

    if contours:
        cnt_terbesar = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt_terbesar)
        hull = cv2.convexHull(cnt_terbesar)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        moments = cv2.moments(cnt_terbesar)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    else:
        solidity = 0
        hu_moments_log = np.zeros(7)

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

def tampilkan_gambar_dengan_text(judul, img):
    img = cv2.resize(img, (300, 300))
    warna = (255, 255, 255) if len(img.shape) == 2 else (0, 255, 0)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img, judul, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna, 2)
    return img

def proses_folder_logam(folder_path='dataset/logam', output_dir='hasil_ekstraksi/logam'):
    if not os.path.exists(folder_path):
        print(f"[!] Folder tidak ditemukan: {folder_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(folder_path):
        path_img = os.path.join(folder_path, filename)
        image = cv2.imread(path_img)
        if image is None:
            print(f"[!] Gagal membaca gambar: {path_img}")
            continue

        fitur, gray, threshold = ekstraksi_fitur_logam(image)
        image_resized = cv2.resize(image, (300, 300))
        gray_resized = cv2.resize(gray, (300, 300))
        threshold_resized = cv2.resize(threshold, (300, 300))
        kontur_resized = gambar_kontur_logam(image_resized, threshold_resized)

        print(f"\n--- Fitur untuk {filename} ---")
        for k, v in fitur.items():
            print(f"{k}: {v:.4f}")

        # Buat versi anotasi
        vis1 = tampilkan_gambar_dengan_text("Gambar Asli", image_resized)
        vis2 = tampilkan_gambar_dengan_text("Grayscale", gray_resized)
        vis3 = tampilkan_gambar_dengan_text("Threshold Area Terang", threshold_resized)
        vis4 = tampilkan_gambar_dengan_text("Deteksi Kontur Logam", kontur_resized)

        nama_dasar = os.path.splitext(filename)[0]
        simpan_gambar = [
            (vis1, f"{nama_dasar}_asli.jpg"),
            (vis2, f"{nama_dasar}_gray.jpg"),
            (vis3, f"{nama_dasar}_thresh.jpg"),
            (vis4, f"{nama_dasar}_kontur.jpg")
        ]

        for i, (img_out, nama_file) in enumerate(simpan_gambar):
            path_simpan = os.path.join(output_dir, nama_file)
            cv2.imwrite(path_simpan, img_out)
            cv2.imshow(f"[{filename}] - {nama_file}", img_out)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == 27:  # ESC
                print("[X] Dihentikan oleh pengguna.")
                return

if __name__ == '__main__':
    proses_folder_logam()
