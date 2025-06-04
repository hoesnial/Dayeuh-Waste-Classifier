import cv2
import numpy as np
import os
import pandas as pd

def ekstraksi_warna_presisi(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    mean_L, std_L = np.mean(L), np.std(L)
    mean_A, std_A = np.mean(A), np.std(A)
    mean_B, std_B = np.mean(B), np.std(B)

    fitur = np.array([mean_L, std_L, mean_A, std_A, mean_B, std_B])
    return fitur, L, A, B

def tampilkan_gambar_dengan_teks(judul, image, mean_std):
    img = cv2.resize(image, (300, 300))
    teks = f"{judul}\nMean: {mean_std[0]:.2f} | Std: {mean_std[1]:.2f}"
    y0, dy = 20, 25
    for i, line in enumerate(teks.split('\n')):
        y = y0 + i * dy
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    return img

# ---------- UJI VISUALISASI ----------
if __name__ == '__main__':
    img_path = 'dataset/kertas/contoh1.jpg'  # Ganti path
    image = cv2.imread(img_path)

    if image is None:
        print("Gambar tidak ditemukan!")
        exit()

    fitur, L, A, B = ekstraksi_warna_presisi(image)

    # Hitung statistik untuk ditampilkan di masing-masing kanal
    mean_std_L = (np.mean(L), np.std(L))
    mean_std_A = (np.mean(A), np.std(A))
    mean_std_B = (np.mean(B), np.std(B))

    # Buat gambar grayscale dari tiap kanal
    vis_L = tampilkan_gambar_dengan_teks("Kanal L", L, mean_std_L)
    vis_A = tampilkan_gambar_dengan_teks("Kanal A", A, mean_std_A)
    vis_B = tampilkan_gambar_dengan_teks("Kanal B", B, mean_std_B)

    # Gambar asli
    img_asli = cv2.resize(image, (300, 300))
    cv2.putText(img_asli, "Gambar Asli (BGR)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Gabungkan semua untuk ditampilkan
    atas = np.hstack([img_asli, cv2.cvtColor(vis_L, cv2.COLOR_GRAY2BGR)])
    bawah = np.hstack([cv2.cvtColor(vis_A, cv2.COLOR_GRAY2BGR),
                       cv2.cvtColor(vis_B, cv2.COLOR_GRAY2BGR)])
    gabung = np.vstack([atas, bawah])

    cv2.imshow("Proses Ekstraksi Warna - Deteksi Sampah Kertas", gabung)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------- EKSTRAKSI SEMUA DATASET & SIMPAN CSV ----------
    def ekstrak_dataset_ke_csv(folder_dataset, output_csv='fitur_warna_dataset.csv'):
        label_map = {'plastik': 0, 'logam': 1, 'kertas': 2}
        data_fitur = []

        for kategori, label in label_map.items():
            folder_kategori = os.path.join(folder_dataset, kategori)
            if not os.path.exists(folder_kategori):
                print(f"[!] Folder tidak ditemukan: {folder_kategori}")
                continue

            for nama_file in os.listdir(folder_kategori):
                path_gambar = os.path.join(folder_kategori, nama_file)
                img = cv2.imread(path_gambar)

                if img is None:
                    print(f"[!] Gagal membaca: {path_gambar}")
                    continue

                img = cv2.resize(img, (300, 300))
                fitur, _, _, _ = ekstraksi_warna_presisi(img)
                data_fitur.append(list(fitur) + [label])

        kolom = ['mean_L', 'std_L', 'mean_A', 'std_A', 'mean_B', 'std_B', 'label']
        df = pd.DataFrame(data_fitur, columns=kolom)
        df.to_csv(output_csv, index=False)
        print(f"[✓] Ekstraksi dataset selesai. Disimpan di: {output_csv}")

    ekstrak_dataset_ke_csv('dataset')
