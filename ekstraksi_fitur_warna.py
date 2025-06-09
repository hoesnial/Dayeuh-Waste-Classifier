import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def ekstraksi_warna_presisi(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mean_L, std_L = np.mean(L), np.std(L)
    mean_A, std_A = np.mean(A), np.std(A)
    mean_B, std_B = np.mean(B), np.std(B)
    fitur = np.array([mean_L, std_L, mean_A, std_A, mean_B, std_B])
    return fitur, L, A, B 

def ekstraksi_hsv_mean_hist(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    mean_H, mean_S, mean_V = np.mean(H), np.mean(S), np.mean(V)
    
    hist_H = cv2.calcHist([H], [0], None, [256], [0,256])
    hist_S = cv2.calcHist([S], [0], None, [256], [0,256])
    hist_V = cv2.calcHist([V], [0], None, [256], [0,256])
    
    hist_H = hist_H / hist_H.max()
    hist_S = hist_S / hist_S.max()
    hist_V = hist_V / hist_V.max()
    
    fitur = np.array([mean_H, mean_S, mean_V])
    return fitur, H, S, V, hist_H, hist_S, hist_V

def tampilkan_gambar_dengan_teks(judul, image, mean_std):
    img = cv2.resize(image, (300, 300))
    teks = f"{judul}\nMean: {mean_std[0]:.2f} | Std: {mean_std[1]:.2f}" if len(mean_std) == 2 else f"{judul}\nMean: {mean_std[0]:.2f}"
    y0, dy = 20, 25
    for i, line in enumerate(teks.split('\n')):
        y = y0 + i * dy
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    return img

def tampilkan_histogram(hist_H, hist_S, hist_V):
    plt.figure(figsize=(9, 3))
    plt.subplot(1,3,1)
    plt.title('Histogram Hue')
    plt.plot(hist_H, color='r')
    plt.xlim([0, 256])
    plt.subplot(1,3,2)
    plt.title('Histogram Saturation')
    plt.plot(hist_S, color='g')
    plt.xlim([0, 256])
    plt.subplot(1,3,3)
    plt.title('Histogram Value')
    plt.plot(hist_V, color='b')
    plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()

def proses_kertas_polos_berwarna(folder_kertas):
    if not os.path.exists(folder_kertas):
        print(f"[!] Folder tidak ditemukan: {folder_kertas}")
        return

    for nama_file in os.listdir(folder_kertas):
        path_gambar = os.path.join(folder_kertas, nama_file)
        img = cv2.imread(path_gambar)
        if img is None:
            print(f"[!] Gagal membaca: {path_gambar}")
            continue

        img = cv2.resize(img, (300, 300))

        # Ekstraksi LAB
        fitur_lab, L, A, B = ekstraksi_warna_presisi(img)
        stds = [np.std(L), np.std(A), np.std(B)]
        total_std = sum(stds)
        tipe_kertas = "Polos" if total_std < 10 else "Berwarna"

        mean_std_L = (np.mean(L), np.std(L))
        mean_std_A = (np.mean(A), np.std(A))
        mean_std_B = (np.mean(B), np.std(B))

        vis_L = tampilkan_gambar_dengan_teks("Kanal L", L, mean_std_L)
        vis_A = tampilkan_gambar_dengan_teks("Kanal A", A, mean_std_A)
        vis_B = tampilkan_gambar_dengan_teks("Kanal B", B, mean_std_B)

        # Ekstraksi HSV
        fitur_hsv, H, S, V, hist_H, hist_S, hist_V = ekstraksi_hsv_mean_hist(img)
        mean_H = np.mean(H)
        mean_S = np.mean(S)
        mean_V = np.mean(V)

        vis_H = tampilkan_gambar_dengan_teks("Kanal H", H, (mean_H,))
        vis_S = tampilkan_gambar_dengan_teks("Kanal S", S, (mean_S,))
        vis_V = tampilkan_gambar_dengan_teks("Kanal V", V, (mean_V,))

        # Gabungkan hasil visualisasi LAB
        img_asli = img.copy()
        cv2.putText(img_asli, f"{nama_file} ({tipe_kertas})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        atas_lab = np.hstack([img_asli, cv2.cvtColor(vis_L, cv2.COLOR_GRAY2BGR)])
        bawah_lab = np.hstack([cv2.cvtColor(vis_A, cv2.COLOR_GRAY2BGR),
                           cv2.cvtColor(vis_B, cv2.COLOR_GRAY2BGR)])
        gabung_lab = np.vstack([atas_lab, bawah_lab])

        # Gabungkan hasil visualisasi HSV
        atas_hsv = np.hstack([img_asli, cv2.cvtColor(vis_H, cv2.COLOR_GRAY2BGR)])
        bawah_hsv = np.hstack([cv2.cvtColor(vis_S, cv2.COLOR_GRAY2BGR),
                           cv2.cvtColor(vis_V, cv2.COLOR_GRAY2BGR)])
        gabung_hsv = np.vstack([atas_hsv, bawah_hsv])

        # Tampilkan semua jendela
        cv2.imshow("1. Ekstraksi Warna LAB", gabung_lab)
        cv2.imshow("2. Ekstraksi Warna HSV", gabung_hsv)

        print(f"[OK] {nama_file} -> Kertas {tipe_kertas} | Total Std Dev LAB: {total_std:.2f}")
        print(f"     HSV Mean (H, S, V): {fitur_hsv[0]:.2f}, {fitur_hsv[1]:.2f}, {fitur_hsv[2]:.2f}")

        tampilkan_histogram(hist_H, hist_S, hist_V)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            print("[X] Proses dihentikan oleh pengguna.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

if __name__ == '__main__':
    proses_kertas_polos_berwarna('dataset/kertas')
