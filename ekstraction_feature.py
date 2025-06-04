import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image
import mahotas

# ----------- FUNGSI EKSTRAKSI -----------

def ekstraksi_warna(image):
    mean = cv2.mean(image)[:3]
    return np.array(mean)

def ekstraksi_bentuk(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binarized = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter != 0 else 0
        return np.array([area, perimeter, circularity])
    else:
        return np.array([0, 0, 0])

def ekstraksi_tekstur(image):
    gray = rgb2gray(image)
    gray = (gray * 255).astype(np.uint8)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick[:5]

# ----------- LOAD DATASET -----------

def load_dataset(dataset_path, visualisasi=False):
    X, y = [], []
    label_names = os.listdir(dataset_path)
    fitur_data = []

    for label in label_names:
        folder_path = os.path.join(dataset_path, label)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(file_path).convert('RGB')
                img = img.resize((256, 256))
                img_np = np.array(img)

                warna = ekstraksi_warna(img_np)
                bentuk = ekstraksi_bentuk(img_np)
                tekstur = ekstraksi_tekstur(img_np)

                fitur = np.concatenate([warna, bentuk, tekstur])
                X.append(fitur)
                y.append(label)
                fitur_data.append(list(fitur) + [label])

                # --- Visualisasi dengan OpenCV (diperbesar) ---
                if visualisasi:
                    vis_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    teks = f"{label} | W: {warna.round(1)} | B: {bentuk.round(1)} | T: {tekstur[:3].round(2)}"
                    cv2.putText(vis_img, teks, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                    # Perbesar gambar (misal 4x lebar, 3x tinggi)
                    vis_img = cv2.resize(vis_img, (0, 0), fx=4.0, fy=3.0)

                    cv2.imshow('Visualisasi Gambar', vis_img)
                    print("Tekan tombol apa saja untuk lanjut, atau 'q' untuk keluar dari visualisasi.")
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        return np.array(X), np.array(y), fitur_data

            except Exception as e:
                print(f"Error memproses {file_path}: {e}")

    cv2.destroyAllWindows()
    return np.array(X), np.array(y), fitur_data

# ----------- MAIN -----------

def main():
    dataset_path = 'dataset'
    X, y, fitur_data = load_dataset(dataset_path, visualisasi=True)

    # Simpan fitur ke CSV
    col_names = [f'fitur_{i+1}' for i in range(X.shape[1])] + ['label']
    df = pd.DataFrame(fitur_data, columns=col_names)
    df.to_csv("fitur_dataset.csv", index=False)
    print("Fitur disimpan ke fitur_dataset.csv")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    # --- KNN ---
    k = min(3, len(X_train))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    print("\n=== KNN Classification Report ===")
    print(classification_report(y_test, y_pred_knn, target_names=le.classes_))
    print("Akurasi KNN:", accuracy_score(y_test, y_pred_knn))

    # --- SVM ---
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)

    print("\n=== SVM Classification Report ===")
    print(classification_report(y_test, y_pred_svm, target_names=le.classes_))
    print("Akurasi SVM:", accuracy_score(y_test, y_pred_svm))

    # --- Tampilkan hasil prediksi ---
    print("\nHasil prediksi gambar uji (KNN vs SVM):")
    for i in range(len(X_test)):
        label_asli = le.inverse_transform([y_test[i]])[0]
        pred_knn = le.inverse_transform([y_pred_knn[i]])[0]
        pred_svm = le.inverse_transform([y_pred_svm[i]])[0]
        print(f"[{i+1}] Asli: {label_asli} | KNN: {pred_knn} | SVM: {pred_svm}")

if __name__ == '__main__':
    main()
