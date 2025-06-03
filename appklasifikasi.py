import cv2
import numpy as np
from joblib import load
import mahotas
from skimage.color import rgb2gray

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

def extract_features(img_np):
    warna = ekstraksi_warna(img_np)
    bentuk = ekstraksi_bentuk(img_np)
    tekstur = ekstraksi_tekstur(img_np)
    fitur = np.concatenate([warna, bentuk, tekstur])
    return fitur

def detect(fitur, model_type='svm'):
    model = load(f'{model_type}_model.joblib')
    le = load('label_encoder.joblib')
    fitur = np.array(fitur).reshape(1, -1)
    pred = model.predict(fitur)
    label = le.inverse_transform(pred)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(fitur).max()
    return label, prob

if __name__ == '__main__':
    path = "botol5.jpg"
    img = cv2.imread(path)
    if img is None:
        print("Gambar tidak ditemukan.")
        exit()

    clone = img.copy()
    r = cv2.selectROI("Pilih objek (drag rectangle, tekan ENTER)", img, showCrosshair=True)
    if r == (0,0,0,0):
        print("Rectangle tidak dipilih.")
        exit()
    x, y, w, h = r

    # Ekstraksi fitur dari area rectangle (tanpa crop manual)
    area_rgb = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    fitur = extract_features(area_rgb)

    # Klasifikasi
    label, prob = detect(fitur, model_type='svm')
    print(f"Hasil Deteksi: {label} (Probabilitas: {prob:.2f})")

    # Tampilkan hasil pada gambar
    cv2.rectangle(clone, (x, y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(clone, f"{label} ({prob:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Hasil Deteksi", clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()