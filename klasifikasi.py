import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from ekstraksi_fitur_warna import load_color_features
from ekstraksi_fitur_tekstur import load_texture_features
from ekstraksi_fitur_bentuk import load_shape_features
from skimage.feature import greycomatrix, greycoprops



# Load semua fitur
color_features, color_labels = load_color_features("dataset/kertas", "kertas")
texture_features, texture_labels = load_texture_features("dataset/plastik", "plastik")
shape_features, shape_labels = load_shape_features("dataset/logam", "logam")

# Samakan dimensi semua fitur dengan padding nol
def pad_features(features, target_dim):
    padded = np.zeros((len(features), target_dim))
    for i, feat in enumerate(features):
        padded[i, :len(feat)] = feat
    return padded

max_dim = max(len(color_features[0]), len(texture_features[0]), len(shape_features[0]))
X_color = pad_features(color_features, max_dim)
X_texture = pad_features(texture_features, max_dim)
X_shape = pad_features(shape_features, max_dim)

# Gabungkan fitur dan label
X = np.concatenate([X_color, X_texture, X_shape])
y = np.array(color_labels + texture_labels + shape_labels)

# Standarisasi
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# ✅ Model 1: Random Forest Classifier
# -------------------------------
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print("\n📊 Hasil Klasifikasi: Random Forest")
print(classification_report(y_test, rfc_pred))

# -------------------------------
# ✅ Model 2: Support Vector Machine (SVM)
# -------------------------------
svm = SVC(kernel='rbf', C=1, gamma='scale')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print("\n📊 Hasil Klasifikasi: SVM")
print(classification_report(y_test, svm_pred))
