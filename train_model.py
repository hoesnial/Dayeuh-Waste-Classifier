import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

def load_features(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y

def train_and_save_models(csv_path):
    X, y = load_features(csv_path)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    # KNN
    k = min(3, len(X_train))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print("\n=== KNN Classification Report ===")
    print(classification_report(y_test, y_pred_knn, target_names=le.classes_))
    print("Akurasi KNN:", accuracy_score(y_test, y_pred_knn))

    # SVM
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print("\n=== SVM Classification Report ===")
    print(classification_report(y_test, y_pred_svm, target_names=le.classes_))
    print("Akurasi SVM:", accuracy_score(y_test, y_pred_svm))

    # Save models and label encoder
    dump(knn, 'knn_model.joblib')
    dump(svm, 'svm_model.joblib')
    dump(le, 'label_encoder.joblib')
    print("Models and label encoder saved.")

if __name__ == '__main__':
    train_and_save_models('fitur_dataset.csv')