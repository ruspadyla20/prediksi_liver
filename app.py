import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.metrics import classification_report_imbalanced

# Judul Aplikasi
st.title("Aplikasi Prediksi Penyakit Liver")

# Input Data
st.sidebar.header("Input Parameter")
def user_input_features():
    age = st.sidebar.number_input("Umur", 20, 100, 50)
    gender = st.sidebar.selectbox("Jenis Kelamin", (0, 1))
    totbil = st.sidebar.number_input("Total Bilirubin", 0.0, 100.0, 1.0)
    dirbil = st.sidebar.number_input("Direct Bilirubin", 0.0, 10.0, 0.5)
    alpho = st.sidebar.number_input("Alkaline Phosphotase", 50, 3000, 100)
    alamino = st.sidebar.number_input("Alamine Aminotransferase", 0, 3000, 40)
    asparami = st.sidebar.number_input("Aspartate Aminotransferase", 0, 3000, 40)
    totalpro = st.sidebar.number_input("Total Proteins", 0.0, 10.0, 7.0)
    albumin = st.sidebar.number_input("Albumin", 0.0, 6.0, 3.0)
    agr = st.sidebar.number_input("Albumin dan Globulin Ratio", 0.0, 3.0, 1.0)
    
    data = {
        'Age': age,
        'Gender': gender,
        'Total_Bilirubin': totbil,
        'Direct_Bilirubin': dirbil,
        'Alkaline_Phosphotase': alpho,
        'Alamine_Aminotransferase': alamino,
        'Aspartate_Aminotransferase': asparami,
        'Total_Proteins': totalpro,
        'Albumin': albumin,
        'Albumin_and_Globulin_Ratio': agr
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Tampilkan input pengguna
st.subheader('Parameter Input')
st.write(df)

# Load dataset
data = pd.read_csv('/content/drive/MyDrive/ML 6/Data Pasien penyakit liver.csv')  # Pastikan file dataset tersedia

# Preprocessing
X = data.drop(columns='Dataset')  # Ganti 'Dataset' dengan nama kolom target yang sesuai
y = data['Dataset']  # Ganti 'Dataset' dengan nama kolom target yang sesuai

# Standardisasi data
scaler = StandardScaler()
X = scaler.fit_transform(X)
df = scaler.transform(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prediksi
prediction = knn.predict(df)

# Tampilkan hasil prediksi
st.subheader('Hasil Prediksi')
st.write('Penyakit Liver' if prediction[0] == 1 else 'Tidak Ada Penyakit Liver')

# Evaluasi model
st.subheader('Evaluasi Model')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, knn.predict(X_test))
st.write("Confusion Matrix:", conf_matrix)

# Classification Report
class_report = classification_report_imbalanced(y_test, knn.predict(X_test))
st.write("Classification Report:", class_report)

# Accuracy Score
accuracy = accuracy_score(y_test, knn.predict(X_test))
st.write(f"Accuracy Score: {accuracy}")
