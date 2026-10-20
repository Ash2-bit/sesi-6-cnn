# Sesi 6: AI dan Penginderaan Jauh (Remote Sensing)
**Klasifikasi Deforestasi Lanskap via Citra Udara Menggunakan *Convolutional Neural Network* (CNN)**

## 📖 Deskripsi Studi Kasus
Pemantauan kesehatan lahan, perluasan perkebunan, atau pelacakan laju deforestasi di wilayah beriklim tropis memakan biaya yang sangat mahal jika hanya mengandalkan survei lapangan (*ground patrol*) manual. Sebaliknya, data citra satelit (optik maupun radar) menyediakan visibilitas yang luas, namun membutuhkan interpretasi visual yang memakan waktu jika dilakukan oleh manusia.

Studi kasus ini mendemonstrasikan perancangan arsitektur algoritma *Deep Learning* berbasis *Convolutional Neural Network* (CNN) untuk mengekstraksi dan mengklasifikasikan pola piksel dari citra udara secara otomatis. Model ini bertugas membedakan secara biner antara citra yang merupakan "Kawasan Hijau/Sehat" dan citra yang terindikasi "Deforestasi/Lahan Kritis", yang nantinya dapat diintegrasikan ke dalam platform web pemantauan operasional.

## 🎯 Tujuan Eksekusi
1. **Ekstraksi Fitur Spasial:** Menggunakan *Convolutional Layers* (Conv2D) untuk mengenali pola primitif citra seperti batas tepi, warna dedaunan, dan tekstur tanah tanpa rekayasa fitur manual (*hand-crafted features*).
2. **Reduksi Dimensi:** Mengimplementasikan *Pooling Layers* (MaxPooling) untuk menyusutkan beban komputasi sambil mempertahankan informasi spasial yang paling dominan.
3. **Klasifikasi Biner:** Membangun *Dense Layer* dengan fungsi aktivasi Sigmoid pada tahap akhir untuk mengeluarkan probabilitas (0 hingga 1) status kesehatan area tersebut.

## 💻 Implementasi Kode (Python - TensorFlow & Keras)
Berikut adalah skrip dasar perancangan arsitektur jaringannya yang dapat dijalankan melalui file `cnn_deforestation_classifier.py`.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

print("=== ARSITEKTUR CNN UNTUK DETEKSI DEFORESTASI ===\n")

# 1. Membangun Arsitektur Jaringan CNN
def bangun_model_cnn():
    model = models.Sequential()
    
    # Blok Ekstraksi Fitur 1
    # Memproses input citra berukuran 128x128 piksel dengan 3 channel warna (RGB)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3), name='Conv_1'))
    model.add(layers.MaxPooling2D((2, 2), name='MaxPool_1'))
    
    # Blok Ekstraksi Fitur 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', name='Conv_2'))
    model.add(layers.MaxPooling2D((2, 2), name='MaxPool_2'))
    
    # Blok Ekstraksi Fitur 3
    model.add(layers.Conv2D(64, (3, 3), activation='relu', name='Conv_3'))
    
    # Blok Klasifikasi
    # Meratakan matriks menjadi array 1D
    model.add(layers.Flatten(name='Flatten'))
    
    # Hidden layer untuk menalar fitur yang sudah diekstraksi
    model.add(layers.Dense(64, activation='relu', name='Dense_Hidden'))
    
    # Output layer: 1 neuron dengan aktivasi Sigmoid untuk probabilitas Biner
    # Output mendekati 0 -> Area Hijau/Hutan Normal
    # Output mendekati 1 -> Deforestasi/Lahan Terbuka
    model.add(layers.Dense(1, activation='sigmoid', name='Output_Layer'))
    
    return model

# Inisialisasi Model
model_citra = bangun_model_cnn()

# 2. Kompilasi Model
model_citra.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Menampilkan Ringkasan Arsitektur
print("--- Ringkasan Arsitektur Model ---")
model_citra.summary()

# 3. Simulasi Pengujian Prediksi (Dummy Data)
print("\n--- Simulasi Prediksi (Inference) ---")
# Mensimulasikan 2 buah gambar udara berukuran 128x128 pixel (RGB) acak
citra_dummy = np.random.rand(2, 128, 128, 3) 

# Meminta model memprediksi gambar dummy tersebut
hasil_prediksi = model_citra.predict(citra_dummy)

for i, probabilitas in enumerate(hasil_prediksi):
    status = "Deforestasi (Peringatan!)" if probabilitas[0] > 0.5 else "Kawasan Hijau (Normal)"
    print(f"Citra #{i+1} -> Skor Klasifikasi: {probabilitas[0]:.4f} | Status: {status}")
