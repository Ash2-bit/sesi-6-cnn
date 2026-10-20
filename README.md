# Deteksi Deforestasi via Penginderaan Jauh Menggunakan Convolutional Neural Network (CNN)

## Deskripsi

Proyek ini merupakan studi kasus penerapan *Deep Learning* untuk analisis citra penginderaan jauh (*Remote Sensing*). 
Proyek ini mengadaptasi materi tentang visi komputer (*Computer Vision*) dan evaluasi model klasifikasi. Fokus utamanya adalah membangun arsitektur *Convolutional Neural Network* (CNN) untuk mendeteksi secara otomatis area lahan yang mengalami deforestasi atau kerusakan ekosistem dari udara, serta bagaimana menangani ketidakseimbangan kelas (*class imbalance*) dalam dataset lingkungan.

## Tujuan

Tujuan proyek ini adalah:
1. Memproses matriks citra digital 3 channel (RGB) menjadi fitur spasial menggunakan *Convolutional Layers*.
2. Membangun model klasifikasi biner untuk membedakan antara "Kawasan Hijau (Aman)" dan "Lahan Kritis/Deforestasi".
3. Mengetahui pengaruh *class weights* (pembobotan kelas) pada dataset yang tidak seimbang (area deforestasi biasanya lebih sedikit dari hutan normal).
4. Mengevaluasi performa model menggunakan *Confusion Matrix* dan metrik *Recall* sebagai pendukung keputusan (*Decision Support*).

## Studi Kasus

Kementerian Lingkungan Hidup dan Kehutanan (KLHK) menerima ribuan citra satelit/drone setiap minggunya untuk memantau laju deforestasi ilegal. Menginspeksi gambar ini secara manual sangat mahal dan lambat.
Sistem AI ini dibangun sebagai *Decision Support System* untuk menyaring gambar. Sistem akan menyoroti citra yang terindikasi rusak agar segera diinspeksi oleh manusia. Dalam bisnis dan mitigasi lingkungan, gagal mendeteksi lahan yang gundul (*False Negative*) berakibat hilangnya ekosistem yang fatal. Oleh karena itu, model tidak hanya dinilai dari *Accuracy*, melainkan dari seberapa tinggi nilai *Recall* (kemampuan model menangkap sebanyak mungkin potensi deforestasi).

## Dataset

Mengingat ukuran citra satelit asli sangat masif, dataset disimulasikan menggunakan Python dan NumPy.
*   Total Data: 1.000 citra berukuran 64x64 piksel (RGB).
*   Distribusi Kelas (*Class Imbalance*): 85% Kawasan Hijau (Kelas 0) dan 15% Deforestasi (Kelas 1).
*   *Feature Engineering*: Citra "Hijau" disimulasikan memiliki dominasi nilai piksel pada *channel* warna hijau, sementara citra "Deforestasi" didominasi warna coklat/kemerahan agar CNN dapat mempelajari pola perbedaannya secara empiris.

## Metode

Tahapan utama yang dilakukan:
1. *Image Data Generation & Augmentation Simulation* (Pembuatan dataset matriks 4D).
2. *Train-Test Split* & *Normalization* (Skala piksel 0.0 - 1.0).
3. *CNN Architecture Modeling* (Penyusunan layer Conv2D, MaxPooling, Flatten, dan Dense).
4. *Imbalance Handling* (Penerapan parameter `class_weight` saat pelatihan).
5. *Model Evaluation* (Analisis *Loss/Accuracy Curves* dan *Confusion Matrix*).

## Evaluasi

Performa model dievaluasi menggunakan:
*   *Accuracy*: Tingkat kebenaran keseluruhan.
*   *Precision*: Seberapa akurat model saat menebak deforestasi.
*   *Recall*: Persentase deforestasi aktual yang berhasil dideteksi oleh sistem.
*   *Confusion Matrix*: Pemetaan silang antara prediksi AI dan kebenaran lapangan (*Ground Truth*).

## Teknologi

*   Python
*   Google Colab / Jupyter Notebook (`.ipynb`)
*   TensorFlow / Keras
*   Scikit-Learn (Metrik Evaluasi)
*   Matplotlib & Seaborn (Visualisasi)

## Cara Menjalankan

1. Proyek ini direkomendasikan untuk dijalankan di Google Colab (aktifkan akselerasi GPU: Runtime -> Change runtime type -> T4 GPU).
2. Buka file `cnn_deforestation_classifier.ipynb`.
3. Jalankan *notebook* dari awal hingga akhir.
4. Grafik pelatihan (Loss/Accuracy) dan *Confusion Matrix* akan dirender secara otomatis pada *cell* evaluasi.

## Struktur Repository

```text
sesi-6-cnn/
├── c
├── README.md
└── requirements.txt
