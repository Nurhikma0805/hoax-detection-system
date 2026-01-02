# 🛡️ Sistem Deteksi Hoaks - Information Retrieval

Sistem pencarian dan deteksi hoaks berita Indonesia menggunakan TF-IDF dan Naive Bayes.

## 🚀 Fitur
- Pencarian berita berbasis kata kunci
- Deteksi otomatis: HOAKS atau FAKTA
- Skor kepercayaan deteksi
- Interface web interaktif dengan Streamlit

## 🛠️ Teknologi
- Python 3.13.2
- Streamlit
- Scikit-learn (TF-IDF, Naive Bayes)
- Pandas, NumPy

## 📦 Instalasi
```bash
# Clone repository
git clone https://github.com/username/sistem-deteksi-hoaks.git
cd sistem-deteksi-hoaks

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

## 📋 Requirements
```
streamlit
pandas
scikit-learn
numpy
```

## 📊 Dataset
- Total: 1461 berita Indonesia
- Label: hoaks / fakta
- Format: CSV (title, content, kategori)

### 📸 Screenshot
## Dashboard Utama
<img width="1231" height="896" alt="Beranda" src="https://github.com/user-attachments/assets/29ef0096-5431-4c1a-b6a1-ac8503fb8987" />
*Tampilan halaman utama sistem dengan fitur pencarian dan hasil deteksi berita*

## Hasil Deteksi
<img width="932" height="752" alt="hasil" src="https://github.com/user-attachments/assets/a2674227-213f-4ec6-9d72-9828246a50eb" />
*Detail informasi analisis berita dengan klasifikasi dan tingkat kepercayaan sistem*
