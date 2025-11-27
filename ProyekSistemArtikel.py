from flask import Flask, render_template_string, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# === Inisialisasi Flask ===
app = Flask(__name__)

# === Path files ===
DATASET_PATH = "dataset_gabungan_training.csv"
PREPROCESSED_PATH = "dataset_preprocessed.csv"
MODEL_PATH = "model_naivebayes.pkl"
VECTORIZER_PATH = "vectorizer_tfidf.pkl"

# === Preprocessing sederhana (lebih cepat dari Sastrawi) ===
STOPWORDS = set("yang dan di ke dari untuk pada adalah ini itu dengan juga tidak atau sebagai ada telah tapi kalau saya kita dia mereka kami oleh karena olehnya saja masih baru akan lebih sudah lagi semua setiap suatu tanpa seorang seorangnya sebuah".split())

def clean_text(text):
    """Preprocessing cepat tanpa Sastrawi"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    # Stemming sederhana (hapus akhiran umum)
    tokens = []
    for word in text.split():
        if word not in STOPWORDS:
            # Hapus akhiran -kan, -an, -i, -nya
            word = re.sub(r'(kan|an|i|nya)$', '', word)
            if len(word) > 2:  # Minimal 3 karakter
                tokens.append(word)
    
    return " ".join(tokens)

# === Load atau buat data preprocessed ===
if os.path.exists(PREPROCESSED_PATH):
    print("="*70)
    print("✅ LOADING DATA PREPROCESSED (CEPAT!)")
    print("="*70)
    df = pd.read_csv(PREPROCESSED_PATH)
    print(f"✅ Data loaded: {len(df)} dokumen")
else:
    print("="*70)
    print("⏳ PREPROCESSING DATA PERTAMA KALI...")
    print("="*70)
    df = pd.read_csv(DATASET_PATH)
    print(f"Total data: {len(df)}")
    
    print("\n⏳ Memproses preprocessing (akan disimpan untuk penggunaan selanjutnya)...")
    df["clean_text"] = df["content"].apply(clean_text)
    
    # Simpan hasil preprocessing
    df.to_csv(PREPROCESSED_PATH, index=False)
    print(f"✅ Preprocessing selesai dan disimpan ke {PREPROCESSED_PATH}")

print(f"Distribusi kategori:\n{df['kategori'].value_counts()}")
print("="*70)

# === Encode label ===
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["kategori"])

# === Load atau train model ===
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    print("\n✅ LOADING MODEL & VECTORIZER (INSTANT!)")
    print("="*70)
    
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    print("✅ Model dan Vectorizer loaded!")
    
    # Transform all data untuk retrieval
    tfidf_all = vectorizer.transform(df["clean_text"])
    
else:
    print("\n⏳ TRAINING MODEL PERTAMA KALI...")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label_enc"], 
        test_size=0.2, random_state=42, stratify=df["label_enc"]
    )
    
    print(f"Data training: {len(X_train)} dokumen")
    print(f"Data testing: {len(X_test)} dokumen")
    
    # TF-IDF
    print("\n⏳ Membangun TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Statistik TF-IDF
    print(f"\nStatistik TF-IDF Matrix:")
    print(f"Shape: {X_train_tfidf.shape}")
    print(f"Sparsity: {(1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])) * 100:.2f}%")
    
    # Train model
    print("\n⏳ Melatih Multinomial Naive Bayes...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    # Evaluasi
    print("\n" + "="*70)
    print("EVALUASI MODEL")
    print("="*70)
    
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=2))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Transform all data
    tfidf_all = vectorizer.transform(df["clean_text"])
    
    # Analisis Cosine Similarity (sample)
    print("\n" + "="*70)
    print("ANALISIS COSINE SIMILARITY (SAMPLE)")
    print("="*70)
    
    sample_size = min(200, len(df))
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    tfidf_sample = tfidf_all[sample_indices]
    
    print(f"\nMenghitung cosine similarity untuk {sample_size} dokumen...")
    similarity_sample = cosine_similarity(tfidf_sample)
    
    sim_values = similarity_sample[np.triu_indices_from(similarity_sample, k=1)]
    print(f"\nRata-rata similarity: {sim_values.mean():.4f}")
    print(f"Median similarity: {np.median(sim_values):.4f}")
    print(f"Std similarity: {sim_values.std():.4f}")
    
    # Simpan model
    print("\n⏳ Menyimpan model dan vectorizer...")
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model disimpan ke {MODEL_PATH}")
    print(f"✅ Vectorizer disimpan ke {VECTORIZER_PATH}")
    print("="*70)

print("\n" + "="*70)
print("✅ SISTEM SIAP DIGUNAKAN!")
print("="*70)

# === Template HTML ===
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistem Deteksi Hoaks</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1B3C53 0%, #234C6A 50%, #456882 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: white;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        .header h1 {
            color: #1B3C53;
            font-size: 2.5em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }
        .header p {
            color: #456882;
            font-size: 1.1em;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
            border-top: 4px solid #1B3C53;
        }
        .stat-card:hover {
            transform: translateY(-5px);
        }
        .stat-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #1B3C53;
            margin-bottom: 5px;
        }
        .stat-label {
            color: #456882;
            font-size: 1em;
        }
        .search-box {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        .search-form {
            display: flex;
            gap: 15px;
        }
        .search-input {
            flex: 1;
            padding: 18px 25px;
            border: 2px solid #D2C1B6;
            border-radius: 50px;
            font-size: 1.1em;
            transition: all 0.3s;
        }
        .search-input:focus {
            outline: none;
            border-color: #1B3C53;
            box-shadow: 0 0 0 3px rgba(27, 60, 83, 0.1);
        }
        .search-btn {
            padding: 18px 45px;
            background: linear-gradient(135deg, #1B3C53 0%, #234C6A 100%);
            color: white;
            border: none;
            border-radius: 50px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .search-btn:hover {
            transform: scale(1.05);
            background: linear-gradient(135deg, #234C6A 0%, #456882 100%);
        }
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
            background: white;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        .spinner {
            border: 4px solid #D2C1B6;
            border-top: 4px solid #1B3C53;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .results-header {
            background: white;
            padding: 20px 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-left: 5px solid #1B3C53;
        }
        .results-header h2 {
            color: #1B3C53;
            font-size: 1.5em;
        }
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .result-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
            cursor: pointer;
            border-left: 5px solid #456882;
        }
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(27, 60, 83, 0.2);
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 15px;
        }
        .result-title {
            font-size: 1.3em;
            font-weight: 600;
            color: #1B3C53;
            flex: 1;
            margin-right: 15px;
        }
        .result-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            white-space: nowrap;
        }
        .badge-hoaks {
            background: #fee;
            color: #c33;
        }
        .badge-fakta {
            background: #e8f5e9;
            color: #2e7d32;
        }
        .result-snippet {
            color: #456882;
            line-height: 1.6;
            margin-bottom: 15px;
        }
        .result-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-top: 15px;
            border-top: 1px solid #D2C1B6;
        }
        .confidence {
            color: #1B3C53;
            font-weight: 600;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(27, 60, 83, 0.8);
            overflow-y: auto;
        }
        .modal-content {
            background: white;
            margin: 50px auto;
            padding: 0;
            border-radius: 20px;
            max-width: 800px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            animation: slideDown 0.3s;
        }
        @keyframes slideDown {
            from { transform: translateY(-50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        .modal-header {
            padding: 30px;
            border-bottom: 2px solid #D2C1B6;
            position: relative;
            background: linear-gradient(135deg, #1B3C53 0%, #234C6A 100%);
            border-radius: 20px 20px 0 0;
        }
        .modal-title {
            font-size: 1.8em;
            color: white;
            padding-right: 40px;
        }
        .close {
            position: absolute;
            right: 25px;
            top: 25px;
            font-size: 2em;
            cursor: pointer;
            color: white;
            transition: color 0.3s;
        }
        .close:hover {
            color: #D2C1B6;
        }
        .modal-body {
            padding: 30px;
        }
        .alert-box {
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .alert-hoaks {
            background: #fee;
            border-left: 4px solid #c33;
        }
        .alert-fakta {
            background: #e8f5e9;
            border-left: 4px solid #2e7d32;
        }
        .alert-icon {
            font-size: 2em;
        }
        .alert-content h3 {
            margin-bottom: 5px;
            font-size: 1.3em;
            color: #1B3C53;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 25px;
        }
        .info-item {
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 3px solid #1B3C53;
        }
        .info-label {
            font-size: 0.9em;
            color: #456882;
            margin-bottom: 5px;
        }
        .info-value {
            font-size: 1.2em;
            font-weight: 600;
            color: #1B3C53;
        }
        .content-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            border: 1px solid #D2C1B6;
        }
        .content-box h4 {
            margin-bottom: 15px;
            color: #1B3C53;
        }
        .content-text {
            color: #456882;
            line-height: 1.8;
        }
        .tips-box {
            background: #fff8e1;
            border-left: 4px solid #f57c00;
            padding: 20px;
            border-radius: 10px;
        }
        .tips-box h4 {
            margin-bottom: 15px;
            color: #e65100;
        }
        .tips-box ul {
            margin-left: 20px;
            color: #f57c00;
            line-height: 1.8;
        }
        .no-results {
            background: white;
            border-radius: 15px;
            padding: 60px;
            text-align: center;
        }
        .no-results-icon {
            font-size: 5em;
            margin-bottom: 20px;
            opacity: 0.3;
            color: #1B3C53;
        }
        .no-results h3 {
            color: #1B3C53;
            margin-bottom: 10px;
        }
        .no-results p {
            color: #456882;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Sistem Temu Kembali Deteksi Hoaks</h1>
            <p>Implementasi Temu Kembali Berita Indonesia dengan Integrasi Deteksi Hoaks Menggunakan TF-IDF dan Naive Bayes</p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-icon">💾</div>
                <div class="stat-value">{{ total_news }}</div>
                <div class="stat-label">Total Berita</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📊</div>
                <div class="stat-value">TF-IDF</div>
                <div class="stat-label">Metode Pencarian</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🤖</div>
                <div class="stat-value">Naive Bayes</div>
                <div class="stat-label">Algoritma ML</div>
            </div>
        </div>

        <div class="search-box">
            <form method="POST" class="search-form" onsubmit="showLoading()">
                <input type="text" name="query" class="search-input" 
                       placeholder="🔍 Masukkan kata kunci atau teks berita yang ingin diverifikasi..." 
                       required value="{{ query or '' }}">
                <button type="submit" class="search-btn">
                    🔍 Analisis
                </button>
            </form>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Menganalisis berita...</p>
        </div>

        {% if results %}
        <div class="results-header">
            <h2>🎯 Ditemukan {{ results|length }} hasil untuk: "{{ query }}"</h2>
        </div>

        <div class="results-grid">
            {% for item in results %}
            <div class="result-card" onclick="openModal({{ loop.index0 }})">
                <div class="result-header">
                    <div class="result-title">{{ item.title }}</div>
                    <div class="result-badge {% if item.label == 'hoaks' %}badge-hoaks{% else %}badge-fakta{% endif %}">
                        {% if item.label == 'hoaks' %}⚠️ HOAKS{% else %}✅ FAKTA{% endif %}
                    </div>
                </div>
                <div class="result-snippet">{{ item.snippet }}</div>
                <div class="result-footer">
                    <span class="confidence">Confidence: {{ item.prob }}%</span>
                    <span style="color: #456882;">Klik untuk detail →</span>
                </div>
            </div>

            <div id="modal{{ loop.index0 }}" class="modal">
                <div class="modal-content">
                    <div class="modal-header">
                        <h2 class="modal-title">📰 Detail Informasi Berita</h2>
                        <span class="close" onclick="closeModal({{ loop.index0 }})">&times;</span>
                    </div>
                    <div class="modal-body">
                        <h3 style="margin-bottom: 20px; color: #1B3C53;">{{ item.title }}</h3>
                        
                        <div class="alert-box {% if item.label == 'hoaks' %}alert-hoaks{% else %}alert-fakta{% endif %}">
                            <div class="alert-icon">{% if item.label == 'hoaks' %}⚠️{% else %}✅{% endif %}</div>
                            <div class="alert-content">
                                <h3>{% if item.label == 'hoaks' %}⚠️ TERDETEKSI HOAKS{% else %}✅ TERDETEKSI FAKTA{% endif %}</h3>
                                <p>{% if item.label == 'hoaks' %}Berita ini terindikasi sebagai informasi PALSU/HOAKS. Mohon untuk tidak menyebarkan dan verifikasi dari sumber terpercaya.{% else %}Berita ini terindikasi sebagai informasi VALID/FAKTA berdasarkan analisis sistem.{% endif %}</p>
                            </div>
                        </div>

                        <h4 style="margin-bottom: 15px; color: #1B3C53;">📋 Informasi Analisis</h4>
                        <div class="info-grid">
                            <div class="info-item">
                                <div class="info-label">🏷️ Klasifikasi</div>
                                <div class="info-value">{{ item.label|upper }}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">📈 Tingkat Kepercayaan</div>
                                <div class="info-value">{{ item.prob }}%</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">🔬 Metode Deteksi</div>
                                <div class="info-value" style="font-size: 0.9em;">Naive Bayes dengan TF-IDF</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">🎯 Kemiripan Query</div>
                                <div class="info-value" style="font-size: 0.9em;">Cosine Similarity</div>
                            </div>
                        </div>

                        <div class="content-box">
                            <h4>📄 Isi Berita</h4>
                            <div class="content-text">{{ item.full_content }}</div>
                        </div>

                        {% if item.label == 'hoaks' %}
                        <div class="tips-box">
                            <h4>💡 Tips Menghindari Hoaks:</h4>
                            <ul>
                                <li>Cek sumber berita dari media terpercaya</li>
                                <li>Jangan langsung percaya judul sensasional</li>
                                <li>Cari berita serupa dari sumber lain</li>
                                <li>Verifikasi melalui fact-checking websites</li>
                            </ul>
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        {% elif query %}
        <div class="no-results">
            <div class="no-results-icon">🔍</div>
            <h3>Tidak ada hasil ditemukan</h3>
            <p>Coba gunakan kata kunci yang berbeda</p>
        </div>
        {% endif %}
    </div>

    <script>
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
        }

        function openModal(index) {
            document.getElementById('modal' + index).style.display = 'block';
        }

        function closeModal(index) {
            document.getElementById('modal' + index).style.display = 'none';
        }

        window.onclick = function(event) {
            if (event.target.classList.contains('modal')) {
                event.target.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

# === Fungsi untuk mencari berita mirip ===
def retrieve_news(query, top_k=8):
    q_clean = clean_text(query)
    q_vec = vectorizer.transform([q_clean])
    sims = cosine_similarity(q_vec, tfidf_all).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    
    results = []
    for i in top_idx:
        if sims[i] < 0.1:
            continue
        
        text = df.iloc[i]["content"]
        pred = model.predict(vectorizer.transform([df.iloc[i]["clean_text"]]))[0]
        prob = model.predict_proba(vectorizer.transform([df.iloc[i]["clean_text"]]))[0][pred]
        label = le.inverse_transform([pred])[0]
        
        snippet = text[:400] + ("..." if len(text) > 400 else "")
        
        results.append({
            "title": df.iloc[i]["title"] if "title" in df.columns else "Berita Tanpa Judul",
            "snippet": snippet,
            "full_content": text,
            "label": label.lower(),
            "prob": round(prob * 100, 1)
        })
    
    return results

# === Route utama ===
@app.route("/", methods=["GET", "POST"])
def index():
    query = None
    results = None
    
    if request.method == "POST":
        query = request.form["query"]
        results = retrieve_news(query, top_k=8)
    
    return render_template_string(
        HTML_TEMPLATE,
        query=query,
        results=results,
        total_news=len(df)
    )

# === Jalankan server ===
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 SERVER SIAP BERJALAN - SUPER CEPAT!")
    print("="*70)
    print("Akses aplikasi di: http://localhost:5000")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)