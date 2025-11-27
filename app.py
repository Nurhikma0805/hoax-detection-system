import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# === Konfigurasi Streamlit ===
st.set_page_config(
    page_title="Sistem Deteksi Hoaks",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === Custom CSS (Sama seperti Flask) ===
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1B3C53 0%, #234C6A 50%, #456882 100%);
        padding: 20px;
    }
    .stApp {
        background: linear-gradient(135deg, #1B3C53 0%, #234C6A 50%, #456882 100%);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .header-box {
        background: white;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 30px;
    }
    .stat-container {
        display: flex;
        gap: 20px;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }
    .stat-card {
        background: white;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-top: 4px solid #1B3C53;
        flex: 1;
        min-width: 250px;
    }
    .search-box {
        background: white;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 30px;
    }
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #456882;
    }
    .badge-hoaks {
        background: #fee;
        color: #c33;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
    }
    .badge-fakta {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
    }
    .alert-hoaks {
        background: #fee;
        border-left: 4px solid #c33;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .alert-fakta {
        background: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .tips-box {
        background: #fff8e1;
        border-left: 4px solid #f57c00;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .stTextInput input {
        border-radius: 50px !important;
        border: 2px solid #D2C1B6 !important;
        padding: 18px 25px !important;
        font-size: 1.1em !important;
    }
    .stButton button {
        background: linear-gradient(135deg, #1B3C53 0%, #234C6A 100%);
        color: white;
        border-radius: 50px;
        padding: 18px 45px;
        font-size: 1.1em;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #234C6A 0%, #456882 100%);
    }
</style>
""", unsafe_allow_html=True)

# === Path files ===
DATASET_PATH = "dataset_gabungan_training.csv"
PREPROCESSED_PATH = "dataset_preprocessed.csv"
MODEL_PATH = "model_naivebayes.pkl"
VECTORIZER_PATH = "vectorizer_tfidf.pkl"

# === Preprocessing ===
STOPWORDS = set("yang dan di ke dari untuk pada adalah ini itu dengan juga tidak atau sebagai ada telah tapi kalau saya kita dia mereka kami oleh karena olehnya saja masih baru akan lebih sudah lagi semua setiap suatu tanpa seorang seorangnya sebuah".split())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    tokens = []
    for word in text.split():
        if word not in STOPWORDS:
            word = re.sub(r'(kan|an|i|nya)$', '', word)
            if len(word) > 2:
                tokens.append(word)
    
    return " ".join(tokens)

# === Load Data ===
@st.cache_data
def load_data():
    if os.path.exists(PREPROCESSED_PATH):
        df = pd.read_csv(PREPROCESSED_PATH)
    else:
        df = pd.read_csv(DATASET_PATH)
        df["clean_text"] = df["content"].apply(clean_text)
        df.to_csv(PREPROCESSED_PATH, index=False)
    
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["kategori"])
    return df, le

# === Load Model ===
@st.cache_resource
def load_model(df, le):
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            df["clean_text"], df["label_enc"], 
            test_size=0.2, random_state=42, stratify=df["label_enc"]
        )
        
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)
        
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
    
    tfidf_all = vectorizer.transform(df["clean_text"])
    return model, vectorizer, tfidf_all

# === Retrieve Function ===
def retrieve_news(query, df, model, vectorizer, tfidf_all, le, top_k=8):
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

# === Main App ===
try:
    df, le = load_data()
    model, vectorizer, tfidf_all = load_model(df, le)
    
    # Header
    st.markdown("""
    <div class="header-box">
        <h1 style="color: #1B3C53; font-size: 2.5em; margin-bottom: 10px;">
            🛡️ Sistem Temu Kembali Deteksi Hoaks
        </h1>
        <p style="color: #456882; font-size: 1.1em;">
            Implementasi Temu Kembali Berita Indonesia dengan Integrasi Deteksi Hoaks Menggunakan TF-IDF dan Naive Bayes
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats
    st.markdown(f"""
    <div class="stat-container">
        <div class="stat-card">
            <div style="font-size: 3em; margin-bottom: 15px;">💾</div>
            <div style="font-size: 2.5em; font-weight: bold; color: #1B3C53;">{len(df)}</div>
            <div style="color: #456882;">Total Berita</div>
        </div>
        <div class="stat-card">
            <div style="font-size: 3em; margin-bottom: 15px;">📊</div>
            <div style="font-size: 2.5em; font-weight: bold; color: #1B3C53;">TF-IDF</div>
            <div style="color: #456882;">Metode Pencarian</div>
        </div>
        <div class="stat-card">
            <div style="font-size: 3em; margin-bottom: 15px;">🤖</div>
            <div style="font-size: 2.5em; font-weight: bold; color: #1B3C53;">Naive Bayes</div>
            <div style="color: #456882;">Algoritma ML</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Search Box
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    query = st.text_input("", placeholder="🔍 Masukkan kata kunci atau teks berita yang ingin diverifikasi...", key="search", label_visibility="collapsed")
    search_button = st.button("🔍 Analisis", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Results
    if search_button and query:
        with st.spinner("Menganalisis berita..."):
            results = retrieve_news(query, df, model, vectorizer, tfidf_all, le, top_k=8)
        
        if results:
            st.markdown(f"""
            <div style="background: white; padding: 20px 30px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); border-left: 5px solid #1B3C53;">
                <h2 style="color: #1B3C53; font-size: 1.5em;">🎯 Ditemukan {len(results)} hasil untuk: "{query}"</h2>
            </div>
            """, unsafe_allow_html=True)
            
            for idx, item in enumerate(results):
                with st.expander(f"{'⚠️ HOAKS' if item['label'] == 'hoaks' else '✅ FAKTA'} - {item['title']}", expanded=False):
                    badge_class = "badge-hoaks" if item['label'] == 'hoaks' else "badge-fakta"
                    alert_class = "alert-hoaks" if item['label'] == 'hoaks' else "alert-fakta"
                    
                    st.markdown(f'<div class="{badge_class}">{"⚠️ HOAKS" if item["label"] == "hoaks" else "✅ FAKTA"}</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="{alert_class}">
                        <h3 style="margin-bottom: 10px;">{"⚠️ TERDETEKSI HOAKS" if item['label'] == 'hoaks' else "✅ TERDETEKSI FAKTA"}</h3>
                        <p>{"Berita ini terindikasi sebagai informasi PALSU/HOAKS. Mohon untuk tidak menyebarkan dan verifikasi dari sumber terpercaya." if item['label'] == 'hoaks' else "Berita ini terindikasi sebagai informasi VALID/FAKTA berdasarkan analisis sistem."}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🏷️ Klasifikasi", item['label'].upper())
                        st.metric("🔬 Metode Deteksi", "Naive Bayes + TF-IDF")
                    with col2:
                        st.metric("📈 Tingkat Kepercayaan", f"{item['prob']}%")
                        st.metric("🎯 Kemiripan Query", "Cosine Similarity")
                    
                    st.markdown("#### 📄 Isi Berita")
                    st.write(item['full_content'])
                    
                    if item['label'] == 'hoaks':
                        st.markdown("""
                        <div class="tips-box">
                            <h4 style="color: #e65100; margin-bottom: 15px;">💡 Tips Menghindari Hoaks:</h4>
                            <ul style="color: #f57c00; line-height: 1.8;">
                                <li>Cek sumber berita dari media terpercaya</li>
                                <li>Jangan langsung percaya judul sensasional</li>
                                <li>Cari berita serupa dari sumber lain</li>
                                <li>Verifikasi melalui fact-checking websites</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: white; border-radius: 15px; padding: 60px; text-align: center;">
                <div style="font-size: 5em; margin-bottom: 20px; opacity: 0.3; color: #1B3C53;">🔍</div>
                <h3 style="color: #1B3C53; margin-bottom: 10px;">Tidak ada hasil ditemukan</h3>
                <p style="color: #456882;">Coba gunakan kata kunci yang berbeda</p>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"""
    ⚠️ **Error:** Dataset tidak ditemukan!
    
    Pastikan file `dataset_gabungan_training.csv` ada di folder yang sama dengan `app.py`
    
    **Detail error:** {str(e)}
    """)