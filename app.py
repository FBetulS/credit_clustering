import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Sayfa ayarları
st.set_page_config(page_title="Kredi Kartı Segmentasyonu", layout="wide")
st.title("📊 Kredi Kartı Müşteri Segmentasyonu")

@st.cache_resource
def load_data_and_models():
    try:
        # Veri setini yükle ve NaN'leri temizle
        df = pd.read_csv("CC GENERAL.csv")
        
        # Eksik verileri kontrol et
        if df.isnull().sum().sum() > 0:
            st.warning(f"Veri setinde {df.isnull().sum().sum()} eksik değer bulundu. Temizleniyor...")
            df = df.dropna()
        
        # Model ve scaler'ı yükle
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        return df, kmeans, scaler
        
    except Exception as e:
        st.error(f"Hata oluştu: {str(e)}")
        st.stop()

df, kmeans, scaler = load_data_and_models()

# Eğer CLUSTER sütunu yoksa tahmin yap
if 'CLUSTER' not in df.columns:
    try:
        clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT", "PAYMENTS"]]
        
        # Son bir NaN kontrolü ve temizleme
        clustering_data = clustering_data.dropna()
        
        # Scaler ile dönüştürme
        clustering_scaled = scaler.transform(clustering_data)
        
        # Tahmin yap
        df = df.loc[clustering_data.index]  # Aynı indeksleri koru
        df['CLUSTER'] = kmeans.predict(clustering_scaled)
        
    except Exception as e:
        st.error(f"Tahmin yaparken hata: {str(e)}")
        st.stop()


# Sidebar
st.sidebar.header("🔍 Müşteri Bilgilerini Girin")
balance = st.sidebar.number_input("Bakiye", min_value=0.0, value=1000.0)
purchases = st.sidebar.number_input("Harcamalar", min_value=0.0, value=500.0)
credit_limit = st.sidebar.number_input("Kredi Limiti", min_value=0.0, value=3000.0)
payments = st.sidebar.number_input("Ödemeler", min_value=0.0, value=1000.0)

# Tahmin yap
if st.sidebar.button("Segmenti Belirle"):
    input_data = np.array([[balance, purchases, credit_limit, payments]])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]
    st.sidebar.success(f"✅ Bu müşteri **Segment {cluster}** grubuna aittir")

# Ana sayfa
tab1, tab2 = st.tabs(["📈 Grafikler", "📊 İstatistikler"])

with tab1:
    st.header("Müşteri Segmentleri Dağılımı")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'purple']
    
    for i in range(4):
        cluster_data = df[df['CLUSTER'] == i]
        ax.scatter(cluster_data['BALANCE'], 
                  cluster_data['PURCHASES'],
                  c=colors[i],
                  label=f'Segment {i}',
                  alpha=0.6)
    
    ax.set_xlabel('Bakiye')
    ax.set_ylabel('Harcamalar')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.header("Segment İstatistikleri")
    
    # Temel istatistikler
    st.subheader("Müşteri Sayıları")
    cluster_counts = df['CLUSTER'].value_counts().sort_index()
    st.bar_chart(cluster_counts)
    
    # Ortalamalar
    st.subheader("Ortalama Değerler")
    cluster_stats = df.groupby('CLUSTER')[['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']].mean()
    st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))

# Debug bilgileri (isteğe bağlı)
with st.expander("ℹ️ Sistem Bilgileri"):
    st.write(f"Toplam Müşteri Sayısı: {len(df)}")
    st.write("Model Bilgisi:", kmeans)
    st.write("Örnek Veri:", df.head(2))