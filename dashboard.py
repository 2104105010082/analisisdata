import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans

# === SETUP DASAR ===
st.set_page_config(page_title="Dashboard Bike Sharing", layout="wide")

# === LOAD DATA ===
DATA_PATH = "dataclean_analisis.csv"

if not os.path.exists(DATA_PATH):
    st.error(f"File {DATA_PATH} tidak ditemukan! Upload file ini ke repositori GitHub.")
    st.stop()

df = pd.read_csv(DATA_PATH)

if df.empty:
    st.error("Data tidak dapat di-load! Pastikan file tidak kosong.")
    st.stop()

# === PREPROCESSING ===
df['dteday'] = pd.to_datetime(df['dteday'], errors='coerce')
df.dropna(subset=['dteday'], inplace=True)

reverse_season_map = {
    'Spring': 1,
    'Summer': 2,
    'Fall': 3,
    'Winter': 4
}
df['season'] = df['season'].map(reverse_season_map)
df['season'] = df['season'].fillna(0).astype(int)

# Tambahkan kolom year_month jika belum ada
if 'year_month' not in df.columns:
    df['year_month'] = df['dteday'].dt.to_period('M').astype(str)

# === HEADER ===
st.title("\U0001F4CA Dashboard Bike Sharing")
st.markdown("Visualisasi data dan insight dari dataset Bike Sharing")

# === SIDEBAR ===
with st.sidebar:
    st.header("Filter Musim")
    selected_season_num = st.selectbox("Pilih Musim", sorted(df["season"].dropna().unique()))
    selected_season = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}.get(selected_season_num, "Unknown")
    filtered_df = df[df["season"] == selected_season_num]
    st.write(f"Musim yang dipilih: {selected_season}")
    st.write(f"Jumlah data: {filtered_df.shape[0]} baris")

# === TAB ===
tab1, tab2 = st.tabs(["\U0001F4C8 Visualisasi Umum", "\U0001F4CA Pertanyaan Analitik"])

# === TAB 1: Visualisasi Umum ===
with tab1:
    st.subheader(f"Distribusi Penyewaan Sepeda - Musim: {selected_season}")
    if filtered_df.empty:
        st.warning(f"Tidak ada data untuk musim {selected_season}.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_df["cnt"], bins=30, kde=True, ax=ax)
        ax.set_xlabel("Jumlah Penyewaan")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)

    st.subheader("Korelasi Antar Variabel Numerik")
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# === TAB 2: Pertanyaan Analitik ===
with tab2:
    st.subheader("\U0001F50D Pertanyaan Analitik & Visualisasi")

    # 1. Pola penggunaan sepeda berdasarkan waktu
    st.markdown("1. Bagaimana pola penggunaan sepeda berdasarkan waktu (harian, bulanan, musiman)?")
    monthly_trend = df.groupby('year_month')['cnt'].sum()
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_trend.plot(marker='o', linestyle='-', color='blue', ax=ax)
    ax.set_xlabel("Waktu (Bulan-Tahun)")
    ax.set_ylabel("Total Penyewaan Sepeda")
    ax.set_title("Tren Penggunaan Sepeda dari Waktu ke Waktu")
    plt.grid(True)
    st.pyplot(fig)

    # 2. Korelasi variabel numerik tahun 2012
    st.markdown("2. Apa saja tiga variabel numerik yang paling berkorelasi dengan jumlah peminjaman sepeda bulanan pada tahun 2012?")
    df_2012 = df[df['dteday'].dt.year == 2012].copy()
    df_2012['year_month'] = df_2012['dteday'].dt.to_period('M')
    monthly_avg = df_2012.groupby('year_month').mean(numeric_only=True)
    correlation = monthly_avg.corr()['cnt'].drop('cnt')
    top3_corr = correlation.abs().sort_values(ascending=False).head(3)
    fig, ax = plt.subplots(figsize=(8, 5))
    top3_corr.sort_values().plot(kind='barh', color='skyblue', ax=ax)
    ax.set_title("Top 3 Korelasi Variabel dengan Jumlah Peminjaman Sepeda Bulanan (2012)")
    ax.set_xlabel("Korelasi (|nilai absolut|)")
    ax.grid(True)
    st.pyplot(fig)

    # 3. Segmentasi pengguna (Fall 2012)
    st.markdown("3. Bagaimana segmentasi pengguna berdasarkan pola peminjaman mereka pada musim gugur 2012?")
    fall_2012_df = df[(df['dteday'].dt.year == 2012) & (df['season'] == 3)].copy()

    if fall_2012_df.empty:
        st.error("❌ Tidak ada data untuk musim gugur tahun 2012. Silakan cek kembali file CSV.")
        st.dataframe(df[df['dteday'].dt.year == 2012][['dteday', 'season']].drop_duplicates())
    else:
        segmentation_data = fall_2012_df[['casual', 'registered']]
        kmeans = KMeans(n_clusters=3, random_state=42)
        fall_2012_df['Cluster'] = kmeans.fit_predict(segmentation_data)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=fall_2012_df,
            x='casual',
            y='registered',
            hue='Cluster',
            palette='Set1',
            s=80,
            ax=ax
        )
        ax.set_title("Segmentasi Pengguna Sepeda (Musim Gugur 2012)")
        ax.set_xlabel("Jumlah Peminjaman Kasual")
        ax.set_ylabel("Jumlah Peminjaman Terdaftar")
        ax.grid(True)
        ax.legend(title='Cluster')
        st.pyplot(fig)

# === FOOTER ===
st.markdown("---")
st.markdown("By: Syakira - Dashboard Bike Sharing")