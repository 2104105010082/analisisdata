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
    st.error(f"File {DATA_PATH} tidak ditemukan! Pastikan file ada di direktori yang benar.")
    st.stop()

df = pd.read_csv(DATA_PATH)

if df.empty:
    st.error("Data tidak dapat di-load! Pastikan file ada di direktori yang benar.")
    st.stop()

# === PREPROCESSING ===
df['dteday'] = pd.to_datetime(df['dteday'], errors='coerce')
df.dropna(subset=['dteday'], inplace=True)

# Mapping musim ke angka
reverse_season_map = {
    'Spring': 1,
    'Summer': 2,
    'Fall': 3,
    'Winter': 4
}

df['season'] = df['season'].map(reverse_season_map)
df['season'] = df['season'].fillna(0).astype(int)

# Buat kolom gabungan bulan-tahun untuk analisis waktu
df['year_month'] = df['dteday'].dt.to_period('M').astype(str)

# === HEADER ===
st.title("\U0001F4CA Dashboard Bike Sharing")
st.markdown("Visualisasi data dan insight dari dataset Bike Sharing")

# === SIDEBAR ===
with st.sidebar:
    st.header("Filter Musim")
    selected_season_num = st.selectbox("Pilih Musim", sorted(df["season"].dropna().unique()))
    selected_season = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}.get(selected_season_num, "Unknown")

    st.header("Filter Cuaca")
    selected_weather = st.multiselect(
        "Pilih Kondisi Cuaca (1: Cerah, 2: Mendung, 3/4: Buruk)",
        options=sorted(df["weathersit"].dropna().unique()),
        default=sorted(df["weathersit"].dropna().unique())
    )

    # Filter dataset
    filtered_df = df[(df["season"] == selected_season_num) & (df["weathersit"].isin(selected_weather))]

    st.write(f"Musim yang dipilih: {selected_season}")
    st.write(f"Cuaca yang dipilih: {selected_weather}")
    st.write(f"Jumlah data: {filtered_df.shape[0]} baris")

# === TAB ===
tab1, tab2 = st.tabs(["\U0001F4C8 Visualisasi Umum", "\U0001F4CA Pertanyaan Analitik"])

# === TAB 1: Visualisasi Umum ===
with tab1:
    st.subheader(f"Distribusi Penyewaan Sepeda - Musim: {selected_season} & Cuaca")
    if filtered_df.empty:
        st.warning(f"Tidak ada data untuk kombinasi musim {selected_season} dan cuaca yang dipilih.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_df["cnt"], bins=30, kde=True, ax=ax)
        ax.set_xlabel("Jumlah Penyewaan")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)

    st.subheader("Korelasi Antar Variabel Numerik")
    numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64'])
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# === TAB 2: Pertanyaan Analitik ===
with tab2:
    st.subheader("\U0001F50D Pertanyaan Analitik & Visualisasi")

    # 1. Pola penggunaan sepeda berdasarkan waktu
    st.markdown("**1. Bagaimana pola penggunaan sepeda berdasarkan waktu (harian, bulanan, musiman)?**")
    monthly_trend = df.groupby('year_month')['cnt'].sum()
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_trend.plot(marker='o', linestyle='-', color='blue', ax=ax)
    ax.set_xlabel("Waktu (Bulan-Tahun)")
    ax.set_ylabel("Total Penyewaan Sepeda")
    ax.set_title("Tren Penggunaan Sepeda dari Waktu ke Waktu")
    plt.grid(True)
    st.pyplot(fig)

# 2. Faktor yang mempengaruhi jumlah peminjaman sepeda (versi 2012)
st.markdown("**2. Apa saja tiga variabel numerik yang paling berkorelasi dengan jumlah peminjaman sepeda bulanan pada tahun 2012?**")

df_2012 = df[df['dteday'].dt.year == 2012].copy()
df_2012['year_month'] = df_2012['dteday'].dt.to_period('M')

monthly_avg = df_2012.groupby('year_month').mean(numeric_only=True).reset_index()
top_vars = ['registered', 'casual', 'temp', 'cnt']

# Line chart semua variabel
fig, ax = plt.subplots(figsize=(10, 6))
for col in top_vars:
    ax.plot(monthly_avg['year_month'].astype(str), monthly_avg[col], marker='o', label=col)
ax.set_title("Tren Bulanan: Variabel Utama vs Jumlah Peminjaman Sepeda (2012)")
ax.set_xlabel("Bulan")
ax.set_ylabel("Nilai Rata-rata")
plt.xticks(rotation=45)
ax.legend(title="Variabel")
st.pyplot(fig)

# Scatter plot dengan regresi linier untuk top 3 variabel
import numpy as np

for col in ['registered', 'casual', 'temp']:
    fig, ax = plt.subplots(figsize=(5, 4))
    x = monthly_avg[col]
    y = monthly_avg['cnt']
    ax.scatter(x, y)
    ax.set_title(f"Hubungan antara '{col}' dan Jumlah Peminjaman (cnt)")
    ax.set_xlabel(col)
    ax.set_ylabel("Jumlah Peminjaman Sepeda (cnt)")

    # Tambahkan garis regresi linier
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m*x + b, color='red')
    st.pyplot(fig)



    # 3. Segmentasi pengguna berdasarkan pola peminjaman pada musim gugur 2012
    st.markdown("**3. Bagaimana segmentasi pengguna berdasarkan pola peminjaman mereka pada musim gugur 2012?**")
    fall_2012_df = df[(df['dteday'].dt.year == 2012) & (df['season'] == 3)].copy()

    if fall_2012_df.empty:
        st.error("❌ Tidak ada data untuk musim gugur tahun 2012. Silakan cek kembali file CSV.")
        st.write("### Data yang ada di CSV untuk 2012, Season:")
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
st.markdown("By: Syakira - Dashboard Bike Sharing")
