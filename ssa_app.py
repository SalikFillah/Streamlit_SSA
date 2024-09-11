import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from my_function import SSA

# vertical menu
with st.sidebar:
    selected = option_menu("SSA", ["Pengenalan", "Unggah Data", "Config", 'Result'], 
                           icons=['house', 'cloud-upload', "gear", 'graph-up'],
                           menu_icon="activity", default_index=1)
    selected

# Konten berdasarkan menu yang dipilih
if selected == "Pengenalan":
    st.markdown("<h3 style='font-size: 24px;'>Singular Spectrum Analysis</h3>", unsafe_allow_html=True)
    st.write('''
             Metode SSA pada umumnya menguraikan deret waktu menjadi sekumpulan komponen yang dapat dijumlahkan yang masing-masing dikelompokkan sebagai **tren**, **periodisitas**, dan **noise**.
             Berikut beberapa hal yang perlu dipertimbangkan sebelum menjalankan metode SSA pada web ini:

             1. Data yang diterima harus berupa data **deret waktu univariat** dengan interval waktu yang konsisten antara setiap titik data.
             2. Anda bisa mengatur besar **panjang jendela (L)** sesuai dengan batasan yang telah ditetapkan.
             3. Hasil dari metode SSA berupa data **deret waktu baru** yang telah terbagi menjadi komponen-komponen diatas.
             ''')
    #st.image('bg.jpg', caption='Gambar Background')  # Ganti 'bg.jpg' dengan path gambar yang sesuai

elif selected == 'Unggah Data':
    st.write('''
             Format data yang diunggah harus berbentuk **CSV (*Comma-Separated Values*)** 
             ''')
    # Menambahkan tombol untuk upload file
    uploaded_file = st.file_uploader('Upload Data', type=["csv"])
    
    # Jika ada file yang diunggah, file tersebut akan diproses
    if uploaded_file is not None:
        # Membaca file CSV menggunakan pandas
        data = pd.read_csv(uploaded_file)
        
        # Menyimpan data ke session state
        st.session_state['data'] = data

        # Menampilkan beberapa baris dari file yang diunggah
        st.write("Berikut adalah tampilan data yang anda unggah:")
        st.write(data.head(10))
        st.write(f"Jumlah Kolom : {data.shape[1]}")
        st.write(f"Jumlah Baris : {data.shape[0]}")

elif selected == 'Config':
    # Memastikan data telah diunggah sebelum mengakses
    if 'data' in st.session_state:
        data = st.session_state['data']
        N = len(data)

        # Membatasi L antara 2 dan N/2
        min_value = 5
        max_value = N // 2  # Membatasi L hingga N/2 (dibulatkan ke bawah)

        # Membuat slider dengan batas L yang disesuaikan
        L = st.slider("Pilih Panjang Jendela (L)", min_value, max_value, (min_value + max_value) // 2)

        # Simpan nilai L di session state
        st.session_state['L'] = L
        
        st.write(f"L yang dipilih: {L}")
    else:
        st.write("Mohon unggah data terlebih dahulu di bagian **Unggah Data**.")

elif selected == 'Result':
    # Memastikan data dan L telah ditentukan sebelum menghasilkan output
    if 'data' in st.session_state and 'L' in st.session_state:
        data = st.session_state['data']
        L = st.session_state['L']
        
        # Contoh hasil: lakukan analisis SSA atau proses lainnya menggunakan L dan data
        ts_ntp = data['Nilai'].to_numpy()
        # Di sini kita misalkan memanggil fungsi SSA dari 'my_function'
        result = SSA(ts_ntp, L)

        # Tentukan indeks untuk masing-masing komponen
        # Misalkan d = 5, maka kita akan menggunakan indeks 0-4 untuk komponen
        trend_indices = [0]  # Misalnya komponen pertama adalah tren
        periodicity1_indices = [1, 2]  # Komponen kedua dan ketiga sebagai periodisitas1
        periodicity2_indices = [3, 4]  # Komponen keempat dan kelima sebagai periodisitas2
        noise_indices = list(range(5, result.d))  # Sisanya sebagai noise

        # Rekonstruksi setiap komponen
        trend = result.reconstruct(indices=trend_indices)
        periodicity1 = result.reconstruct(indices=periodicity1_indices)
        periodicity2 = result.reconstruct(indices=periodicity2_indices)
        noise = result.reconstruct(indices=noise_indices)

        # Gabungkan semua komponen menjadi satu DataFrame
        new_ntp = pd.DataFrame({
            'Tren': trend,
            'Periodisitas1': periodicity1,
            'Periodisitas2': periodicity2,
            'Noise': noise
        })

        # Gabungkan dengan data time-series asli
        ntp_ssa = pd.concat([data, new_ntp], axis=1)

        st.write(f"Berikut adalah tampilan data deret waktu baru setelah reduksi SSA dengan panjang L: {L}")
        st.write(ntp_ssa.head())
        st.write(f"Jumlah Kolom : {ntp_ssa.shape[1]}")
        st.write(f"Jumlah Baris : {ntp_ssa.shape[0]}")

        st.write('Distribusi Setiap Komponen Data Deret Waktu Baru')
        st.line_chart(ntp_ssa)

    else:
        st.write("Mohon **Unggah Data** dan pilih nilai L di bagian **Config** terlebih dahulu.")
