import streamlit as st
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import joblib
import json
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_excel("datasets/veriler_hepsi_grupB.xlsx")
df = df_.copy()

def preprocess_data(df):
    # Fiyat(USD) sütununu oluşturma
    df["Fiyat(USD)"] = df["Fiyat(TL)"] / 32

    # Sayısal ve kategorik sütunları tanımlama
    num_cols = ["Dahili Hafıza (GB)", "Ekran Boyutu(inch)", "RAMG(GB)", "Kamera(MP)", "Ön Kamera(MP)"]
    cat_cols = ["Model", "Renk", "Garanti", "Kimden", "Durumu"]

    # XS sütununu düşürme
    if "XS" in df.columns:
        df = df.drop(columns=["XS"], axis=1)

    # Pil(%) sütunundaki eksik değerleri medyan ile doldurma
    df["Pil(%)"] = df.groupby(["Model", "Durumu"])["Pil(%)"].transform(lambda x: x.fillna(x.median()))

    # log_Fiyat(USD) sütununu oluşturma
    df["log_Fiyat(USD)"] = np.log1p(df["Fiyat(USD)"])

    # Sayısal sütunları ölçekleme
    scaler = MinMaxScaler()
    scaled_num_cols = scaler.fit_transform(df[num_cols])
    scaled_num_cols = pd.DataFrame(scaled_num_cols, columns=num_cols)

    # Kategorik sütunları dönüştürme
    for col in cat_cols:
        df[col] = df[col].astype(str)

    # Kategorik sütunları kodlama
    ohe = OneHotEncoder(drop='first')
    encoded_cols = ohe.fit_transform(df[cat_cols])
    encoded_cols = pd.DataFrame(encoded_cols.toarray(), columns=ohe.get_feature_names_out(cat_cols))

    # İşlenmiş verileri birleştirme
    merged_df = pd.concat([encoded_cols, scaled_num_cols], axis=1)

    return merged_df, df["log_Fiyat(USD)"], cat_cols, num_cols, ohe

X, y, cat_cols, num_cols, ohe = preprocess_data(df)

# En iyi modeli yükleyin
best_ridge_model = joblib.load("best_ridge_model.pkl")

# Veri yükleme
df_ = pd.read_excel("datasets/veriler_hepsi_grupB.xlsx")

# Streamlit arayüzü oluşturma
st.title("iPhone Fiyat Tahmini Uygulaması")

# iPhone modellerini seçme
selected_model = st.selectbox("iPhone Modeli Seçin", df["Model"].unique())
selected_renk = st.selectbox("iPhone Rengini Seçin", df["Renk"].unique())
selected_garanti = st.selectbox("iPhone Garanti Durumunu Seçin", df["Garanti"].unique())
selected_kimden = st.selectbox("iPhone Kimden Olduğunu Seçin", df["Kimden"].unique())
selected_durumu = st.selectbox("iPhone Durumunu Seçin", df["Durumu"].unique())
selected_onkamera = st.number_input("iPhone ÖN KAMERA(MP) Seçin", min_value=0.0, step=1.0)
selected_ram = st.number_input("iPhone RAM(GB) Seçin", min_value=0.0, step=1.0)
selected_dahilihafiza = st.number_input("iPhone Dahili Hafıza (GB) Seçin", min_value=0.0, step=1.0)
selected_ekranboyut = st.number_input("iPhone Ekran Boyutu(inch)  Seçin", min_value=0.0, step=1.0)
selected_kamera = st.number_input("iPhone Kamera(MP) Seçin", min_value=0.0, step=1.0)
selected_pil = st.number_input("iPhone Pil Yüzdesi Seçin", min_value=0.0, step=1.0)

input_data = {
    "Model": selected_model,
    "Renk": selected_renk,
    "Garanti": selected_garanti,
    "Kimden": selected_kimden,
    "Durumu": selected_durumu,
    "Ön Kamera(MP)": selected_onkamera,
    "RAMG(GB)": selected_ram,
    "Dahili Hafıza (GB)": selected_dahilihafiza,
    "Ekran Boyutu(inch)": selected_ekranboyut,
    "Kamera(MP)": selected_kamera,
    "Pil(%)": selected_pil
}

A = pd.DataFrame([input_data])
cat_features = pd.DataFrame(ohe.transform(A[cat_cols]).toarray(), columns=ohe.get_feature_names_out(cat_cols))
num_features = pd.DataFrame(MinMaxScaler().fit_transform(A[num_cols]), columns=num_cols)
input_df = pd.concat([cat_features, num_features], axis=1)

st.write(cat_features)

# Seçilen özelliklerle tahmin yapma
if st.button("Tahmin Et"):
    # Tahmin yapma
    predicted_price = np.exp(best_ridge_model.predict(input_df)) - 1  # log dönüşümünü tersine çevirme

    # Tahmini fiyatı görüntüleme
    st.write(f"Seçilen iPhone modelinin tahmini fiyatı: {predicted_price[0]} USD")
