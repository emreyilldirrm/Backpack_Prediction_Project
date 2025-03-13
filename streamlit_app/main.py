import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# Sayfa başlığı ve yapılandırma
st.set_page_config(page_title="Tahmin Uygulaması", layout="wide")

st.title("📊 Veri Bilimi Tahmin Uygulaması")
st.write("""
Bu uygulama, makine öğrenmesi modeli kullanarak tahmin yapmaktadır.
Sol taraftaki menüden farklı sayfalara geçiş yapabilirsiniz.
""")

st.sidebar.title("📌 Menü")
st.sidebar.success("Tahmin yapmak için 'Prediction' sayfasına geçin.")


# Çalışan dizini yazdır
import os
# Mevcut çalışma dizinini kontrol et
current_dir = os.getcwd()
dataset_path = os.path.join(current_dir, "datasets", "test.csv")

# CSV dosyasını oku
df_te = pd.read_csv(dataset_path)


# Streamlit layout
st.title("---")
pie_fig = px.pie(df_te, names="Material", title="Distribution of Bag Materials")


# Pie chart below the two plots
st.plotly_chart(pie_fig)



