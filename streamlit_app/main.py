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

# df_te = pd.read_csv("../datasets/test.csv")


# Create the visualizations
scatter_fig = px.scatter(df_te, x="Compartments", y="Weight Capacity (kg)", color="Brand",
                         title="Weight Capacity vs Compartments by Brand")
bar_fig = px.bar(df_te, x="Brand", y="Weight Capacity (kg)", title="Weight Capacity by Brand")
pie_fig = px.pie(df_te, names="Material", title="Distribution of Bag Materials")

# Streamlit layout
st.title("Çanta özellik Görselleştirmeleri")

# Create two columns for layout
col1, col2 = st.columns(2)

# Place the scatter plot in the first column
with col1:
    st.plotly_chart(scatter_fig)

# Place the bar plot in the second column
with col2:
    st.plotly_chart(bar_fig)

# Pie chart below the two plots
st.plotly_chart(pie_fig)



