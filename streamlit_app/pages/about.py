import streamlit as st

import os


st.write("Çalışma dizini:", os.getcwd())
st.write("Mevcut dosyalar:", os.listdir(os.getcwd()))

st.title("ℹ️ Hakkında")
st.write("""
Bu proje, Streamlit kullanarak bir makine öğrenmesi modeliyle tahmin yapmayı sağlamaktadır.
- Model, önceden eğitilmiş bir **XGBRegressor** modelidir.
- Basit bir arayüz ile modelimi canlıya taşıdığım bir projedir.
- Proje çanta veri seti özelliklerini barındırır. 
- Kullanınıcın girdileriyle veri içerisinde yakalanan patternler doğrultusunda çanta fiyat tahmini gerçekleşir.
- Kullanıcı, 'Prediction' sayfasından giriş değerleri girerek tahmin alabilir.


📌 **Geliştirici:** Emre Yıldırım
""")