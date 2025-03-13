import streamlit as st
import pandas as pd
from utils import load_model  # Modeli yükleyen yardımcı fonksiyon

st.title("🎒 Çanta Fiyat Tahmini Yapalım")
st.sidebar.header("📌 Çanta özelliklerini Giriniz")


# Kullanıcıdan giriş alma
def user_input():
    brand = st.sidebar.selectbox("Marka", ['Adidas', 'Under Armour', 'Nike', 'Puma', 'Jansport'])
    material = st.sidebar.selectbox("Malzeme", ['Polyester', 'Leather', 'Nylon', 'Canvas'])
    size = st.sidebar.selectbox("Boyut", ['Medium', 'Large', 'Small'])
    compartments = st.sidebar.slider("Bölme Sayısı", 1, 10, 5)
    laptop_compartment = st.sidebar.selectbox("Dizüstü Bilgisayar Bölmesi", ['Yes', 'No'])
    waterproof = st.sidebar.selectbox("Su Geçirmez", ['Yes', 'No'])
    style = st.sidebar.selectbox("Çanta Stili", ['Messenger', 'Tote', 'Backpack'])
    color = st.sidebar.selectbox("Renk", ['Pink', 'Gray', 'Blue', 'Red', 'Black', 'Green'])
    weight_capacity = st.sidebar.slider("Ağırlık Kapasitesi (kg)", 5, 30, 15)

    # Kullanıcının girdiği verileri DataFrame olarak döndür
    data = {
        "Brand": brand,
        "Material": material,
        "Size": size,
        "Compartments": compartments,
        "Laptop Compartment": laptop_compartment,
        "Waterproof": waterproof,
        "Style": style,
        "Color": color,
        "Weight Capacity (kg)": weight_capacity
    }

    return pd.DataFrame([data])


# Kullanıcıdan alınan veriyi sakla
input_data = user_input()

# Modeli yükle
model = load_model()

# 📌 Modelin eğitim sırasında kullandığı sütunları al
expected_columns = model.get_booster().feature_names

def özellik_mühendislği(df_tr):
    def grab_col_names(dataframe, cat_th=10, car_th=20):
        """
          Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

          Parameters
          ----------
          dataframe: dataframe
              değişken isimleri alınmak istenen dataframe'dir.
          cat_th: int, float
              numerik fakat kategorik olan değişkenler için sınıf eşik değeri
          car_th: int, float
              kategorik fakat kardinal değişkenler için sınıf eşik değeri

          Returns
          -------
          cat_cols: list
              Kategorik değişken listesi
          num_cols: list
              Numerik değişken listesi
          cat_but_car: list
              Kategorik görünümlü kardinal değişken listesi

          Notes
          ------
          cat_cols + num_cols + cat_but_car = toplam değişken sayısı
          num_but_cat cat_cols'un içerisinde.

          """
        cat_cols = [col for col in dataframe
                    if str(dataframe[col].dtypes) in ["object", "category"]]
        num_but_cat = [col for col in dataframe
                       if dataframe[col].nunique() < cat_th and
                       dataframe[col].dtypes in ["int", "float"]]
        cat_but_car = [col for col in dataframe if
                       dataframe[col].nunique() > car_th and
                       str(dataframe[col].dtypes) in ["category", "object"]]

        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        num_cols = [col for col in dataframe if dataframe[col].dtypes in ["float", "int"]]
        num_cols = [col for col in num_cols if col not in cat_cols]

        print(f"Observation: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f"cat_cols: {len(cat_cols)}")
        print(f"num_cols: {len(num_cols)}")
        print(f"cat_but_car: {len(cat_but_car)}")
        print(f"num_but_cat: {len(num_but_cat)}")

        return cat_cols, num_cols, cat_but_car

    cat_cols_tr, num_cols_tr, cat_but_car_tr = grab_col_names(df_tr)

    # eksik değerleri median ile doldurdum tek sayısal değişkenim vardı
    df_tr["Weight Capacity (kg)"] = df_tr["Weight Capacity (kg)"].fillna(df_tr["Weight Capacity (kg)"].median())

    # Cep sayısının fazla olması
    df_tr["NEW_Many_Compartments"] = df_tr["Compartments"].apply(lambda x: 1 if x > 7 else 0)

    # Laptop bölmesi varsa rengi siyahi koyu renk mi
    df_tr.loc[(df_tr["Laptop Compartment"] == "Yes") &
              ((df_tr["Color"] == "Black") | (df_tr["Color"] == "Gray")), "NEW_Businness_Backpack"] = 1
    df_tr["NEW_Businness_Backpack"].fillna(0, inplace=True)

    num_cols_tr = [col for col in num_cols_tr if "Price" != col]

    # Material canvas olan Compartment olan çantalar
    # daha fazla cep ortalama dilimine sahip
    #  Canvas olan çantaların Compartment sayısındaki farkı öğrenebilir, diğer çantalar için sıfır olacaktır.
    df_tr["NEW_Canvas_Compartment_Interaction"] = (df_tr["Material"] == "Canvas").astype(int) * df_tr["Compartments"]

    # çanta su geçirmez ise daha mı ağır Waterproof yes olan çantalar
    df_tr["NEW_Waterproof_Binary"] = df_tr["Waterproof"].map({"Yes": 1, "No": 0})
    df_tr.drop(columns=["Waterproof"], inplace=True)

    # Bu değişken, yalnızca su geçirmez olanların ağırlığını korur, diğerlerini sıfırlar.
    df_tr["NEW_Waterproof_Weight"] = df_tr["Weight Capacity (kg)"] * df_tr["NEW_Waterproof_Binary"]

    # Oransal Fark Hesaplama Su geçirmez ürünlerin diğerlerine kıyasla farkını gösterdiği için özellikle regresyon modellerinde etkili olabilir.
    mean_non_waterproof = df_tr[df_tr["NEW_Waterproof_Binary"] == 0]["Weight Capacity (kg)"].mean()
    df_tr["NEW_Weight_Pct_Diff"] = (df_tr["Weight Capacity (kg)"] - mean_non_waterproof) / mean_non_waterproof

    # Compartments değişkenini kategorikleştiriyorum
    df_tr.loc[pd.to_numeric(df_tr["Compartments"].between(1,3)),"NEW_Compartments"] = "Low"
    df_tr.loc[df_tr["Compartments"].between(4,7),"NEW_Compartments"] = "Average"
    df_tr.loc[df_tr["Compartments"].between(8,10),"NEW_Compartments"] = "High"

    return df_tr

input_data = özellik_mühendislği(input_data)

# 📌 One-Hot Encoding (OHE) Uygula
input_data = pd.get_dummies(input_data)


# 📌 Modelin eğitiminde kullanılan sütunlarla uyumlu hale getir
for col in expected_columns:
    if col not in input_data:
        input_data[col] = 0  # Eksik olan sütunları 0 ile doldur

# 📌 Fazla sütunları kaldır, sadece modelin beklediği sütunları kullan
input_data = input_data[expected_columns]

# Model ile tahmin yap
prediction = model.predict(input_data)

# Tahmin sonucunu göster
st.subheader("📊 Tahmin Sonucu")
st.write(f"Çanta Fiyat Tahmini: **{prediction[0]:.2f}** $")
