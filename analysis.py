import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
import  seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import streamlit as st


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option("display.max_rows", 300)

"""
id → Ürün kimlik numarası
Brand → Marka
Material → Malzeme
Size → Boyut
Compartments → Bölme sayısı
Laptop Compartment → Dizüstü bilgisayar bölmesi var mı? (Evet/Hayır)
Waterproof → Su geçirmez mi? (Evet/Hayır)
Style → Çanta stili/türü
Color → Renk
Weight → Ağırlık (kg)
Capacity (kg) → Taşıma kapasitesi (kg)
"""


def grab_col_names(dataframe,cat_th=10,car_th=20):
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
                if str(dataframe[col].dtypes) in ["object","category"]]
    num_but_cat = [col for col in dataframe
                    if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes in ["int" ,"float"]]
    cat_but_car = [col for col in dataframe if
                   dataframe[col].nunique() > car_th and
                   str(dataframe[col].dtypes) in ["category","object"]]

    cat_cols = cat_cols+num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe if dataframe[col].dtypes in ["float","int"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols,num_cols,cat_but_car

def veri_seti():
    df_tr_extra = pd.read_csv("datasets/training_extra.csv")
    df_tr = pd.read_csv("datasets/train.csv")
    df_te = pd.read_csv("datasets/test.csv")

    # yaptığım submissionlar sonucu extra veri setini eklemeyi yararlı buldum
    df_tr = pd.concat([df_tr, df_tr_extra])

    return df_tr_extra,df_tr,df_te

df_tr_extra,df_tr,df_te = veri_seti()

def özellik_mühendislği(df_tr,df_te):

    cat_cols_te,num_cols_te,cat_but_car_te =grab_col_names(df_te)
    cat_cols_tr,num_cols_tr,cat_but_car_tr =grab_col_names(df_tr)

    # eksik değerleri median ile doldurdum tek sayısal değişkenim vardı
    df_tr["Weight Capacity (kg)"] = df_tr["Weight Capacity (kg)"].fillna(df_tr["Weight Capacity (kg)"].median())

    missing_percentage = df_te.isnull().mean() * 100
    print("\nEksik Veri Yüzdesi:\n", missing_percentage)

    df_te["Weight Capacity (kg)"] = df_te["Weight Capacity (kg)"].fillna(df_te["Weight Capacity (kg)"].median())


    # feature engineering


    # Cep sayısının fazla olması
    df_tr["NEW_Many_Compartments"] = df_tr["Compartments"].apply(lambda x: 1 if x > 7 else 0)
    df_te["NEW_Many_Compartments"] = df_te["Compartments"].apply(lambda x: 1 if x > 7 else 0)


    # Laptop bölmesi varsa rengi siyahi koyu renk mi
    df_tr.loc[ (df_tr["Laptop Compartment"] == "Yes") &
               ( (df_tr["Color"] == "Black") | (df_tr["Color"] == "Gray") ) , "NEW_Businness_Backpack"] = 1
    df_tr["NEW_Businness_Backpack"].fillna(0,inplace=True)

    df_te.loc[ (df_te["Laptop Compartment"] == "Yes") &
               ( (df_te["Color"] == "Black") | (df_te["Color"] == "Gray") ) , "NEW_Businness_Backpack"] = 1
    df_te["NEW_Businness_Backpack"].fillna(0,inplace=True)


    num_cols_tr = [col for col in num_cols_tr if "Price" != col]


    # Material canvas olan Compartment olan çantalar
    # daha fazla cep ortalama dilimine sahip
    #  Canvas olan çantaların Compartment sayısındaki farkı öğrenebilir, diğer çantalar için sıfır olacaktır.
    df_tr["NEW_Canvas_Compartment_Interaction"] = (df_tr["Material"] == "Canvas").astype(int) * df_tr["Compartments"]

    df_te["NEW_Canvas_Compartment_Interaction"] = (df_te["Material"] == "Canvas").astype(int) * df_te["Compartments"]


    # çanta su geçirmez ise daha mı ağır Waterproof yes olan çantalar
    df_tr["NEW_Waterproof_Binary"] = df_tr["Waterproof"].map({"Yes": 1, "No": 0})
    df_tr.drop(columns=["Waterproof"], inplace=True)

    df_te["NEW_Waterproof_Binary"] = df_te["Waterproof"].map({"Yes": 1, "No": 0})
    df_te.drop(columns=["Waterproof"], inplace=True)

    # Bu değişken, yalnızca su geçirmez olanların ağırlığını korur, diğerlerini sıfırlar.
    df_tr["NEW_Waterproof_Weight"] = df_tr["Weight Capacity (kg)"] * df_tr["NEW_Waterproof_Binary"]

    df_te["NEW_Waterproof_Weight"] = df_te["Weight Capacity (kg)"] * df_te["NEW_Waterproof_Binary"]

    # Oransal Fark Hesaplama Su geçirmez ürünlerin diğerlerine kıyasla farkını gösterdiği için özellikle regresyon modellerinde etkili olabilir.
    mean_non_waterproof = df_tr[df_tr["NEW_Waterproof_Binary"] == 0]["Weight Capacity (kg)"].mean()
    df_tr["NEW_Weight_Pct_Diff"] = (df_tr["Weight Capacity (kg)"] - mean_non_waterproof) / mean_non_waterproof

    mean_non_waterproof_te = df_te[df_te["NEW_Waterproof_Binary"] == 0]["Weight Capacity (kg)"].mean()
    df_te["NEW_Weight_Pct_Diff"] = (df_te["Weight Capacity (kg)"] - mean_non_waterproof_te) / mean_non_waterproof_te


    # Compartments değişkenini kategorikleştiriyorum
    """
    plt.hist(df_tr["Compartments"])
    plt.show()
    
    df_tr["Compartments"].value_counts()
    """


    df_tr["NEW_Compartments"] = pd.qcut(df_tr["Compartments"], q=3,
                                        labels=["Low", "Average","High"])

    df_te["NEW_Compartments"] = pd.qcut(df_te["Compartments"], q=3,
                                        labels=["Low","Average", "High"])

    return df_tr,df_te

df_tr,df_te = özellik_mühendislği(df_tr,df_te)

df_tr.shape
df_te.shape

dff_tr = pd.get_dummies(df_tr, drop_first=True)
dff_te = pd.get_dummies(df_te, drop_first=True)

# Eğitim veri setini yükle
X_train = dff_tr.drop(columns=['Price'])  # Eğitim verisi, 'target' dışındaki özellikler
y_train = dff_tr['Price']  # Eğitim verisinin hedef değeri

# Modeli oluştur ve eğit
# XGBRegressor modelini oluştur ve eğit
model = XGBRegressor(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Eğitim verisi üzerinde tahmin yap
y_train_pred = model.predict(X_train)

# MSE hesapla
mse_train = mean_squared_error(y_train, y_train_pred)

# RMSE hesapla
rmse_train = np.sqrt(mse_train)

print(f"Eğitim Verisi RMSE: {rmse_train}")
"""
Eğitim Verisi RMSE: 38.81968357911567
"""


joblib.dump(model, "streamlit_app/model.pkl")


# Önceden kaydedilmiş modeli yükle
#model = joblib.load("streamlit_app/model.pkl")


# Özellik önem dereceleri için grafik oluşturalım
feature_importances = model.feature_importances_
features = X_train.columns

# Özellik önemlerini bir DataFrame'e koy
feat_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Hata kontrolü: Boş olup olmadığını kontrol et
if feat_importance_df.empty:
    print("Hata: Özellik önem dereceleri boş! Model eğitildi mi?")
else:
    # Veriyi sıralayarak ilk 15 özelliği al
    feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False)

    # Seaborn ile grafik çizdir
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_importance_df, palette='Blues_r')
    plt.xlabel("Önem Derecesi")
    plt.ylabel("Özellikler")
    plt.title("XGBoost Özellik Önem Dereceleri")
    plt.show()



def tahmin_yap(dff_te,X_test):
    # Test verisi üzerinde tahmin yap
    X_test = dff_te  # Test verisindeki özellikler
    y_pred = model.predict(X_test)

    # Test verisindeki id'leri al
    ids = dff_te['id']  # Test veri setindeki 'id' kolonu

    # Sonuçları bir DataFrame'e dönüştür
    results = pd.DataFrame({
        'id': ids,  # Test verisindeki 'id' kolonu
        'Predicted_Price': y_pred  # Tahmin edilen fiyatlar
    })

    # Sonuçları dosyaya kaydet (örneğin CSV)
    results.to_csv('predicted_prices_20.csv', index=False)






