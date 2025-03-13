import joblib

def load_model():
    """Kaydedilmiş makine öğrenmesi modelini yükler."""
    import os
    # Mevcut çalışma dizinini kontrol et
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "streamlit_app", "model.pkl")
    return joblib.load(model_path)
