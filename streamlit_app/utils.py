import joblib

def load_model():
    """Kaydedilmiş makine öğrenmesi modelini yükler."""
    return joblib.load("model.pkl")
