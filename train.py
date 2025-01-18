import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Veri setini yükleyin
data = pd.read_csv("metinlikategori.csv")

# Yemek adını sayısal değere dönüştürme (Label Encoding)
le_title = LabelEncoder()
data['Title'] = le_title.fit_transform(data['Title'])  # Yemek ismini sayısal hale getirin

# Yemek ismini hedef değişken olarak alıyoruz
X = data.drop(columns=["Category", "Title"])  # Malzemelerden oluşan özellikler matrisi
y_title = data["Title"]  # Yemek adı bilgisi

# Veriyi eğitim ve test kümelerine ayırın
X_train, X_test, y_train_title, y_test_title = train_test_split(X, y_title, test_size=0.2, random_state=42)

# Random Forest modelini oluştur
model_title = RandomForestClassifier(n_estimators=50, random_state=42)

# Modeli eğit
model_title.fit(X_train, y_train_title)

# Test verisi üzerinde performansı değerlendirin
y_pred_title = model_title.predict(X_test)

# Modelin doğruluğunu yazdırın
accuracy_title = accuracy_score(y_test_title, y_pred_title)
print(f"Yemek Adı Modeli Accuracy: {accuracy_title * 100:.2f}%")

# Eğitim ve test verisinin sütun adlarını kontrol et
print("Eğitim verisinin sütunları:", X_train.columns)
print("Test verisinin sütunları:", X_test.columns)

# Modeli ve kategorileri (sayısal hale dönüştürülmüş) kaydedin
model_data = {
    "model_title": model_title,  # Yemek adı modelini kaydediyoruz
    "malzemeler": X.columns.tolist(),  # Malzemelerin isimlerini alıyoruz
    "titles": le_title.classes_.tolist()  # Yemek adlarının metin halini kaydediyoruz
}

# Modeli kaydedin
joblib.dump(model_data, "model.joblib")
print("Model başarıyla kaydedildi.")
