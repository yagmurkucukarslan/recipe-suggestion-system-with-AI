# 🧠 Akıllı Tarif Önerici

**Akıllı Tarif Önerici**, kullanıcıların elinde bulunan malzemelere göre en uygun yemek tariflerini öneren bir yapay zeka destekli uygulamadır. Proje, makine öğrenmesi ve doğal dil işleme (NLP) teknikleri kullanarak malzemeleri analiz eder ve en olası yemek kategorilerini tahmin eder.

## 📌 Özellikler

- 📝 Geniş yemek tarifi veri seti (başlık, kategori, malzemeler)
- 🧠 TF-IDF tabanlı malzeme vektörleştirme
- 🔍 Scikit-learn ile sınıflandırma modeli 
- 📊 Model doğruluk ve performans analizi (confusion matrix, accuracy, precision)
- 💡 Girdi malzemelere göre kategori tahmini ve tarif önerisi

## 🧰 Kullanılan Teknolojiler

- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- NLTK (veya spaCy)
- Matplotlib / Seaborn (opsiyonel görselleştirme için)
- Jupyter Notebook (geliştirme süreci için)

## 📁 Dataset

Proje, Türkçe yemek tariflerinden oluşan özel bir veri setini kullanmaktadır. Veri seti şu sütunları içermektedir:

- `title`: Tarifin adı
- `category`: Tarifin ait olduğu kategori (örneğin, Çorba, Ana Yemek, Tatlı)
- `ingredients`: Tarif için gereken malzemeler (metin olarak)

## 🚀 Kurulum ve Kullanım

Gerekli kütüphaneleri yükleyin:
```
pip install -r requirements.txt
```
Projeyi çalıştırmak için `main.py` dosyasını çalıştırın
