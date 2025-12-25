# 📄 AI Makale Tespit Sistemi

Bu proje, akademik metinlerin **insan tarafından mı yoksa yapay zeka (AI) tarafından mı üretildiğini**
tespit etmek amacıyla geliştirilmiş bir **makine öğrenmesi tabanlı web uygulamasıdır**.

Proje kapsamında metinlerin yalnızca kelime içeriği değil, aynı zamanda **yazım stili (style features)** de
dikkate alınarak daha gerçekçi ve genellenebilir bir tespit sistemi oluşturulmuştur.

---

## 🎯 Projenin Amacı

- Akademik metinlerin **AI / Human** olarak sınıflandırılması  
- İnsan benzeri yazılmış AI metinlerinin tespit edilmesi  
- Farklı makine öğrenmesi algoritmalarının karşılaştırılması  
- White-box test yaklaşımı ile sistemin iç işleyişinin doğrulanması  

---

## 🧠 Kullanılan Modeller

Projede üç farklı makine öğrenmesi algoritması kullanılmıştır:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**

Her model:
- TF-IDF vektörleri  
- Stil özellikleri (sentence length, punctuation ratio, vb.)  

kullanılarak eğitilmiştir.

---

## 🧩 Özellik Mühendisliği (Feature Engineering)

### 📌 TF-IDF
- Kelime ve kelime grubu frekansları
- `ngram_range=(1,2)`

### 📌 Style Features
Metnin yazım stilini temsil eden istatistiksel özellikler:
- Ortalama cümle uzunluğu
- Cümle uzunluğu standart sapması
- Ortalama kelime uzunluğu
- Noktalama işareti oranı
- Benzersiz kelime oranı
- Kelime tekrar oranı

Bu yaklaşım sayesinde **insan benzeri AI metinlerinin** daha iyi ayırt edilmesi hedeflenmiştir.

---

## 🖥️ Web Arayüzü (UI)

Kullanıcı arayüzü üzerinden:
- Metin girilebilir
- Aynı anda **3 modelin sonucu** görüntülenir
- Her model için **AI / Human yüzdeleri** gösterilir
- Boş girişlerde kullanıcı uyarılır

Web uygulaması **Flask** kullanılarak geliştirilmiştir.

---

## 🧪 Test Süreci

Projede **white-box test yaklaşımı** benimsenmiştir.

### Yapılan Testler:
- Boş metin giriş kontrolü
- Style feature fonksiyon testi
- Model çıktılarının olasılık tutarlılığı testi
- Tüm modellerin birlikte çalışması
- Model dosyalarının yüklenmesi
- Confusion Matrix (hata analizi)

Tüm testler **STD (Software Test Documentation)** formatına uygun olarak dokümante edilmiştir.

---

## 📊 Performans Analizi

Model performansları:
- Accuracy
- Precision / Recall / F1-score
- Confusion Matrix

kullanılarak değerlendirilmiştir.

Confusion matrix’ler:
- False Positive (Human → AI)
- False Negative (AI → Human)

hata türlerini analiz etmek için kullanılmıştır.

---

## 📁 Proje Yapısı

MakaleTespitProje/
│
├── app.py
├── DatasetTemizleme.py
├── DataHazirlamaBirlestirme.py
├── style_features.py
├── paths.py
│
├── Modeller/
│ ├── model_logistic_style_tuned.pkl
│ ├── model_svm_style_tuned.pkl
│ ├── model_rf_style_tuned.pkl
│ └── tfidf_style.pkl
│
├── Veriler/
│ └── islenmis/
│ └── dataset_combined.csv
│
├── WhiteBoxTestleri/
│
├── Outputs/
│ └── confusion_matrix_*.png
│
├── templates/
├── static/
└── README.md


---

## ⚠️ Model Sınırlamaları

- Gelişmiş dil modelleri tarafından üretilen **insan benzeri AI metinleri**, stilistik olarak insan yazımına çok yakın olabilir.
- Bu nedenle %100 doğruluk hedeflenmemiştir.
- Model, **AI yazım stilini tespit etmeye** odaklanmaktadır.

Bu durum projenin zayıflığı değil, problemin doğasından kaynaklanan bir sınırlılıktır.

---

## 🚀 Çalıştırma

```bash
pip install -r requirements.txt
python app.py

