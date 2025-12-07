# 📌 CLICKBAIT TESPİT SİSTEMİ --- RNN MODELİ

Türkçe haber sitelerinden otomatik veri toplayarak **clickbait (tıklama
tuzağı)** başlıkları tespit eden bir **Makine Öğrenimi / NLP
projesidir**.\
Bu çalışma aynı zamanda **RNN ve LSTM modellerinin performans farkını
incelemek** amacıyla geliştirilmiştir.

## 🚀 Özellikler

-   🔄 **RSS kaynaklarından otomatik veri toplama**
-   🏷️ **Başlık + açıklamadan otomatik clickbait etiketleme**
-   📝 **Türkçe NLP işlemleri**
    -   Noktalama analizi (!) (?) (...)
    -   Stopwords temizliği
    -   Kök bulma (TurkishStemmer)
-   🧠 **RNN tabanlı derin öğrenme modeli**
    -   SimpleRNN
    -   Embedding Layer
    -   Dropout / Recurrent Dropout
-   ⚖️ **Class Weight ile dengesiz veri çözümü**
-   💾 **Eğitim sonrası model kayıtları**
    -   clickbait_model_v1.h5
    -   tokenizer.pickle
-   🔍 **Gerçek cümlelerle canlı test fonksiyonu**

## 📂 Proje Dosya Yapısı

    Clickbait_detection_RNN.py
    clickbait_veriseti.csv
    clickbait_model_v1.h5
    tokenizer.pickle

## 📰 Veri Toplama Süreci

-   Gündem
-   Teknoloji
-   Spor
-   Magazin
-   Ekonomi

Etiketleme kuralları:

-   Clickbait kelimeleri
-   Kısa başlık
-   Başlık--özet uyumsuzluğu
-   Manipülatif işaretler (!, ?, büyük harf)

  Etiket   Anlam
  -------- -----------
  1        Clickbait
  0        Normal

## 🧹 Metin Ön İşleme

-   Küçük harfe çevirme\
-   Karakter temizliği\
-   Stopwords temizliği\
-   Kök bulma\
-   ! ve ? korunur\
-   Başlık 2 kez modele verilir

## 🧠 Model Mimarisi

    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.3))
    model.add(SimpleRNN(64, dropout=0.2, recurrent_dropout=0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

## 📊 Eğitimi Başlatma

``` bash
python Clickbait_detection_RNN.py
```

## 🔍 Tahmin Fonksiyonu

    predict_final(baslik, icerik)

Örnek:

    predict_final(
        "Emekliye müjde! Maaşlar belli oldu...",
        "Çalışma bakanlığı henüz kesin bir açıklama yapmadı."
    )
