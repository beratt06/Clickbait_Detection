import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

RSS_SOURCES = [
    #gündem haberleri
    "https://www.cnnturk.com/feed/rss/all/news",
    "https://www.hurriyet.com.tr/rss/anasayfa",
    "https://www.haber7.com/rss/sondakika.xml",
    "https://www.ensonhaber.com/rss/ensonhaber.xml",
    "https://www.milliyet.com.tr/rss/rssnew/sondakika.xml",
    "https://www.sabah.com.tr/rss/sondakika.xml",

    #teknoloji haberleri
    "https://www.chip.com.tr/rss/rss_teknoloji.xml",
    "https://www.webtekno.com/rss.xml",
    "https://www.donanimhaber.com/rss/tum/",

    # spor haberleri
    "https://www.fotomac.com.tr/rss/anasayfa.xml",
    "https://www.fanatik.com.tr/rss/haberler",
    "https://aspor.com.tr/rss/anasayfa.xml",

    # MAGAZİN haberleri
    "https://www.hurriyet.com.tr/rss/magazin",
    "https://www.milliyet.com.tr/rss/rssnew/magazin.xml",

    # EKONOMİ haberleri
    "https://www.bloomberght.com/rss",
    "https://www.bigpara.com/rss/"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}


def auto_label(baslik, ozet):
    score = 0
    baslik_lower = baslik.lower()

    # Yanıltıcı başlıklarımızı belirledik.
    keywords = [
        "şok", "belli oldu", "flaş", "nefes kesti", "inanılmaz", "şaşırttı",
        "görenler", "işte o", "duyurdu", "dikkat", "uyarı", "müjde", "resmen",
        "korkutan", "gerçeği", "sır", "skandal", "ağızları açık bıraktı",
        "neden", "kim", "nasıl", "ne zaman", "bomba", "patladı", "yok artık",
        "o isim", "transfer", "ayrılık", "deprem", "kahreden", "yıkan haber"
    ]

    for word in keywords:
        if word in baslik_lower:
            score += 1

    if "!" in baslik or "?" in baslik or "..." in baslik:
        score += 1

    if baslik.isupper():
        score += 1


    if len(baslik.split()) < 5 and score > 0:
        score += 2

    # Soru-Cevap Uyuşmazlığı
    if "?" in baslik and len(str(ozet).split()) < 20:
        score += 1

    return 1 if score >= 1 else 0

def mega_veri_topla():
    tum_veriler = []
    total_sources = len(RSS_SOURCES)

    print(f" BAŞLATILIYOR...({total_sources} Kaynak Taranacak)")
    print("-" * 50)

    for i, url in enumerate(RSS_SOURCES):
        print(f"[{i+1}/{total_sources}] Bağlanılıyor: {url}")

        try:
            response = requests.get(url, headers=HEADERS, timeout=8)

            if response.status_code == 200:
                #XML PARSER
                try:
                    soup = BeautifulSoup(response.content, "xml")
                except:
                    soup = BeautifulSoup(response.content, "html.parser")

                items = soup.find_all("item")
                kaynak_sayisi = 0

                for item in items:
                    try:
                        # Verileri Çek
                        baslik_tag = item.find("title")
                        baslik = baslik_tag.get_text(strip=True) if baslik_tag else None

                        # Açıklama temizliği
                        description = item.find("description")
                        if description:
                            raw_desc = description.get_text(strip=True)
                            ozet = BeautifulSoup(raw_desc, "html.parser").get_text(strip=True)
                        else:
                            ozet = ""

                        #linklerimizi getiriyoruz
                        link_tag = item.find("link")
                        link = link_tag.get_text(strip=True) if link_tag else ""


                        if baslik and len(baslik) > 5:
                            etiket = auto_label(baslik, ozet)

                            if not any(d['baslik'] == baslik for d in tum_veriler):
                                tum_veriler.append({
                                    "baslik": baslik,
                                    "icerik": ozet,
                                    "etiket": etiket,
                                    "kaynak_url": link
                                })
                                kaynak_sayisi += 1
                    except:
                        continue

                print(f"{kaynak_sayisi} haber alındı....")
            else:
                print(f" Erişim Hatası ({response.status_code})")

        except Exception as e:
            print(f"Bağlantı Zaman Aşımı veya Hata")
            continue

    #VERİYİ KAYDET
    print("-" * 50)
    if tum_veriler:
        df = pd.DataFrame(tum_veriler)

        dosya_adi = "clickbait_veriseti.csv"
        df.to_csv(dosya_adi, index=False, encoding="utf-8-sig")

        print("Veri seti hazır.")
        print(f"Toplam Satır: {len(df)}")
        print(f"Clickbait Oranı: %{df['etiket'].mean() * 100:.1f}")
        return True
    else:
        print("Hiçbir veri çekilemedi")
        return False

if __name__ == "__main__":
    basari = mega_veri_topla()

    if basari:
        try:
            data = pd.read_csv("clickbait_veriseti.csv")
        except:
            pass

#nan değerlerden kurtuluyoruz
data = data.dropna(subset=["icerik"])
print(data.isnull().sum())
#-------------------------------------------------


import numpy as np
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
# ---------------------------------------------------------

#Metin Ön İşleme adımı

nltk.download('stopwords')
turkish_stopwords = set(stopwords.words('turkish'))

def clean_text(text):
    text = str(text).lower()

    #Ünlem ve soru işareti gibi ifadeler clickbait olasılığını arttırdığı için silmiyoruz.
    text = re.sub(r'([!?])', r' \1 ', text)
    text = re.sub(r'[^a-zA-ZğüşıöçĞÜŞİÖÇ0-9!?\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    #["bir","ve"] gibi stopwordsleri kaldırıyoruz.
    tokens = text.split()
    tokens = [t for t in tokens if t not in turkish_stopwords or t in ['!', '?']]
    return " ".join(tokens)


#Başlık kısmı önemli olduğundan iki kere modele sokuyoruz.
data['combined_input'] = data['baslik'] + " " + data['baslik'] + " " + data['icerik']
data['clean_text'] = data['combined_input'].apply(clean_text)

# ---------------------------------------------------------

#Tokenization

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_DIM = 100

# Filtreden ! ve ? işaretlerini kaldırdık
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='"#$%&()*+,-./:;<=>@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['clean_text'].values)
print(data['clean_text'].values[0:2])

#padding ve sequences
X = tokenizer.texts_to_sequences(data['clean_text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = data['etiket'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=Y)

#Dengesiz veri için az olan veriye yüksek ağırlık veriyor.
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(Y_train),
    y=Y_train
)
class_weights_dict = dict(enumerate(weights))
print(f"Sınıf Ağırlıkları: {class_weights_dict}")

# ---------------------------------------------------------

#MODEL EĞİTİMİ (Bİ-LSTM)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(64, dropout=0.35, recurrent_dropout=0.3)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


print("Model Eğitiliyor")
history = model.fit(X_train, Y_train,
                    epochs=15,
                    batch_size=64,
                    validation_split=0.1,
                    class_weight=class_weights_dict,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

model.save("clickbait_model_v1.h5")
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("\n Model ve Tokenizer başarıyla kaydedildi!")
#---------------------------------------------------------

#Tahmin İçin Hazırlık
def predict_final(baslik, icerik):
    #eğitim formatına getirildi.
    raw_text = baslik + " " + baslik + " " + icerik

    cleaned = clean_text(raw_text)

    # sayısal vektöre çeviriyoruz ve padding uyguluyoruz.
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

    pred = model.predict(padded)[0][0]

    print("-" * 40)
    print(f"HABER: {baslik}")
    print(f"İÇERİK : {icerik}")
    print(f"SKOR : {pred:.2f}")


    threshold = 0.40

    if pred > threshold:
        print("SONUÇ: CLICKBAIT")
    else:
        print("SONUÇ: NORMAL HABER")


print("\n--- TEST BAŞLIYOR ---")

#Örnek Cümleler
predict_final(
    "Emekliye müjde! Maaşlar belli oldu...",
    "Çalışma bakanlığı çalışmalarını sürdürüyor, henüz net bir rakam yok."
)

predict_final(
    "Fenerbahçe derbiyi 3-1 kazandı",
    "Kadıköy'de oynanan maçta sarı lacivertliler rakibini 3 golle geçti."
)

predict_final(
    "şok şok! ünlü oyuncu engin altan bardan çıkarken görüntülendi!",
    "Engin Altan Düzyatan arkadaşlarıyla yemek yediği mekandan ayrılırken kameralara yansıdı."
)

predict_final(
    "Bunu yapan yandı! Cezası 5 bin lira...",
    "Apartman gürültü yönetmeliğine uymayanlara para cezası kesilecek."
)