# HititFinLex — NLP Veri Seti Üretici

Toplanan ham korpustan (`../data_collection/data/processed/`) NLP aşamasının
eğiteceği/değerlendireceği **iki etiketli veri seti** üretir:

| Veri seti | Şartname | Görev | Birim |
|---|---|---|---|
| **Bilgi çıkarım** | 5.3 | Kâr payı oranı, vade, tutar, masraf, kampanya süresi, hedef kitle span'lerini bulma | Pasaj (~600 karakter) |
| **Kampanya türü** | 5.4 | Sekiz kampanya türünden birine sınıflandırma | Doküman |

Etiketler **kural tabanlı ön etiketleme + insan doğrulaması** ile üretilir.
Kurallar hiçbir zaman "kesin etiket" saymaz; her öneri bir güven skoru taşır
ve düşük güvenli olanlar doğrulama kuyruğuna düşer.

---

## 1. Hızlı Başlangıç

Bağımlılıklar veri toplama modülüyle ortaktır (`PyYAML` yeterli, ek kurulum
gerekmez). Aynı sanal ortamı kullanın:

```bash
cd VeriToplama/dataset_builder

# 1) Ön etiketle + doğrulama kuyruğunu üret
python build_dataset.py

# 2) data/dogrulama/*.csv dosyalarını Excel'de açın,
#    ONAY_E_H sütununu E (doğru) / H (yanlış) olarak doldurun.
#    Yanlışsa DUZELTILMIS_ETIKET / DUZELTILMIS_TUR sütununa doğrusunu yazın.

# 3) Onayları uygula ve final seti yeniden üret
python build_dataset.py --review-uygula

# Yalnızca insan onaylı + yüksek güvenli (gold) alt küme istenirse:
python build_dataset.py --review-uygula --sadece-dogrulanmis
```

Diğer seçenekler: `python build_dataset.py --help`

---

## 2. Klasör Yapısı

```
dataset_builder/
├── config/
│   ├── etiket_semasi.yaml   Etiket tanımları, öncelikler, bölme oranları
│   └── kurallar.yaml        Ön etiketleme desenleri (regex + bağlam)
├── labelers/
│   ├── span_labeler.py      Bilgi çıkarım ön etiketleyici (5.3)
│   └── type_labeler.py      Kampanya türü ön etiketleyici (5.4)
├── utils/
│   ├── normalize.py         Standart formata dönüştürme (5.6)
│   ├── segment.py           Belgeyi pasajlara bölme (ofsetler korunur)
│   └── splits.py            Sızıntısız, tabakalı train/val/test
├── data/
│   ├── ara/                 Ön etiketli ara çıktı (pasajlar, dokumanlar)
│   ├── dogrulama/           İnsan doğrulaması için CSV'ler
│   └── final/
│       ├── bilgi_cikarim/   train/val/test .jsonl + *_bio.jsonl
│       ├── siniflandirma/   train/val/test .jsonl
│       ├── etiketler_span.json
│       └── etiketler_sinif.json
├── build_dataset.py
└── README.md
```

---

## 3. Çıktı Biçimleri

### Bilgi çıkarım — `final/bilgi_cikarim/train.jsonl`

```json
{
  "id": "vakif_katilim-b7569bb0e097-p004",
  "belge_id": "vakif_katilim-b7569bb0e097",
  "banka": "vakif_katilim",
  "kaynak_url": "https://...",
  "metin": "15.000 TL'ye kadar vade farksız-masrafsız finansmanınızı...",
  "spanlar": [
    {"baslangic": 0, "bitis": 21, "etiket": "FINANSMAN_TUTARI",
     "metin": "15.000 TL'ye kadar",
     "normal_deger": {"deger": 15000.0, "birim": "TRY", "sinir": "ust"},
     "guven": 0.88, "kaynak": "kural"},
    {"baslangic": 22, "bitis": 34, "etiket": "MASRAF_DURUMU",
     "metin": "vade farksız", "normal_deger": {"metin": "vade farksız"},
     "guven": 0.95, "kaynak": "insan"}
  ],
  "dogrulandi": true
}
```

Aynı klasördeki `*_bio.jsonl` dosyaları token sınıflandırma eğitimi için
BIO dizisi taşır (`B-FINANSMAN_TUTARI`, `I-FINANSMAN_TUTARI`, `O`).
Tokenizasyon basit boşluk+noktalama tabanlıdır; kullanacağınız modelin
tokenizer'ı farklıysa `spanlar` alanındaki karakter ofsetlerinden yeniden
hizalayın — asıl kaynak odur.

### Kampanya türü — `final/siniflandirma/train.jsonl`

```json
{
  "id": "albaraka-3f2a1c...",
  "banka": "albaraka",
  "baslik": "Eğitim Harcamalarınıza Vade Farksız 6 Taksit Kampanyası",
  "metin": "...",
  "etiket": "Kart Kampanyası",
  "etiket_kod": "KART",
  "dogrulandi": false
}
```

### Etiket kümesi

Span etiketleri (13): `KAR_PAYI_ORANI`, `KAR_PAYLASIM_ORANI`,
`FINANSMAN_TUTARI`, `VADE_SURESI`, `TAKSIT_SAYISI`, `TAHSIS_UCRETI`,
`MASRAF_DURUMU`, `HARCAMA_ESIGI`, `ODUL_MIKTARI`, `INDIRIM_ORANI`,
`ALISVERIS_PUANI`, `KAMPANYA_SURESI`, `HEDEF_KITLE`

> `HARCAMA_ESIGI` ile `ODUL_MIKTARI` ayrımına dikkat: *"500 TL ve üzeri
> harcamalarına 50 TL Worldpuan"* cümlesinde 500 TL **eşik**, 50 TL
> **ödül**dür. Ayrı etiket tanımlanmadan önce kart kampanyalarındaki eşik
> tutarlarının tamamı ödül olarak etiketleniyordu.

Kampanya türleri (şartname 5.4 tablosundan): Finansman / İhtiyaç Finansmanı /
Konut Finansmanı / Taşıt Finansmanı / Kart / Alışveriş Puanı / Yeni Müşteri /
Yatırım Ürünü Kampanyası + `Diğer / Kampanya Değil`

---

## 4. Tasarım Kararları

### Neden kural tabanlı ön etiketleme?

143 belgeyi sıfırdan elle etiketlemek günler sürer ve tutarsız olur. Kurallar
işi "yazmak"tan "onaylamak"a indirir. Ayrıca kurallar atılabilir bir ara ürün
değildir: NLP aşamasında hibrit bir çıkarım hattının (kural + model) kural
ayağı olarak doğrudan kullanılır.

### Neden her desen bağlam zorunlu tutuyor?

`%25` tek başına indirim mi kâr payı oranı mı belli değildir. Desenlerin çoğu
çevresinde belirli bir kalıp (`baglam`) görmedikçe öneri üretmez. Tersine
`baglam_disi` yanlış pozitifleri keser: mevcut korpusta "kâr payı" geçen
metinlerin neredeyse tamamı **"kâr payı uygulanmaz"** biçimindeydi; bunları
oran sanmak veri setini baştan bozardı.

### Neden "kampanya mı?" ile "hangi tür?" ayrı kararlar?

Tek puan ölçeğinde yarıştırıldığında `Arzum'da %25 İndirim!` gibi gerçek kart
kampanyaları `Diğer`'e düşüyordu — "kart" kelimesi yalnızca gövdede geçtiği
için puanı eşiği aşmıyordu. Sayfanın kampanya olduğu URL'den veya toplama
aşamasındaki `kampanya_mi` bayrağından zaten biliniyorsa, tür için daha düşük
kanıt yeterlidir; kalan belirsizlik **güven skoruna** yansıtılır, etiketin
tümden kaybedilmesine değil.

### Neden pasaj düzeyinde span, doküman düzeyinde sınıf?

Belgelerin ortalaması ~2.450, en uzunu 19.000 karakter. Bu uzunluk tipik
transformer bağlam penceresini aşar ve insan doğrulamasını zorlaştırır.
Pasajlar bölünürken **karakter ofsetleri korunur**, böylece her span özgün
belgedeki yerine geri izlenebilir. Sınıflandırma ise belgenin bütününe ait
bir karardır, bölünmez.

### Sızıntı önleme

Bankalar aynı perakendeci kampanyasını neredeyse birebir aynı metinle
yayınlıyor ("X'te peşin fiyatına 6 taksit fırsatı"). Bu metinlerin biri
train'e biri test'e düşerse test skoru gerçek başarıyı değil ezberi ölçer.
Bu yüzden belgeler önce kelime imzasıyla kümelenir ve **küme bütün olarak**
tek bir bölüme atanır. Bir belgenin tüm pasajları da aynı bölümde kalır.
Üretim akışı her koşuda `sizinti_denetimi()` çalıştırır; bölümler arasında
ortak küme bulunursa hata olarak loglanır.

### Normalizasyon (şartname 5.6)

`%2,05`, `% 2.05`, `2.05 %` → `{"deger": 2.05, "birim": "yuzde"}`
`500 TL`, `500₺`, `500 Türk Lirası` → `{"deger": 500.0, "birim": "TRY"}`
`120 ay`, `10 yıl` → `{"deger": 120, "birim": "ay"}` (karşılaştırma için hep ay)

Türkçe sayı biçimine dikkat edilir: `50.000,75` içindeki nokta binlik
ayıraçtır, naif `float()` bunu 50.0 okur.

`…'ye varan` / `…'e kadar` ekleri `sinir: "ust"` olarak korunur — şartname
5.7'deki "En Uzun Vade Seçeneği", "En Yüksek Ödül Miktarı" karşılaştırmaları
bu bilgiye ihtiyaç duyar.

**Metin normalizasyonu bilinçli olarak hafiftir:** küçük harfe çevirme,
noktalama atma gibi işlemler yapılmaz. `%1,89` içindeki virgül ve yüzde
işareti bilginin kendisidir.

---

## 5. Doğrulama İş Akışı

`build_dataset.py` dört CSV üretir:

| Dosya | İçerik |
|---|---|
| `01_span_dogrulama.csv` | Güveni 0,90 altındaki tüm span önerileri — **tamamı gözden geçirilmeli** |
| `02_span_ornek_denetim.csv` | Yüksek güvenli span'lerden %20 örneklem — kuralların isabetini ölçer |
| `03_tur_dogrulama.csv` | Güveni 0,90 altındaki tür önerileri |
| `04_tur_ornek_denetim.csv` | Yüksek güvenli tür önerilerinden örneklem |

Her satırda `baglam` sütunu, yakalanan parçayı `⟪…⟫` içinde çevresiyle
birlikte gösterir; doğrulayıcının kaynağa gitmesi gerekmez.

Doldurma kuralı:
- `ONAY_E_H` = **E** → etiket doğru, `kaynak` alanı `insan` olur, güven 1.0
- `ONAY_E_H` = **H**, düzeltme boş → etiket veri setinden **çıkarılır**
- `DUZELTILMIS_ETIKET` doluysa → onay ne olursa olsun düzeltme uygulanır

Örneklem denetiminde hata oranı yüksek çıkarsa çözüm CSV'de tek tek düzeltmek
değil, `config/kurallar.yaml` içindeki deseni düzeltip yeniden üretmektir.

---

## 6. Mevcut Veri Seti (2026-08-24 v1.0 paketi)

Kaynak korpus: **3.351 belge / 10 banka / 1.043.573 kelime**
Canlı/arşiv ayrımı: **771 canlı + 2.580 arşiv belge**

| | train | val | test | toplam |
|---|---:|---:|---:|---:|
| Bilgi çıkarım (pasaj) | 5.015 | 1.177 | 1.046 | **7.238** |
| Sınıflandırma (belge) | 2.346 | 504 | 501 | **3.351** |

Toplam **10.501 span** vardır. Etiket dağılımı: `KAMPANYA_SURESI` 2.296,
`TAKSIT_SAYISI` 1.936, `ODUL_MIKTARI` 1.036, `HARCAMA_ESIGI` 971,
`MASRAF_DURUMU` 902, `VADE_SURESI` 877, `INDIRIM_ORANI` 773,
`ALISVERIS_PUANI` 500, `FINANSMAN_TUTARI` 437, `HEDEF_KITLE` 369,
`TAHSIS_UCRETI` 252, `KAR_PAYLASIM_ORANI` 90, `KAR_PAYI_ORANI` 62.

`MANIFEST.json` bu sayıları paketin kanonik envanteri olarak sabitler. Etiketler
**silver/kural tabanlı ön etiketlerdir**; henüz insan onaylı (gold) kayıt yoktur.

---

## 7. Bilinen Sınırlar

- **`KAR_PAYI_ORANI` en az örnekli etiket (62 span).** Bankalar güncel oranları
  kampanya metninde değil ayrı sayfalarda veya PDF fiyat listelerinde
  yayınlıyor. Toplama modülü bu sayfaları hedefleyecek şekilde güncellendi
  (kâr payı/ücret sayfaları filtreden çıkarıldı ve önceliklendirildi);
  korpustaki oran içeren kayıt sayısı %7,7'den %12,7'ye çıktı. Kalan boşluğun
  iki nedeni var ve ikisi de bu modülün dışında:
  - **TOM Bank oranlarını yalnızca PDF olarak yayınlıyor** (`krediler_kar_
    oranlari_*.pdf`); PDF ayrıştırma kapsam dışı, o bankanın oranları yok.
  - **Türkiye Finans'ın `Kar-Payi-Oranlari.aspx` sayfası toplandı ama
    içinde hiç yüzde değeri yok** — tablo istemci tarafında ayrı bir
    servisten yükleniyor olabilir.
- **Sınıf dengesizliği ciddi.** Kart Kampanyası 1.393, Konut Finansmanı
  Kampanyası 55. Eğitimde sınıf ağırlıklandırma veya örnekleme şart.
- **Ön etiketler istatistiksel olarak doğrulanmadı.** Güven skorları desen
  isabetine dair *tasarım varsayımıdır*, ölçülmüş kesinlik değildir. Gerçek
  kesinlik/duyarlılık ancak örneklem denetimi (`02_`, `04_` dosyaları)
  doldurulduktan sonra hesaplanabilir.
- **Kampanya türü ön etiketleri silver niteliktedir.** Anahtar kelimeye dayalı
  tür tahmini insan doğrulaması olmadan altın standart sayılmamalı; güven
  dağılımı yalnız inceleme önceliği belirlemek için kullanılmalıdır.
- **`Diğer / Kampanya Değil` sınıfı heterojen** (310 belge). Ürün tanıtım
  sayfaları, bilgilendirme metinleri ve türü belirlenemeyen kampanyalar aynı
  etikette. Doğrulama sırasında ayrıştırılması önerilir; `gerekce` alanı
  hangisinin hangi sebeple bu etikete düştüğünü söyler.
- **Etiketsiz pasajlar veri setinde yok.** Yalnızca en az bir span bulunan
  pasajlar yazılıyor. Token sınıflandırma eğitiminde negatif örnek gerekirse
  `data/ara/` üzerinden üretilmeli.
