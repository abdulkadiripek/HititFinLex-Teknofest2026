# HititFinLex Veri Seti — v1.0

**TEKNOFEST 2026 Yapay Zeka Dil Ajanları Yarışması · 2. Senaryo — Katılım Bankacılığı**

Türkiye'deki **10 katılım bankasının tamamının** kampanya ve ürün sayfalarından
derlenmiş Türkçe metin korpusu ve bu korpustan üretilmiş iki etiketli NLP veri seti.

Paket tarihi: **24 Ağustos 2026**

---

## 1. Tek bakışta

### Korpus (ham metin)

| | |
|---|---|
| Kayıt | **3.351** sayfa |
| Metin | 8.181.060 karakter · 1.043.573 kelime |
| Kurum | BDDK listesindeki 10 katılım bankasının tamamı |
| **İçerik tarih aralığı** | **1 Ocak 2016 – 21 Ağustos 2026** (10 yıl 8 ay) |
| Toplama tarihi | 14–24 Ağustos 2026 |
| Kaynak | 771 canlı site (%23) + 2.580 web arşivi (%77) |
| Benzersiz sayfa adresi | 2.467 |

### Veri seti (etiketli)

| Görev | Şartname | Birim | Train | Val | Test | Toplam |
|---|---|---|---:|---:|---:|---:|
| Bilgi çıkarım (span) | 5.3 | pasaj | 5.015 | 1.177 | 1.046 | **7.238** |
| Kampanya türü (sınıflandırma) | 5.4 | belge | 2.346 | 504 | 501 | **3.351** |

Toplam **10.501 span** etiketi, 13 span sınıfı ve 9 kampanya türü üzerinden.

> Pasajlar **birebir tekilleştirilmiştir**: aynı metin veri setinde bir kez bulunur.
> 49.202 kopya pasaj elendi. Gerekçesi §6'da.

---

## 2. Paket içeriği

```
HititFinLex_VeriSeti_v1.0_2026-08-24/
├── OKUBENI.md              ← bu dosya
├── MANIFEST.json           makine okunur özet: sayımlar, dosya listesi, boyutlar
├── LICENSE
│
├── korpus/                 HAM METİN
│   ├── korpus_tum.jsonl        3.351 kayıt — canlı + arşiv, tekilleştirilmiş (ANA DOSYA)
│   ├── korpus_tum.csv          aynı veri, CSV (utf-8-sig, Excel uyumlu)
│   ├── korpus_canli.jsonl        771 kayıt — yalnızca bugünkü siteler
│   └── korpus_arsiv.jsonl      2.670 kayıt — yalnızca web arşivi (tekilleştirme öncesi)
│
├── veri_seti/              ETİKETLİ VERİ SETİ
│   ├── bilgi_cikarim/
│   │   ├── train.jsonl / val.jsonl / test.jsonl        pasaj + karakter ofsetli span'lar
│   │   └── train_bio.jsonl / val_bio.jsonl / test_bio.jsonl   BIO token dizileri
│   ├── siniflandirma/
│   │   └── train.jsonl / val.jsonl / test.jsonl        belge + tek etiket
│   ├── etiketler_span.json     13 span sınıfı
│   └── etiketler_sinif.json     9 kampanya türü
│
├── semalar/                TANIMLAR
│   ├── etiket_semasi.yaml      etiket tanımları, eşikler, bölme oranları
│   ├── kurallar.yaml           kural tabanlı ön etiketleyicinin kural tabanı
│   └── bankalar.yaml           banka listesi, hedef adresler, toplama ayarları
│
├── ozet/                   ÜRETİM KAYITLARI
│   ├── veri_seti_ozeti.json    etiket dağılımları, güven dağılımı, bölme raporu
│   ├── toplama_ozeti_canli.json
│   └── toplama_ozeti_arsiv.json
│
└── dokuman/
    ├── VERI_KARTI.md               korpusun ayrıntılı veri kartı (önce bunu okuyun)
    ├── veri_toplama_README.md      toplama modülünün mimarisi
    └── veri_seti_README.md         veri seti üreticisinin mimarisi
```

---

## 3. Hızlı başlangıç

```python
import pandas as pd, json

# --- Ham korpus ---
korpus = pd.read_json("korpus/korpus_tum.jsonl", lines=True)

# --- Bilgi çıkarım seti (span) ---
train = [json.loads(s) for s in open("veri_seti/bilgi_cikarim/train.jsonl", encoding="utf-8")]
ornek = train[0]
ornek["metin"]      # pasaj metni
ornek["spanlar"]    # [{"etiket","baslangic","bitis","metin","guven",...}, ...]
# Ofsetler metin üzerinde doğrudan çalışır:
s = ornek["spanlar"][0]
assert ornek["metin"][s["baslangic"]:s["bitis"]] == s["metin"]

# --- Sınıflandırma seti ---
sinif = [json.loads(s) for s in open("veri_seti/siniflandirma/train.jsonl", encoding="utf-8")]
sinif[0]["metin"], sinif[0]["etiket"]
```

BIO biçimi (token sınıflandırma için) `*_bio.jsonl` dosyalarındadır. Tokenizasyon
basit boşluk + noktalama ayrımıdır; amaç referans hizalama sağlamaktır, kullanılacak
modelin kendi tokenizer'ı farklı olacaktır.

---

## 4. ⚠️ Bilinmesi zorunlu iki kural

### 4.1 `guncel_mi` — arşiv kayıtları

Korpusun %77'si web arşivinden gelir; yani **kaldırılmış, süresi dolmuş** kampanya
sayfalarıdır. Her kayıt `guncel_mi` alanı taşır:

| Değer | Anlamı |
|---|---|
| `true` | Toplama anında bankanın sitesinde yayındaydı (771 kayıt) |
| `false` | Web arşivinden alındı; `anlik_goruntu_tarihi` o tarihte yayında olduğunu gösterir (2.580 kayıt) |

- **Model eğitiminde ikisini de kullanın.** Süresi dolmuş bir kampanyanın metni de
  tamamen geçerli bir etiketleme örneğidir — kâr payı oranı, vade, tutar span'ları
  aynı biçimde geçer.
- **Kullanıcıya bilgi sunan katmanda (dashboard, chatbot) `guncel_mi=false`
  olanları eleyin.** 2019 tarihli bir kampanyayı yürürlükteymiş gibi göstermek
  hatalı bilgi verir.

```python
egitim = df                                 # her şey
guncel = df[df["guncel_mi"].fillna(True)]   # yalnızca yürürlükteki içerik
```

`guncel_mi` alanı hem korpusta hem de her iki final veri setinde bulunur.

### 4.2 Etiketler **silver**, gold değil

Etiketler **kural tabanlı ön etiketleyiciyle** üretilmiştir ve insan doğrulaması
beklemektedir. Her kayıt `dogrulandi` bayrağı taşır:

| Durum | Pasaj | Belge |
|---|---:|---:|
| Gold (insan onaylı) | 0 | 0 |
| Yüksek güvenli (otomatik kabul) | 4.002 | 44 |
| Doğrulama bekleyen | 3.236 | 3.307 |

Doğrulama kuyruğu (`01_span_dogrulama.csv` vb.) bu pakete dâhil edilmemiştir;
üretim komutu `veri_seti_README.md` içinde açıklanmıştır. İnsan doğrulaması
tamamlandığında `--review-uygula` ile gold sürüm üretilir.

---

## 5. Etiket şeması

### Bilgi çıkarım — 13 span sınıfı

| Etiket | Adet | Etiket | Adet |
|---|---:|---|---:|
| `KAMPANYA_SURESI` | 2.296 | `INDIRIM_ORANI` | 773 |
| `TAKSIT_SAYISI` | 1.936 | `ALISVERIS_PUANI` | 500 |
| `ODUL_MIKTARI` | 1.036 | `FINANSMAN_TUTARI` | 437 |
| `HARCAMA_ESIGI` | 971 | `HEDEF_KITLE` | 369 |
| `MASRAF_DURUMU` | 902 | `TAHSIS_UCRETI` | 252 |
| `VADE_SURESI` | 877 | `KAR_PAYLASIM_ORANI` | 90 |
| | | `KAR_PAYI_ORANI` | 62 |

### Kampanya türü — 9 sınıf

| Etiket | Belge |
|---|---:|
| Kart Kampanyası | 1.393 |
| Yatırım Ürünü Kampanyası | 667 |
| Finansman Kampanyası | 520 |
| Diğer / Kampanya Değil | 310 |
| Alışveriş Puanı Kampanyası | 136 |
| Taşıt Finansmanı Kampanyası | 108 |
| Yeni Müşteri Kampanyası | 96 |
| İhtiyaç Finansmanı Kampanyası | 66 |
| Konut Finansmanı Kampanyası | 55 |

---

## 6. Bölme (train / val / test)

Bölme **rastgele değil, kümelenerek** yapılır. Aynı sayfanın farklı yıllardaki
sürümleri korpusta ayrı kayıtlar olarak durur ve birbirine çok benzer; rastgele
bölünselerdi biri eğitime, diğeri teste düşer ve **eğitim–test sızıntısı** olurdu.

Sızıntı iki ayrı katmanda kapatılır:

1. **Belge düzeyi.** Kayıtlar içerik imzasına göre kümelenir (2.800 küme, en büyük
   küme 45 belge) ve **bütün bir küme tek bir bölüme** atanır.
2. **Pasaj düzeyi.** Aynı pasaj metni birden çok belgede geçebilir — kampanya
   koşullarındaki standart cümleler, oran tablosu başlıkları, aynı sayfanın farklı
   yıllardaki arşiv sürümleri. Bunlar bölmeden **önce** tekilleştirilir; 49.202
   kopya pasaj elendi.

> Doğrulama: `train ∩ val`, `train ∩ test`, `val ∩ test` — üçünde de **0** ortak
> pasaj metni. Belge kimliklerinde de bölümler arası çakışma yok.
>
> Not: ikinci katman olmadan 901 pasaj metni birden fazla bölümde görünüyor ve
> 3.142 pasaj kaydını etkiliyordu. Bu, teslim öncesi denetimde yakalanıp
> `build_dataset.py` içinde kalıcı olarak düzeltildi.

Bölme oranları ve rastgelelik tohumu `semalar/etiket_semasi.yaml` içindedir;
üretim yeniden çalıştırıldığında birebir aynı bölme elde edilir.

---

## 7. Bilinen sınırlar

1. **Etiketler doğrulanmamıştır** (bkz. §4.2). Bu sürüm silver'dır.
2. **Kategori dengesizliği.** `kart` kategorisi korpusun %47'sidir; arşivde kampanya
   sayfaları ürün sayfalarından çok daha yoğun arşivlenmiştir. Sınıf ağırlığı veya
   alt örnekleme düşünülmeli.
3. **Etiket dengesizliği.** `KAR_PAYI_ORANI` yalnızca 62 span; oran bilgisi kampanya
   metinlerinde nadir geçer, çoğunlukla ayrı oran tablolarındadır.
4. **Temsil dengesizliği.** Kuveyt Türk ve Türkiye Finans korpusun üçte birini
   oluşturur; T.O.M. ve Adil Katılım'da toplam 23 kayıt vardır. Bu, bankaların
   gerçek yaş ve büyüklük farkını yansıtır.
5. **Arşiv tarihi, yayın tarihi değildir.** `anlik_goruntu_tarihi` "bu içerik en geç
   bu tarihte yayındaydı" demektir.
6. **Kampanyanın yürürlükte olup olmadığı türetilmiş bilgi değildir.** Bitiş tarihi
   metnin içindedir ve modelin çıkarması gerekir.

Tam liste ve ayrıntılar: `dokuman/VERI_KARTI.md` §7.

---

## 8. Toplama yöntemi ve etik

- Banka listesi ve adresler **BDDK Kuruluş Listesi**'nden doğrulanarak alındı.
- **robots.txt** her host için indirilip uygulandı. Bu koşularda robots.txt engeli: **0**.
- Bankalar sırayla tarandı, paralel değil; istekler arası 2 sn + jitter. Eşzamanlı
  indirme yalnızca `web.archive.org` için ve 4 istekle sınırlı açıldı.
- Bot kendini `User-Agent` içinde iletişim adresiyle tanıttı.
- **Ücretli API veya lisans riski taşıyan kütüphane kullanılmadı.** Internet Archive
  CDX API'si ücretsiz ve anahtarsızdır.
- Yalnızca açık kaynak kütüphaneler: requests (Apache-2.0), BeautifulSoup4 (MIT),
  lxml (BSD-3), PyYAML (MIT), pandas (BSD-3), Playwright (Apache-2.0).
- **On-premise uyumlu:** toplama tek seferlik çevrimdışı bir toplu iştir; üretim
  sistemi ne banka sitelerine ne arşive bağımlıdır.
- İçerik bankaların kamuya açık web sayfalarından alınmıştır; akademik yarışma
  kapsamında araştırma amacıyla kullanılmaktadır.

---

*HititFinLex · TEKNOFEST 2026 TYDA 2. Senaryo · Sürüm v1.0 · 24 Ağustos 2026*
