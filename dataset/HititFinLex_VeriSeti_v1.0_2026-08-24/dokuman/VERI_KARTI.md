# HititFinLex Korpusu — Veri Kartı

**TEKNOFEST 2026 Yapay Zeka Dil Ajanları Yarışması / 2. Senaryo — Katılım Bankacılığı**

Bu belge, projenin ham veri korpusunu tanıtır: nereden, hangi tarih aralığından,
hangi yöntemle toplandığı ve içinde ne olduğu. Paket toplamlarının tek doğruluk
kaynağı üst dizindeki `MANIFEST.json` dosyasıdır; korpus sayıları
`korpus/korpus_tum.jsonl`, türetilmiş veri sayıları ise paketlenmiş JSONL
dosyaları üzerinde ölçülmüştür. Tahmin ya da yuvarlama yoktur.

Son güncelleme: **24 Ağustos 2026**

---

## 1. Tek bakışta

| | |
|---|---|
| **Kayıt sayısı** | **3.351** |
| Toplam metin | 8.181.060 karakter / 1.043.573 kelime |
| Kurum | BDDK listesindeki **10 katılım bankasının tamamı** |
| **İçerik tarih aralığı** | **1 Ocak 2016 → 21 Ağustos 2026** (10 yıl 8 ay) |
| Toplama tarihleri | 14 Ağustos 2026 (canlı) + 24 Ağustos 2026 (arşiv) |
| Kaynak türü | 771 canlı site (%23) + 2.580 web arşivi (%77) |
| Benzersiz sayfa adresi | 2.467 |
| Dil | Türkçe (yabancı dil sayfaları filtrelendi) |
| Biçim | JSONL / JSON / CSV — satır başına bir sayfa kaydı |

---

## 2. Ne zamandan itibaren veri çekiyoruz?

Bu, korpusun en ayırt edici özelliği, o yüzden en başta.

### 2.1 İki ayrı "tarih" var, karıştırmayın

| Alan | Anlamı | Aralık |
|---|---|---|
| `toplanma_tarihi` | Sayfayı **bizim indirdiğimiz** an | 14–24 Ağustos 2026 |
| `anlik_goruntu_tarihi` | Sayfanın **yayında olduğu** an (arşiv kayıtlarında) | 1 Oca 2016 – 21 Ağu 2026 |

Yani korpus 2026 Ağustos'ta toplandı, ama içindeki metinlerin çoğu geçmişe ait.
Bir kampanya sayfası 2019'da yayınlanıp kaldırılmış olsa bile metni elimizde,
üstelik **hangi tarihte yayında olduğu bilgisiyle birlikte**.

### 2.2 İçerik yaşı dağılımı

```
2016   79  ██████
2017   38  ███
2018   13  █
2019  255  ████████████████████
2020  221  █████████████████
2021  221  █████████████████
2022  107  ████████
2023  104  ████████
2024  343  ███████████████████████████
2025  554  ███████████████████████████████████████████
2026  645  ██████████████████████████████████████████████████
```

Bu dağılım 2.580 arşiv kaydı üzerindedir; 771 canlı kayıt zaten Ağustos 2026
içeriğidir. Korpus tek bir döneme yığılmamış: arşiv kayıtlarının %46'sı
2025–2026'dan gelse de 2016–2021 arası 827 kayıtla (%32) temsil ediliyor.

Bu yayılma kasıtlı: aynı sayfanın farklı yıllardaki sürümleri ayrı ayrı
alınıyor, böylece bir kâr payı oranı tablosunun 2019, 2022 ve 2026 hâlleri
korpusa üç farklı örnek olarak giriyor.

### 2.3 Neden 2016?

Alt sınırı biz koyduk (`banks.yaml → arsiv.baslangic_yili: 2016`). Arşivde daha
eskisi de var — Albaraka'nın sitesi **12 Aralık 1998**'den beri arşivleniyor.
2016'da kesmemizin sebebi:

- Katılım bankacılığı ürün ve kampanya sayfalarının bugünkü biçimi (kâr payı
  oranı tabloları, kampanya detay sayfaları) 2010'ların ortasında yerleşti.
- Daha eski sayfalar `<frameset>` tabanlı ve içerik olarak fakir; ayıklandığında
  200 karakterlik asgari eşiği çoğu zaman geçmiyor.
- Ölçüme göre 10 bankanın 5'inde arşiv zaten 2016'dan önce anlamlı içerik
  vermiyor (Vakıf Katılım'ın ilk kaydı 2016-02, Emlak 2019, Hayat Finans 2023,
  Dünya 2024, T.O.M./Adil hiç). Daha geriye gitmek yalnızca en eski üç bankaya
  (Albaraka, Kuveyt Türk, Türkiye Finans) ek kayıt getirirdi.

Bu bir konfigürasyon değeri; `baslangic_yili` düşürülüp koşu tekrarlanabilir.

### 2.4 Banka bazında kapsam

Her banka aynı tarihten başlamıyor — sebebi bankaların kendi yaşları ve alan
adı geçmişleri:

| Banka | Canlı | Arşiv | Toplam | Arşivdeki ilk kayıt | Son kayıt |
|---|---:|---:|---:|---|---|
| Kuveyt Türk | 99 | 492 | **591** | 2016-03-20 | 2026-08-06 |
| Türkiye Finans | 86 | 473 | **559** | 2016-01-09 | 2026-08-13 |
| Ziraat Katılım | 100 | 410 | **510** | 2016-01-13 | 2026-06-11 |
| Vakıf Katılım | 99 | 394 | **493** | 2016-02-28 | 2026-08-21 |
| Albaraka | 99 | 325 | **424** | 2016-01-01 | 2026-07-31 |
| Türkiye Emlak Katılım | 100 | 242 | **342** | 2019-07-22 | 2026-06-06 |
| Hayat Finans | 67 | 185 | **252** | 2023-11-29 | 2026-05-15 |
| Dünya Katılım | 98 | 59 | **157** | 2024-02-25 | 2026-08-11 |
| T.O.M. Katılım | 15 | — | **15** | — | — |
| Adil Katılım | 8 | — | **8** | — | — |

**Okuma notları:**

- İlk beş banka 2016 tabanına kadar iniyor; yani onlarda kapsam tamamen bizim
  koyduğumuz sınırla belirleniyor, arşiv daha geriye gidebilir.
- **Emlak Katılım 2019'da başlıyor** — bankanın katılım bankasına dönüşümüyle
  uyumlu. Dönüşüm öncesi `emlakbank.com.tr` alan adını da tarıyoruz, korpusta
  o adresten 38 kayıt var.
- **Hayat Finans (2023-11) ve Dünya Katılım (2024-02)** yeni bankalar; arşivde
  daha eskisi yok.
- **T.O.M. ve Adil Katılım'da arşiv kaydı yok.** CDX sorgusu eşleşen kampanya
  veya ürün adresi döndürmedi (Adil Katılım'ın sitesi hâlâ bir "yakında"
  sayfası). Bu iki banka için arşiv toplama `banks.yaml`'da kapalı; boşuna
  sorgu atılmıyor. Canlı kayıtları korpusta duruyor.

---

## 3. Nereden topluyoruz

Banka listesi ve resmî adresler **BDDK Kuruluş Listesi**'nden
(`bddk.org.tr/Kurulus/Liste/77`) doğrulanarak alındı; 14 Ağustos 2026 itibarıyla
10 kuruluşun tamamı "faaliyette".

Korpustaki kayıtların alan adı dağılımı — eski alan adları dâhil:

| Alan adı | Kayıt |
|---|---:|
| kuveytturk.com.tr (www + çıplak) | 591 |
| turkiyefinans.com.tr | 558 |
| ziraatkatilim.com.tr | 510 |
| vakifkatilim.com.tr | 493 |
| albaraka.com.tr | 424 |
| emlakkatilim.com.tr | 304 |
| hayatfinans.com.tr | 252 |
| dunyakatilim.com.tr | 157 |
| **emlakbank.com.tr** (eski alan adı) | **38** |
| tombank.com.tr | 15 |
| adilkatilim.com.tr | 8 |
| hizlifinansman.com.tr | 1 |

`emlakbank.com.tr` kayıtları, bankanın katılım bankasına dönüşmeden önceki
sayfalarıdır — özellikle hedeflendi. Albaraka için de eski `albarakaturk.com.tr`
alan adı taranıyor, ancak arşivinde filtreyi geçen sayfa çıkmadı.

`hizlifinansman.com.tr` tek kayıt, Türkiye Finans'ın canlı sitesinden verilen
bir yönlendirmenin hedefidir (bkz. §8, bilinen sınırlar).

---

## 4. İki kaynak, tek şema

Korpus iki toplama koşusunun birleşimidir ve **her iki kaynak da aynı kayıt
şemasını kullanır**. Ayrımı tek bir alan yapar: `guncel_mi`.

| | Canlı toplama | Arşiv toplama |
|---|---|---|
| Komut | `python main.py` | `python main.py --arsiv` |
| Kaynak | Bankaların bugünkü siteleri | Internet Archive (Wayback Machine) |
| Keşif | sitemap.xml → yoksa link taraması | CDX Server API |
| Kayıt | 771 | 2.580 |
| `guncel_mi` | `true` | `false` |
| `anlik_goruntu_tarihi` | boş | yakalama tarihi (UTC) |

### ⚠️ Kritik kullanım kuralı

- **Model eğitimi (bilgi çıkarım + sınıflandırma): ikisini birlikte kullanın.**
  Süresi dolmuş bir kampanyanın metni de tamamen geçerli bir etiketleme
  örneğidir — kâr payı oranı, vade, tutar span'ları aynı biçimde geçer.
- **Kullanıcıya sunum (dashboard, chatbot): `guncel_mi=false` olanları eleyin.**
  2019 tarihli bir kampanyayı yürürlükteymiş gibi göstermek hatalı bilgi verir.

```python
import pandas as pd
df = pd.read_json("katilim_bankalari_tum.jsonl", lines=True)

egitim = df                                    # her şey
guncel = df[df["guncel_mi"].fillna(True)]      # yalnızca yürürlükteki içerik
```

---

## 5. Kayıt şeması

### Şartnamede istenen çekirdek alanlar

| Alan | Açıklama |
|---|---|
| `banka_adi` | BDDK'daki resmî unvan |
| `kaynak_url` | Bankanın kendi adresi (arşiv kayıtlarında da bankanın adresi, arşivinki değil) |
| `sayfa_basligi` | `og:title` → `h1` → `<title>` sırasıyla |
| `ham_metin` | Boilerplate'ten ayıklanmış ana içerik metni |
| `toplanma_tarihi` | Bizim indirdiğimiz an (ISO-8601, saat dilimli) |
| `kategori_tahmini` | `finansman` / `kart` / `yatirim` / `hesap` / `diger` |

### İzlenebilirlik ve filtreleme meta verisi

| Alan | Açıklama |
|---|---|
| `kayit_id` | Benzersiz kayıt kimliği (arşiv kayıtlarında yakalama tarihi de gömülü) |
| `banka_key` | Makine dostu banka anahtarı |
| `breadcrumb` | Sayfanın site içi konumu |
| `kampanya_mi` | Kampanya mı, ürün tanıtımı mı (bağlayıcı olmayan ipucu) |
| `karakter_sayisi`, `kelime_sayisi` | Metin uzunluğu |
| `render_yontemi` | `static` / `playwright` |
| `http_durum` | HTTP yanıt kodu |
| `icerik_parmak_izi` | Metin özeti — tekilleştirme anahtarı |
| `ham_html_dosyasi` | `data/raw/` altındaki ham HTML'in yolu |

### Geçmişe dönük toplamayla eklenen alanlar

| Alan | Açıklama |
|---|---|
| `guncel_mi` | `true` = canlı, `false` = arşiv |
| `anlik_goruntu_tarihi` | Yakalamanın tarihi (ISO-8601, UTC) |
| `arsiv_kaynagi` | `wayback` (canlı kayıtlarda boş) |
| `arsiv_url` | Yakalamanın yeniden üretilebilir tam adresi |

`arsiv_url` sayesinde her arşiv kaydı doğrulanabilir: adresi tarayıcıya
yapıştırınca sayfanın o tarihteki hâli açılır.

---

## 6. İçerikte ne var

### 6.1 Kategori dağılımı

| Kategori | Kayıt | Pay | Canlı | Arşiv |
|---|---:|---:|---:|---:|
| kart | 1.583 | %47,2 | 296 | 1.287 |
| finansman | 921 | %27,5 | 207 | 714 |
| yatirim | 409 | %12,2 | 146 | 263 |
| diger | 272 | %8,1 | 59 | 213 |
| hesap | 166 | %5,0 | 63 | 103 |

### 6.2 Kampanya / ürün ayrımı

| | Kayıt | Pay |
|---|---:|---:|
| Kampanya | 2.522 | %75,3 |
| Ürün / bilgi sayfası | 582 | %17,4 |
| Belirsiz | 247 | %7,4 |

### 6.3 Metin uzunluğu

| | Karakter |
|---|---:|
| En kısa | 200 (asgari eşik) |
| 1. çeyrek | 919 |
| **Medyan** | **1.450** |
| 3. çeyrek | 2.649 |
| %95'lik dilim | 6.986 |
| En uzun | 68.543 |
| Ortalama | 2.441 |

En uzun kayıtlar "Ürün ve Hizmet Ücretleri" tablolarıdır; NLP aşamasında
pasajlara bölünerek işlenirler.

### 6.4 NLP aşaması için sinyal yoğunluğu

Şartname 5.3'te istenen alanların korpusta ne sıklıkta göründüğü — kaba
düzenli ifade taramasıyla ölçüldü:

| Aranan | Kayıt | Pay |
|---|---:|---:|
| Tutar (TL / ₺) | 1.585 | %47,3 |
| Vade (ay / taksit) | 1.254 | %37,4 |
| Oran ifadesi (% / yüzde) | 1.042 | %31,1 |
| Geçerlilik tarihi | 493 | %14,7 |
| "kâr payı" ifadesi | 349 | %10,4 |
| Tahsis ücreti / dosya masrafı | 304 | %9,1 |

Kâr payı oranı özellikle hedeflendi: kampanya metinlerinde nadiren geçtiği
ölçüldüğü için "Kâr Payı Oranları" ve "Ürün ve Hizmet Ücretleri" sayfaları
URL filtrelerinde ve önceliklendirmede en yüksek ağırlığı alıyor. Arşiv
tarafında bu sayfaların **yıllara göre farklı sürümleri** var — tek bir
sayfadan onlarca farklı oran örneği çıkıyor.

---

## 7. Kalite ve tekilleştirme

- `kayit_id` benzersiz: **3.351/3.351** ✅
- `icerik_parmak_izi` benzersiz: **3.351/3.351** ✅ (birebir aynı metin iki kez yok)
- 200 karakterin altındaki sayfalar hiç kaydedilmedi.
- Menü/footer artıkları iki kademede temizlendi: sayfa içi boilerplate ayıklama
  + aynı bankanın sayfalarının %60'ından fazlasında geçen satırların atılması.
- Arşiv koşusunda 495 kopya sayfa, 183 içeriksiz/kısa sayfa elendi.
- Birleştirmede, canlı sürümüyle aynı metne sahip **90 arşiv kaydı** düşürüldü;
  çakışmada **canlı sürüm** korunur.
- Arşiv koşusunda 11 hata (%0,3): CDX yakalamayı listeliyor ama arşivin oynatma
  ucu 404 dönüyor. Veri kaybı ihmal edilebilir.

### Aynı sayfanın birden çok sürümü

| Bir adres için kayıt | Adres sayısı |
|---:|---:|
| 1 | 1.662 |
| 2 | 726 |
| 3 | 79 |

2.467 benzersiz adresten 3.351 kayıt çıkıyor. Bir adresin en fazla 2 arşiv
yakalaması alınıyor (zamana yayılarak: en eski + en yeni), üstüne canlı sürüm
de varsa 3 oluyor.

> **Bölme uyarısı.** Aynı sayfanın sürümleri birbirine çok benzer. Veri setini
> train/val/test'e bölerken bunların **aynı bölüme** düşmesi şart, yoksa
> eğitim–test sızıntısı olur. Mevcut `dataset_builder/utils/splits.py` bunu
> içerik imzasıyla kümeleyerek zaten yapıyor; bölmeyi doğrudan kayıt düzeyinde
> yapan yeni bir kod yazılırsa risk geri gelir.

---

## 8. Bilinen sınırlar

1. **Kategori dengesizliği.** `kart` %47'ye çıktı. Sebebi arşivde kampanya
   sayfalarının ürün sayfalarından çok daha yoğun arşivlenmiş olması. Canlı
   korpusta dağılım daha dengeliydi. Sınıflandırma eğitiminde sınıf ağırlığı
   veya alt örnekleme düşünülmeli.

2. **Temsil dengesizliği.** İki büyük banka (Kuveyt Türk, Türkiye Finans)
   korpusun üçte birini oluşturuyor; T.O.M. ve Adil Katılım'da toplam 23 kayıt
   var. Bu, bankaların gerçek büyüklük ve yaş farkını yansıtıyor, bir toplama
   hatası değil — ama bankalar arası karşılaştırma yapılırken akılda tutulmalı.

3. **22 kayıtta `http_durum` boş.** Playwright ile render edilen canlı
   sayfalar; tarayıcı yolu HTTP kodu döndürmüyor. İçerik sağlam.

4. **2 kayıtta `http_durum` 404.** Albaraka ve Kuveyt Türk'ün, 404 dönerken
   yine de kampanya içeriği gösteren "yumuşak 404" sayfaları. İçerikleri
   geçerli, ama bilinçli bir tercih değil.

5. **1 kayıt banka alan adı dışında** (`hizlifinansman.com.tr`). Türkiye
   Finans'ın canlı sitesinden verilen bir yönlendirmenin hedefi;
   `kaynak_url` yönlendirme sonrası adresi tuttuğu için böyle görünüyor.

6. **Arşiv, sayfanın yayınlandığı tarihi değil, arşivlendiği tarihi verir.**
   Bir kampanya Mart'ta yayınlanıp Ağustos'ta arşivlenmişse
   `anlik_goruntu_tarihi` Ağustos'tur. Yani tarih "bu içerik en geç bu tarihte
   yayındaydı" demektir, "bu tarihte yayınlandı" demek değildir.

7. **Kampanyanın gerçekten yürürlükte olup olmadığı türetilmiş bir bilgi
   değildir.** `guncel_mi=true` yalnızca "toplama anında sitede yayındaydı"
   der; kampanyanın bitiş tarihi metnin içindedir ve NLP aşamasının çıkarması
   gerekir.

---

## 9. Türetilmiş veri seti (durum)

`dataset_builder/` modülü bu korpustan iki etiketli veri seti üretir:

| Görev | Şartname | Etiketler |
|---|---|---|
| Bilgi çıkarım (span) | 5.3 | `KAR_PAYI_ORANI`, `KAR_PAYLASIM_ORANI`, `FINANSMAN_TUTARI`, `VADE_SURESI`, `TAKSIT_SAYISI`, `TAHSIS_UCRETI`, `MASRAF_DURUMU`, `HARCAMA_ESIGI`, `ODUL_MIKTARI`, `INDIRIM_ORANI`, `ALISVERIS_PUANI`, `KAMPANYA_SURESI`, `HEDEF_KITLE` |
| Kampanya türü (sınıflandırma) | 5.4 | Konut / Taşıt / İhtiyaç / Finansman / Alışveriş Puanı / Kart / Yeni Müşteri / Yatırım Ürünü / Diğer |

Depodaki yayın paketi güncel birleşik korpusla eşleşir: **3.351 belge**,
**7.238 pasaj** ve **10.501 span**. Bilgi çıkarımı bölümleri 5.015 eğitim,
1.177 doğrulama ve 1.046 test pasajından; sınıflandırma bölümleri 2.346
eğitim, 504 doğrulama ve 501 test belgesinden oluşur. Bu toplamlar
`MANIFEST.json` ile birebir aynıdır.

Etiket durumu **silver**'dır: kural tabanlı ön etiketleme ve otomatik kalite
kontrolleri uygulanmıştır, ancak bütün span'ler bağımsız insan gold etiketi
değildir. Yeni bir model sonucu raporlanırken bu sınırlama belirtilmelidir.

Yeniden üretmek için:

```bash
cd VeriToplama/dataset_builder
python build_dataset.py --korpus ../data_collection/data/processed/katilim_bankalari_tum.jsonl
```

`guncel_mi` ve `anlik_goruntu_tarihi` alanları final veri setine kadar
taşınıyor, böylece dashboard/chatbot katmanı arşiv kayıtlarını eleyebiliyor.

---

## 10. Korpusu sıfırdan yeniden üretme

```bash
cd VeriToplama/data_collection
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt
playwright install chromium

python main.py              # canlı toplama      (~40 dk,  771 kayıt)
python main.py --arsiv      # geçmişe dönük      (~41 dk, 2.670 kayıt)
python main.py --birlestir  # tek korpusta birleştir → 3.351 kayıt
```

Ayrıntılı mimari, ayarlar ve saha notları için:
[`veri_toplama_README.md`](veri_toplama_README.md)

### Çıktı dosyaları

| Dosya | İçerik |
|---|---|
| `katilim_bankalari.jsonl` | Yalnızca canlı (771) |
| `katilim_bankalari_arsiv.jsonl` | Yalnızca arşiv (2.670, tekilleştirme öncesi) |
| **`katilim_bankalari_tum.jsonl`** | **Birleşik korpus (3.351) — NLP girişi** |
| `toplama_ozeti.json` / `toplama_ozeti_arsiv.json` | Makine okunur koşu özetleri |
| `data/raw/<banka>/*.html` | Ham HTML arşivi (~1,2 GB, git dışı) |

Ham HTML bilinçli olarak saklanıyor: ayıklama kuralları değişirse bankaların
sitelerini yeniden ziyaret etmeden yeniden işleme yapılabilir. Bu aynı zamanda
sonuçların tekrar üretilebilirliğini sağlar.

---

## 11. Etik, lisans ve şartname uyumu

- **robots.txt** her host için indirilip uygulandı; engellenen adres
  indirilmedi. Bu koşularda robots.txt engeli sayısı: **0**.
- Bankalar **sırayla** tarandı, paralel değil; istekler arası 2 sn + jitter.
  Eşzamanlı indirme yalnızca `web.archive.org` için ve 4 istekle sınırlı
  açıldı — banka sitelerine hiçbir koşulda uygulanmıyor.
- Bot kendini `User-Agent` içinde iletişim adresiyle tanıtıyor.
- **Ücretli API veya lisans riski taşıyan kütüphane yok.** Internet Archive
  CDX API'si ücretsiz ve anahtarsızdır.
- **On-premise uyumlu:** toplama tek seferlik çevrimdışı bir toplu iştir;
  üretim sistemi (dashboard/chatbot) ne banka sitelerine ne arşive bağımlıdır.
- İçerik bankaların kamuya açık web sayfalarından alınmıştır; akademik yarışma
  kapsamında araştırma amacıyla kullanılmaktadır.
