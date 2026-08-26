# HititFinLex — Katılım Bankacılığı Veri Toplama Modülü

TEKNOFEST 2026 Yapay Zeka Dil Ajanları Yarışması / **2. Senaryo — Katılım
Bankacılığı** kapsamında geliştirilen veri toplama (web scraping) modülü.

BDDK'nın resmî listesindeki **10 katılım bankasının** web sitelerinden
finansman, kart ve yatırım ürünlerine ait kampanya/ürün metinlerini toplar ve
sonraki NLP aşamasına (bilgi çıkarımı, sınıflandırma, dashboard, chatbot)
yapılandırılmış girdi üretir.

> **Kapsam notu:** Bu modül **yalnızca metin toplar**. Kâr payı oranı, vade,
> kampanya avantajı gibi alanların çıkarımı bilinçli olarak bu modülün dışında
> bırakılmıştır; o iş NLP/bilgi çıkarım aşamasınındır. Buradaki tek sorumluluk,
> ham ve temiz metni güvenilir biçimde toplayıp `(banka, url, başlık, metin,
> tarih, kategori)` yapısında saklamaktır.

---

## 1. Hızlı Başlangıç

### Gereksinimler
- **Python 3.10+** (3.12 ile geliştirildi ve test edildi)
- İnternet erişimi
- ~200 MB disk (Playwright'ın Chromium'u dahil)

### Kurulum

```bash
# 1) Depoyu/klasörü alın ve modül dizinine girin
cd VeriToplama/data_collection

# 2) Sanal ortam oluşturun ve etkinleştirin
python -m venv ../.venv

#    Windows (PowerShell)
../.venv/Scripts/Activate.ps1
#    Windows (cmd)
..\.venv\Scripts\activate.bat
#    Linux / macOS
source ../.venv/bin/activate

# 3) Bağımlılıkları kurun
pip install -r requirements.txt

# 4) (Opsiyonel ama önerilir) JS ile render edilen siteler için tarayıcı
playwright install chromium
```

> **4. adım atlanırsa ne olur?** Modül çalışmaya devam eder. Yalnızca içeriği
> istemci tarafında üreten siteler (bkz. Adil Katılım) atlanır ve özet raporda
> `[ATLA]` durumuyla, gerekçesiyle birlikte raporlanır.

### Çalıştırma

```bash
python main.py                                  # 10 bankanın tamamı
python main.py --list                           # banka anahtarlarını listele
python main.py --banks kuveyt_turk,albaraka     # yalnızca seçilenler
python main.py --limit 20                       # banka başına en fazla 20 kayıt
python main.py --dry-run                        # sadece URL keşfi (indirme yok)
python main.py --no-playwright                  # yalnızca statik HTTP
python main.py --delay 4                        # daha yavaş/nazik tarama
python main.py --log-level DEBUG                # ayrıntılı konsol logu

python main.py --arsiv                          # GEÇMİŞE DÖNÜK toplama (web arşivi)
python main.py --arsiv --banks kuveyt_turk      # tek bankanın arşivi
python main.py --arsiv --dry-run                # arşivde hangi sayfalar var?
python main.py --arsiv --arsiv-eszamanli 2      # arşiv host'una daha nazik davran
python main.py --birlestir                      # canlı + arşiv korpusunu birleştir
```

Tam listeyi görmek için: `python main.py --help`

### İlk kez çalıştıranlar için önerilen sıra

```bash
python main.py --list                # konfigürasyon okunuyor mu?
python main.py --dry-run --limit 5   # hangi URL'ler seçiliyor? (ağ dostu)
python main.py --limit 5             # küçük bir uçtan uca deneme
python main.py                       # tam toplama (canlı siteler)
python main.py --arsiv               # geçmişe dönük toplama (bkz. bölüm 3.1)
python main.py --birlestir           # NLP aşamasına verilecek tek korpus
```

---

## 2. Klasör Yapısı

```
data_collection/
├── config/
│   └── banks.yaml           Banka listesi, hedef URL'ler, filtreler, hız ayarları
├── scrapers/
│   ├── __init__.py          Scraper sınıflarını YAML'daki isimden yükleyen kayıt defteri
│   ├── base_scraper.py      Ortak arayüz: discover / fetch / parse / save
│   ├── archive_scraper.py   Banka scraper'ını web arşivine bağlayan mixin
│   ├── adil_katilim.py      ─┐
│   ├── albaraka.py           │
│   ├── dunya_katilim.py      │
│   ├── hayat_finans.py       │ Her banka için bir sınıf.
│   ├── kuveyt_turk.py        │ Yalnızca o bankaya özgü farkları içerir;
│   ├── tom_bank.py           │ ağ/temizlik/robots mantığı base'dedir.
│   ├── emlak_katilim.py      │
│   ├── turkiye_finans.py     │
│   ├── vakif_katilim.py      │
│   └── ziraat_katilim.py    ─┘
├── utils/
│   ├── config.py            banks.yaml okuma ve doğrulama
│   ├── http_client.py       Hız sınırlama, retry, robots.txt, Playwright fallback
│   ├── wayback.py           Internet Archive CDX istemcisi (geçmişe dönük keşif)
│   ├── text_cleaner.py      HTML→metin ayıklama ve boilerplate temizliği
│   ├── categorizer.py       Kategori tahmini + süresi dolmuş kampanya tespiti
│   ├── storage.py           Ham HTML ve işlenmiş kayıtların yazımı
│   └── logging_setup.py     Konsol + dosya loglama
├── data/
│   ├── raw/<banka>/         Ham HTML + `_manifest.jsonl`
│   └── processed/           Temizlenmiş kayıtlar (JSONL / JSON / CSV) + özet
├── logs/                    Çalıştırma logları (her koşu için ayrı dosya)
├── main.py                  Orkestratör (CLI)
├── requirements.txt
└── README.md
```

---

## 3. Çıktılar

| Dosya | İçerik |
|---|---|
| `data/processed/katilim_bankalari.jsonl` | Tüm bankalar, satır başına bir kayıt (NLP aşaması için önerilen giriş) |
| `data/processed/katilim_bankalari.json` | Aynı veri, tek JSON dizisi |
| `data/processed/katilim_bankalari.csv` | Aynı veri, CSV (`utf-8-sig`, Excel uyumlu) |
| `data/processed/<banka_key>.jsonl` | Banka bazlı kayıtlar |
| `data/processed/toplama_ozeti.json` | Makine tarafından okunabilir çalışma özeti |
| `data/processed/katilim_bankalari_arsiv.*` | `--arsiv` koşusunun çıktısı (aynı şema, `guncel_mi=false`) |
| `data/processed/<banka_key>_arsiv.jsonl` | Banka bazlı arşiv kayıtları |
| `data/processed/toplama_ozeti_arsiv.json` | Arşiv koşusunun özeti (yakalama yılı dağılımı dâhil) |
| `data/processed/katilim_bankalari_tum.*` | `--birlestir` çıktısı: canlı + arşiv, tekilleştirilmiş |
| `data/raw/<banka_key>/*.html` | Ham HTML arşivi |
| `data/raw/<banka_key>/_manifest.jsonl` | URL → dosya eşlemesi, HTTP durumu, render yöntemi |
| `logs/veri_toplama_*.log` | Ayrıntılı çalışma logu (DEBUG seviyesinde) |

### Kayıt şeması

Şartnamede istenen çekirdek alanlar:

| Alan | Açıklama |
|---|---|
| `banka_adi` | BDDK'daki resmî unvan |
| `kaynak_url` | Yönlendirmeler çözülmüş kanonik URL |
| `sayfa_basligi` | `og:title` → `h1` → `<title>` sırasıyla |
| `ham_metin` | Temizlenmiş ana içerik metni |
| `toplanma_tarihi` | ISO-8601, saat dilimli |
| `kategori_tahmini` | `finansman` / `kart` / `yatirim` / `hesap` / `diger` |

NLP aşamasında izlenebilirlik ve filtreleme için eklenen meta alanlar:
`kayit_id`, `banka_key`, `breadcrumb`, `kampanya_mi`, `karakter_sayisi`,
`kelime_sayisi`, `render_yontemi`, `http_durum`, `icerik_parmak_izi`,
`ham_html_dosyasi`.

Geçmişe dönük toplamayla eklenen alanlar (bkz. bölüm 3.1):

| Alan | Açıklama |
|---|---|
| `guncel_mi` | `true` = canlı siteden toplandı, `false` = web arşivinden |
| `anlik_goruntu_tarihi` | Arşiv yakalamasının tarihi (ISO-8601, UTC) |
| `arsiv_kaynagi` | Arşiv sağlayıcısı (`wayback`); canlı kayıtlarda boş |
| `arsiv_url` | Yakalamanın yeniden üretilebilir tam adresi |

**Ham HTML neden saklanıyor?** Ayıklama kuralları NLP aşamasında
değiştiğinde, banka sitelerini yeniden ziyaret etmeden `data/raw/` üzerinden
yeniden işlem yapılabilsin diye. Bu aynı zamanda sonuçların tekrar
üretilebilirliğini (reproducibility) sağlar.

---

## 3.1 Geçmişe Dönük Toplama (`--arsiv`)

### Neden

Kampanya sayfaları kısa ömürlüdür. Bir banka bugün sitesinde 30 kampanya
yayınlıyorsa, son yedi yılda yayınlayıp kaldırdığı yüzlercesi vardır. Bu
metinler NLP aşaması için **hâlâ geçerli eğitim verisidir**: kâr payı oranı,
vade, tutar ve geçerlilik tarihi span'ları aynı biçimde geçer. Üstelik
kâr payı oranı tabloları (`finansman-kar-oranlari.aspx` gibi) arşivde
yıllara göre farklı değerlerle durur — tek bir sayfadan onlarca farklı
oran örneği çıkar.

Kaldırılmış sayfaların metni Internet Archive'ın Wayback Machine'inde
duruyor ve **CDX Server API** ile programatik olarak listelenebiliyor.

### Nasıl çalışır

```
banks.yaml (arsiv:)      →  hangi alan adları, hangi yıllar, kaç yakalama
        ↓
utils/wayback.py         →  CDX sorgusu: statuscode=200, mimetype=text/html,
                            kampanya/ürün URL filtresi, collapse=digest
        ↓
select_snapshots()       →  URL başına N yakalama, zamana yayılarak
                            (aynı digest ve utm_* varyantları elenir)
        ↓
archive_scraper.py       →  web.archive.org/web/<ts>id_/<url> adresinden indir
        ↓
bankanın kendi scraper'ı →  CONTENT_SELECTORS ile ayıklama (değişmez)
        ↓
data/processed/<banka>_arsiv.jsonl
```

`id_` soneki kritiktir: Wayback'in enjekte ettiği araç çubuğu/JS olmadan
**orijinal HTML**'i döndürür. Böylece mevcut ayıklama katmanı hiç
değişmeden çalışır ve arşiv kayıtları canlı kayıtlarla aynı şemayı taşır.

Mimari olarak arşiv toplama ayrı bir scraper ailesi değil, **mixin**'dir:

```
ArsivAlbarakaScraper = make_archive_scraper(AlbarakaScraper)
# MRO: ArsivAlbaraka → ArchiveScraperMixin → Albaraka → BaseScraper
```

Bankaya özel CSS seçicileri ve ayıklama kancaları korunur; yalnızca keşif
(CDX) ve kayıt meta verisi (yakalama tarihi + `guncel_mi=false`) değişir.

> **Tek bilinçli gevşetme:** bankaya özel `should_collect` ezmeleri arşiv
> modunda uygulanmaz. Bu ezmeler bugünkü URL yapısına göre yazılmıştır
> (ör. Albaraka'da "URL'de `/tr/` olmalı"); 2019'daki adresler o yapıya
> uymadığı için uygulanırsa arşivin tamamı elenir. Yerine `banks.yaml`
> içindeki include/exclude/dil kalıpları, bankanın eski alan adlarını da
> içeren genişletilmiş host listesiyle uygulanır.

### Kritik kural: `guncel_mi` alanı

Arşiv kayıtları korpusa `guncel_mi=false` ile girer.

* **Bilgi çıkarım ve sınıflandırma modelleri** ikisini birlikte kullanır —
  süresi dolmuş bir kampanyanın metni de geçerli bir etiketleme örneğidir.
* **Dashboard ve chatbot bu kayıtları FİLTRELEMELİDİR.** Kullanıcıya 2019
  tarihli bir kampanyayı yürürlükteymiş gibi sunmak hatalı bilgi verir.

```python
guncel = [k for k in kayitlar if k.get("guncel_mi", True)]
```

### Yapılandırma (`banks.yaml`)

`defaults.arsiv` bloğu tüm bankalara uygulanır; banka bloğundaki `arsiv:`
alt bloğu bunu ezer:

| Ayar | Anlamı |
|---|---|
| `etkin` | `false` ise banka `--arsiv` koşusunda atlanır (arşivde kaydı olmayan yeni bankalar) |
| `alan_adlari` | Sorgulanacak alan adları; bankanın **eski** alan adı da yazılır |
| `baslangic_yili` / `bitis_yili` | Yakalama tarih aralığı |
| `url_basina_azami_yakalama` | Aynı sayfanın kaç sürümü alınsın (zamana yayılır) |
| `azami_kayit` | Banka başına azami arşiv kaydı |
| `url_filtresi` | CDX'e gönderilen sunucu taraflı kaba filtre |

Eski alan adları önemlidir: Albaraka `albarakaturk.com.tr`'den
`albaraka.com.tr`'ye, Türkiye Emlak Katılım ise `emlakbank.com.tr`'den
`emlakkatilim.com.tr`'ye taşınmıştır; her ikisinin arşivi de o adreslerde.

### Şartname uyumu

* CDX API **ücretsiz ve anahtarsızdır** — ücretli API/servis yasağı ihlal
  edilmez.
* Toplama tek seferlik çevrimdışı bir toplu iştir; üretim sistemi
  (dashboard/chatbot) arşive bağımlı değildir, kurum içi çalışmayı
  engellemez.
* İstekler aynı `HttpClient` üzerinden gider: hız sınırlama, yeniden
  deneme ve robots.txt kontrolü aynen uygulanır.
  (Gözlem: `web.archive.org/robots.txt` 404 döner, yani kısıt yoktur.)

### Hız: neden arşiv host'unda eşzamanlı istek var

İlk sıralı koşu 2,5 saat sürdü. Nedeni ölçüldü ve sezgiye aykırı çıktı:

| | Süre | Pay |
|---|---:|---:|
| Sunucunun yanıt beklemesi (3.181 istek × ort. 2,58 sn) | 8.212 sn | **%91** |
| Hız sınırı beklemesi | ~760 sn | %9 |

Yani darboğaz nezaket değil, arşivin kendi gecikmesiydi; `--delay` düşürmek
en fazla 12 dakika kazandırırdı. Etkin hız istek başına ~57 KB/sn olduğu
için sorun bant genişliği de değil — sadece gecikme. Tek çözüm istekleri
üst üste bindirmek.

Bu yüzden `HttpClient` tek bir host için eşzamanlılık tavanı açabiliyor
(`set_host_concurrency`). İşçi sayısı kıyaslaması (49 sayfa):

| İşçi | sn/sayfa | Yeniden deneme |
|---:|---:|---:|
| 1 | 2,57 | 0 |
| **4** | **0,74** | **0** |
| 6 | 0,86 | 4 |
| 8 | 0,74 | 2 (+1 hata) |

4'ten sonra kazanç yok, archive.org kısmaya başlıyor. Varsayılan **4**;
`arsiv.eszamanli_istek` ile ya da `--arsiv-eszamanli N` ile değiştirilebilir.

> **Bu ayar yalnızca `web.archive.org`'a uygulanır.** Banka siteleri her
> zaman katı sıralı taranır — şartnamenin "agresif tarama yok" kuralı
> onlar için geçerli ve değişmedi. Eşzamanlılık açılan host'ta istek
> başlangıçları `request_delay / işçi` aralıklarla dağıtılır, jitter de
> aynı oranda küçültülür.

İş parçacığı güvenliği için: her iş parçacığı kendi `requests.Session`'ını
kullanır (`Session` çerez kavanozu nedeniyle paylaşılamaz) ve sayaçlar
kilitli artırılır. Ayıklama/kayıt tek iş parçacığında sıralı kalır; yalnızca
indirme paralelleşir. **Doğrulandı:** 4 işçili çıktı, sıralı koşunun
çıktısıyla alan alan birebir aynı.

### Maliyet

4 işçiyle 2.670 kayıt için **41 dakika** ve **~1 GB** ham HTML
(gitignore'da) beklenmelidir. Diskten tasarruf için `--no-raw`
kullanılabilir, ancak o zaman ayıklama kuralları değiştiğinde yeniden
indirme gerekir.

### Ölçülen sonuç (2026-08-24 koşusu)

`python main.py --arsiv` (4 eşzamanlı işçi) → **41 dakika**, 3.349 sayfa
denendi, **2.670 kayıt**, 498 MB indirildi, 11 hata (hepsi arşivin `id_`
uç noktasının 404'ü; CDX kaydı listeliyor ama oynatma yapamıyor — %0,3),
4 yeniden deneme.

Aynı iş sıralı indirmeyle 2 saat 30 dakika sürüyordu (bkz. "Hız" alt
bölümü); sonuçlar eşdeğer çıktı (2.664 → 2.670 kayıt; fark, sıralı koşuda
zaman aşımına uğrayan 7 sayfanın bu kez başarılı olması).

| Banka | Canlı | Arşiv | Toplam |
|---|---:|---:|---:|
| Kuveyt Türk | 99 | 492 | 591 |
| Türkiye Finans | 86 | 473 | 559 |
| Ziraat Katılım | 100 | 410 | 510 |
| Vakıf Katılım | 99 | 394 | 493 |
| Albaraka | 99 | 325 | 424 |
| Emlak Katılım | 100 | 242 | 342 |
| Hayat Finans | 67 | 185 | 252 |
| Dünya Katılım | 98 | 59 | 157 |
| T.O.M. | 15 | — | 15 |
| Adil Katılım | 8 | — | 8 |
| **Toplam** | **771** | **2.580** | **3.351** |

Arşiv sütunu, `--birlestir` sonrası (canlı sürümüyle aynı metne sahip 90
arşiv kaydı tekilleştirmede elendi) değerleri gösterir. Birleşik korpus
3.351 kayıt / 8,18 milyon karakter; `kayit_id` ve `icerik_parmak_izi`
alanlarının benzersizliği doğrulandı.

Yakalama yılı dağılımı — korpus tek bir döneme yığılmamış:

```
2016:  79   2019: 256   2022: 107   2025: 572
2017:  38   2020: 226   2023: 107   2026: 700
2018:  13   2021: 223   2024: 349
```

> **Dikkat edilecek yan etki:** arşivde kampanya sayfaları ürün sayfalarından
> daha yoğun temsil ediliyor. Birleşik korpusta `kart` kategorisi %47'ye
> (1.583/3.351) çıktı; canlı korpusta bu oran çok daha dengeliydi. Sınıf
> dengesi gerektiren eğitimlerde bu göz önünde bulundurulmalı.

---

## 4. Mimari ve Veri Akışı

```
main.py  (bankaları SIRAYLA gezer, paralel değil)
   │
   └── BaseScraper.run()  ── her banka için bağımsız, hata izole
         │
         ├── discover()   URL keşfi
         │     ├── 1) sitemap.xml  (varsa; sitemapindex takibi dahil)
         │     │      └── geçersizse (XML değilse) otomatik olarak ↓
         │     └── 2) link taraması (seed sayfalardan, derinlik sınırlı)
         │           └── include/exclude/dil filtreleri + önceliklendirme
         │
         ├── fetch(url)   HttpClient
         │     ├── robots.txt kontrolü      (engelliyse sayfa atlanır)
         │     ├── hız sınırlama            (host bazında, Crawl-delay'e uyar)
         │     ├── retry + üstel geri çekilme (429/5xx/timeout)
         │     └── Playwright fallback      (aşağıya bakınız)
         │
         ├── parse()      BeautifulSoup
         │     ├── boilerplate temizliği (3 kademeli, aşağıya bakınız)
         │     ├── ana içerik konteyneri seçimi
         │     └── başlık + breadcrumb + kategori tahmini
         │
         └── save()       Storage
               ├── ham HTML  → data/raw/<banka>/
               └── kayıtlar  → data/processed/<banka>.jsonl
                     └── öncesinde: sayfalar arası boilerplate temizliği,
                        süresi dolmuş kampanya eleme, parmak izi ile
                        kopya eleme
```

### JS fallback iki kademelidir

1. **Ucuz ön eleme** (`HttpClient`): statik yanıtın ham metni çok kısaysa
   doğrudan tarayıcıya düşülür.
2. **Asıl tetikleyici** (`BaseScraper`): sayfa 200 dönse ve HTML dolu görünse
   bile, **boilerplate temizliğinden sonra** kayda değer metin kalmadıysa
   içerik istemci tarafında yükleniyor demektir; sayfa Playwright ile yeniden
   alınır. Ham HTML uzunluğuna bakan bir kontrol bu durumu yakalayamaz.

### Boilerplate temizliği neden üç kademeli?

Tek bir "şu class'ları sil" listesi güvenilir değil. Gerçek ölçümle
karşılaşılan üç ayrı tuzak vardı:

| Tuzak | Gözlendiği yer | Çözüm |
|---|---|---|
| Tüm sayfa `<form>` içinde | Türkiye Finans (SharePoint) | **Sarmalayıcı koruması:** silinmesi sayfa metninin %50'sinden fazlasını götüren blok "çerçeve" değil "içerik"tir, korunur |
| İçerik `<nav>`/`<header>` içine gömülü | Emlak Katılım | Aynı koruma |
| İçerik `class="... search-content"` içinde | Kuveyt Türk | **Link yoğunluğu testi:** gerçek menü metninin çoğunu bağlantı olarak taşır; düz paragraf taşıyan blok, adı ne olursa olsun içeriktir |

Bu iki koruma bilinçli olarak **temkinli** çalışır: şüpheli durumda blok
korunur. İçeri sızan menü/footer kırıntıları ise **sayfalar arası** temizlikle
atılır — aynı bankanın sayfalarının %60'ından fazlasında geçen kısa satırlar
şablon kabul edilip silinir (`drop_repeated_lines`). Fazladan menü satırı
taşımak sonradan düzeltilebilir bir hata; sayfanın tek içeriğini silmek değil.

---

## 5. Şartname Uyumu

| Şartname maddesi | Bu moduldeki karşılığı |
|---|---|
| **5.1 Veri Toplama** — BDDK listesindeki kuruluşların tümü | 10 bankanın tamamı `config/banks.yaml` içinde; adresler BDDK sayfasından **doğrulanarak** alındı (bkz. §6) |
| **5.1 Veri Toplama** — veri hacmi | Canlı toplamaya ek olarak `--arsiv` modu, kaldırılmış kampanya sayfalarını web arşivinden getirir (bkz. §3.1); korpus birkaç katına çıkar |
| **5.8 Veri Ön İşleme** | `text_cleaner.py`: boilerplate temizliği, Unicode/boşluk normalizasyonu, kopya eleme |
| **5.9 On-Premise Uygulanabilirlik** | Harici servis, API anahtarı veya bulut bağımlılığı **yok**. Tümü yerelde çalışır; dış erişim yalnızca hedef banka siteleri ve (arşiv modunda) ücretsiz/anahtarsız Internet Archive CDX API'sidir. Her ikisi de tek seferlik toplama aşamasındadır; üretim sistemi çevrimdışı çalışır |
| **5.10 Açık Kaynak Kod Yaklaşımı** | Yalnızca izin verici lisanslı paketler: requests (Apache-2.0), BeautifulSoup4 (MIT), lxml (BSD-3), PyYAML (MIT), pandas (BSD-3), Playwright (Apache-2.0). Ücretli/lisans riskli bağımlılık yok |
| **6. Kurulum adımlarının net belirtilmesi** | Bu dosyanın §1 bölümü |
| **7. Kod yapısının modüler ve okunabilir olması** | Her banka için ayrı sınıf; ağ/temizlik/depolama katmanları ayrık; konfigürasyon koddan ayrı |

### Etik ve yasal tarama

- `robots.txt` her host için indirilir, doğrulanır ve **uygulanır**. Engellenen
  URL indirilmez, özet raporda sayılır.
- `Crawl-delay` varsa ona uyulur; yoksa varsayılan bekleme (2 sn + jitter).
- Bot kendini `User-Agent` içinde tanıtır (iletişim adresi ile birlikte).
- Bankalar **sırayla** taranır, paralel değil — tek bir siteye eşzamanlı yük
  bindirilmez.
- 5xx dönen bir host'ta robots.txt okunamazsa, temkinli davranılıp o host
  tamamen atlanır.

> `respect_robots: false` yapılabilir ancak bu **şartname ihlalidir**;
> ayar yalnızca teşhis amacıyla vardır.

---

## 6. Banka Yapılandırması ve Saha Notları

Adresler ve site davranışları **2026-08-14 tarihinde tek tek ölçülerek**
doğrulanmıştır. BDDK listesindeki 10 kuruluşun tamamı "faaliyette" durumundadır.

| Banka (`key`) | Kullanılan adres | Keşif | Not |
|---|---|---|---|
| `adil_katilim` | adilkatilim.com.tr | Playwright + tarama | SPA; `sitemap.xml` XML değil HTML dönüyor. **Banka henüz ürün/kampanya yayınında değil** — site bir "yakında" sayfası |
| `albaraka` | **albaraka.com.tr** | sitemap | BDDK'daki `albarakaturk.com.tr` buraya yönleniyor. Sitemap'in ilk kaydı bozuk (`<loc>url</loc>`) |
| `dunya_katilim` | **dunyakatilim.com.tr** (www'suz) | sitemap | `www` → 308 yönlendirme. robots.txt sitemap'i üçüncü bir host'ta gösteriyor; oraya **gidilmiyor** |
| `hayat_finans` | hayatfinans.com.tr | sitemap | Sitemap `www`'da, içindeki adresler www'suz |
| `kuveyt_turk` | kuveytturk.com.tr | sitemap | ~3500 URL'lik büyük sitemap; kampanya arşivi hariç tutuldu |
| `tom_bank` | tombank.com.tr | link taraması | robots.txt ve sitemap.xml **yok** (404). Düz `.html` sayfalar |
| `emlak_katilim` | **emlakkatilim.com.tr** | sitemap | BDDK'daki `emlakbank.com.tr` buraya yönleniyor. TR alt sitemap'i doğrudan kullanılıyor |
| `turkiye_finans` | turkiyefinans.com.tr | tarama (sitemap fallback) | SharePoint. `sitemap.xml` bir index ama alt sitemap'leri XML yerine HTML dönüyor → otomatik olarak taramaya düşülür |
| `vakif_katilim` | vakifkatilim.com.tr | **tarama önce**, sitemap yedek | `sitemap.xml` 404 (doğrusu `sitemap-tr.xml`). Sitemap'teki 300+ kampanyanın neredeyse tamamı süresi dolmuş; canlı kampanyalar yalnızca liste sayfasında |
| `ziraat_katilim` | ziraatkatilim.com.tr | sitemap | Drupal; sitemapindex `?page=1..8` |

### Yeni banka eklemek / bir site değişirse

1. `config/banks.yaml` içine yeni bir kayıt ekleyin (`key`, `ad`, `base_url`,
   `scraper`, `sitemap_urls`, `seed_paths`).
2. `scrapers/<yeni_banka>.py` içinde `BaseScraper`'dan türeyen bir sınıf yazın.
   Çoğu durumda birkaç satır yeterlidir; gerekirse şu kancalar ezilir:
   `CONTENT_SELECTORS`, `should_collect()`, `PREFER_CRAWL`,
   `REQUIRE_INCLUDE_MATCH`, `_default_sitemap_urls()`, `post_process_text()`.
3. `main.py`'ye **dokunmanız gerekmez**; sınıf, YAML'daki `scraper` alanından
   çalışma anında yüklenir.

Site yapısı değişip bir banka boş dönerse önce şunu çalıştırın:

```bash
python main.py --dry-run --banks <key> --log-level DEBUG
```

---

## 7. Ayarlar (`config/banks.yaml` → `defaults`)

| Ayar | Varsayılan | Açıklama |
|---|---|---|
| `request_delay_seconds` | 2.0 | Aynı host'a iki istek arası asgari bekleme |
| `delay_jitter_seconds` | 1.0 | Rastgele ek bekleme (0–1 sn) |
| `max_retries` | 3 | Geçici hatalarda deneme sayısı |
| `backoff_factor` | 2.0 | Üstel geri çekilme çarpanı |
| `timeout_seconds` | 30 | İstek zaman aşımı |
| `respect_robots` | true | robots.txt uygulansın mı (**kapatmayın**) |
| `max_pages_per_bank` | 120 | Banka başına azami **kayıt** sayısı |
| `max_crawl_depth` | 2 | Link taramasının derinliği |
| `min_text_length` | 200 | Bu uzunluğun altındaki sayfalar kaydedilmez |
| `allow_playwright` | true | JS fallback açık mı |
| `js_fallback_min_chars` | 400 | Statik yanıt için ucuz ön eleme eşiği |
| `global_include_patterns` | — | URL bu kalıplardan birine uymalı |
| `global_exclude_patterns` | — | Bu kalıplara uyan URL hiç toplanmaz |
| `language_exclude_patterns` | — | Türkçe dışı sayfaları eler |
| `arsiv.baslangic_yili` | 2016 | Arşiv keşfinde en eski yakalama yılı |
| `arsiv.url_basina_azami_yakalama` | 2 | Aynı sayfanın kaç sürümü alınsın |
| `arsiv.azami_kayit` | 400 | Banka başına azami arşiv kaydı |
| `arsiv.cdx_azami_satir` | 20000 | Tek CDX sorgusundan alınacak azami satır |
| `arsiv.eszamanli_istek` | 4 | Arşiv host'una eşzamanlı istek (banka sitelerini etkilemez) |
| `arsiv.url_filtresi` | — | CDX'e gönderilen sunucu taraflı kaba filtre |

> **Kâr payı / ücret sayfaları özel olarak hedeflenir.** NLP aşamasının en
> kritik alanı olan kâr payı oranı kampanya metinlerinde nadiren geçiyor
> (ölçüldü: kayıtların yalnızca %12,7'sinde). Oranlar ve tahsis ücreti /
> dosya masrafı bilgileri "Kâr Payı Oranları" ve "Ürün ve Hizmet Ücretleri"
> sayfalarında tablo halinde duruyor. Bu yüzden `kar-payi`, `oranlari`,
> `urun[-_a-z]*ucret` gibi kalıplar include listesinde ve `prioritize()`
> içinde en yüksek önceliğe sahip — sayfa kotasına asla takılmazlar.
> Kalıplar bilerek dardır: bare `ucret` yazıldığında "ücretsiz otopark"
> gibi onlarca ayrıcalık sayfası da eşleşip gerçek tabloları kotadan atıyor.

Bankaya özel `extra_include_patterns`, `extra_exclude_patterns` ve
`extra_allowed_hosts` alanları global listelerin **üzerine eklenir**.

---

## 8. Sorun Giderme

| Belirti | Neden / Çözüm |
|---|---|
| `Playwright kurulu degil` uyarısı | `pip install playwright && playwright install chromium` |
| `Playwright baslatilamadi` | Paket var ama tarayıcı yok: `playwright install chromium` |
| Bir banka `[ATLA]` | JS gerektiriyor ve Playwright kullanılamıyor (yukarı bakınız) |
| Bir banka `[HATA]`, "Hicbir aday URL bulunamadi" | Site yapısı değişmiş. `--dry-run --log-level DEBUG` ile keşfi inceleyin, `banks.yaml`'daki `sitemap_urls`/`seed_paths`'i güncelleyin |
| Çok sayıda "içeriksiz/kısa" eleme | Sayfalar JS ile yükleniyor olabilir; Playwright'ın kurulu olduğundan emin olun |
| "süresi dolmuş kampanya" elemesi yüksek | Normal — sitemap arşiv kayıtları listeliyor. İlgili banka için `PREFER_CRAWL = True` düşünülebilir |
| `robots.txt tarafindan engellendi` | Beklenen davranış; o URL bilinçli olarak toplanmaz |
| Kurumsal proxy arkasında TLS hatası | `banks.yaml` → `verify_tls: false` (yalnızca kurum içi güvenli ağda) |
| Türkçe karakterler CSV'de bozuk | CSV `utf-8-sig` yazılır; Excel'de "Veri → Metinden" ile UTF-8 seçin |

Çıkış kodları: `0` başarılı, `1` hiç kayıt üretilmedi, `2` konfigürasyon hatası,
`130` kullanıcı iptali.

---

## 9. Sonraki Aşamaya Devir

NLP/bilgi çıkarım aşaması için önerilen giriş noktası:

```python
import pandas as pd

# Canlı + arşiv birleşik korpus (önce: python main.py --birlestir)
df = pd.read_json("data/processed/katilim_bankalari_tum.jsonl", lines=True)

# Yalnızca kampanya sayfaları
kampanyalar = df[df["kampanya_mi"] == True]

# Kategoriye göre
finansman = df[df["kategori_tahmini"] == "finansman"]

# MODEL EĞİTİMİ: arşiv kayıtları dâhil (süresi dolmuş kampanya da geçerli
# bir etiketleme örneğidir)
egitim = df

# KULLANICIYA SUNUM (dashboard / chatbot): yalnızca yürürlükteki içerik
guncel = df[df["guncel_mi"].fillna(True)]
```

Bu aşamada yapılacaklar (bu modülün **kapsamı dışında**): kâr payı oranı, vade,
kampanya avantajı, masraf ve hedef kitle bilgilerinin `ham_metin` üzerinden
çıkarılması; kampanya türü sınıflandırması; bankalar arası karşılaştırma;
dashboard ve chatbot.

> **Bölme (train/val/test) uyarısı.** Arşiv korpusunda aynı kampanyanın
> birden çok sürümü bulunur. `dataset_builder/utils/splits.py` içindeki
> `kume_anahtari` yakın kopyaları içerik imzasına göre aynı kovaya
> düşürdüğü için bu sürümler aynı bölüme gider; bölmeyi doğrudan kayıt
> düzeyinde yapan yeni bir kod yazılırsa bu sızıntı riski geri gelir.
