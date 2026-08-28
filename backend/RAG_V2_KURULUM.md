# HititFinLex RAG V2 kurulum ve isletim kilavuzu

RAG V2, eski `/search`, `/chat` ve `/history/*` uclarini kaldirmadan yeni
`/rag/v2/*` yuzeyini ekler. Calisma zamani Docker gerektirmez; Windows ve
PostgreSQL 18 uzerinde dogrudan calisir.

## Akis

1. Sunucu tarafindaki oturum durumu, son kullanici/asistan turlari ve sinirli
   konusma ozeti takip sorusunu bagimsiz sorguya donusturur. Onceki asistan
   metni konusma baglamidir; finansal kanit degildir. EVREN JSON-schema
   yonlendirmesi basarisiz olursa deterministik cozumleyici kullanilir.
2. `bge-m3-embed` ile 1024 boyutlu EVREN embedding uretilir ve takim-izole
   Qdrant koleksiyonunda yogun arama yapilir.
3. Ayni sorgu PostgreSQL `simple + unaccent` agirlikli `tsvector` indeksinde
   guvenli OR/prefix lexical aramaya girer.
4. Sonuclar `dense_weight / (rrf_k + dense_rank) + lexical_weight /
   (rrf_k + lexical_rank)` ile birlestirilir. Urun yalnizca yumusak sinyaldir;
   banka, kapsam, tarih ve teklif kimligi sert filtredir.
5. Teklif ve belge tekillestirmesi ile karsilastirmalarda banka cesitliligi
   uygulanir. Siniflandirma statusu, confidence ve conflict degerleri denetim
   metadata'si olarak korunur; varsayilan akista kanit kullanimini engellemez.
6. EVREN `llm-fast`, yalnizca secilmis kanit kayitlarini veri olarak gorur.
   Her olgusal cumle `[S1]` biciminde kaynaklanir.
7. Bilinmeyen kaynak, kaynaksiz iddia, yanlis birimli/desteksiz sayi veya
   teklifler arasi baglam birlestirme bulunursa cevap fail-closed engellenir.
8. Finans disi sohbet `chat` intent'iyle aramayi atlar ve EVREN tarafindan
   `conversational` durumunda yanitlanir. Banka teklifi, oran, tutar veya vade
   sorulari bu dala gecemez ve her turda yeniden kanit arar.

Banka belirtilmeyen liste/karsilastirma sorgularinda kanit kaydi bulunan
bankalar cesitlendirilir ve her banka icin en iyi kanit ayri satirda verilir.
Kaynagi olmayan banka icin sayi veya kosul uretilmez. Normal kanit limiti `6`,
banka-kapsamli sorgu limiti `12` olarak ayri yapilandirilabilir.

Reranker varsayilan olarak kapali ve ayrica `RAG_V2_RERANKER_VALIDATED=true`
olmadan etkinlesmez.

## Ortam degiskenleri

`.env.example` dosyasini yerel ve Git tarafindan yok sayilan `.env` dosyasina
kopyalayin. Gercek anahtarlari yalniz bu dosyada veya bir secret manager'da
tutun. Asgari RAG V2 degiskenleri:

```env
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=katilim_finans
DB_USER=hititfinlex_app
DB_PASSWORD=<runtime-role-password>

EVREN_API_KEY=<team-key>
EVREN_BASE_URL=https://evren-llmapi.ssyz.org.tr/v1
EVREN_TEXT_MODEL=llm-fast
EVREN_EMBEDDING_MODEL=bge-m3-embed
EVREN_EMBEDDING_DIMENSION=1024

QDRANT_URL=<team-isolated-qdrant-url>
QDRANT_API_KEY=<team-qdrant-key>
QDRANT_COLLECTION=hititfinlex_rag_v2
```

RRF, siniflandirma, timeout, havuz, kanit ve oturum ayarlarinin tamami
`.env.example` icinde belgelenmistir. Baslangic degerleri yogun `1.0`, lexical
`0.5`, `rrf_k=60`, normal kabul `0.80`, inceleme alt siniri `0.65`, sliding
oturum TTL `2592000` saniye, son gecmis `6` tur, prompt gecmis siniri `16000`,
konusma ozeti siniri `32000` karakter ve UI transcript siniri `100` turdur.

`RAG_V2_ENFORCE_CLASSIFICATION_POLICY=false` varsayilaniyla `accepted`,
`review`, `required`, `verified`, confidence ve conflict degerleri retrieval,
cevap kaniti veya session baglamini elemez. Veri ve ona ait kanit metni varsa
kullanilir. Eski esik davranisi gerekirse bu degisken `true` yapilarak yeniden
acilabilir. `RAG_V2_REQUIRE_TEXTUAL_PRODUCT_CONFIRMATION=false` ise
yapilandirilmis urun+alan eslesmesini, baslikta urun ifadesi gecmedigi icin
reddetmez. Her iki ayar da banka/tarih/kapsam sert filtrelerini, kaynak
atiflarini, sayi kontrolunu ve offer izolasyonunu degistirmez.

## PostgreSQL migration

API runtime rolu schema DDL yetkisine sahip olmamalidir. `0003_rag_v2.sql` ve
`0004_rag_v2_conversation.sql` migration'larini schema sahibi/migrator roluyla
uygulayin. `0004`, kanitsiz sohbet yanitlari icin ayri `conversational` mesaj
durumunu ekler. Docker gerekli degildir:

```powershell
$env:DATABASE_URL = "postgresql://<migrator>:<password>@127.0.0.1:5432/katilim_finans"
.\.venv\Scripts\python.exe db\migrate.py check
.\.venv\Scripts\python.exe db\migrate.py status
.\.venv\Scripts\python.exe db\migrate.py up
.\.venv\Scripts\python.exe db\migrate.py smoke
Remove-Item Env:DATABASE_URL
```

Migration idempotent tablolari, `unaccent`, agirlikli `tsvector`, GIN
indeksleri, guven/tarih/teklif indeksleri ve su oturum tablolarini olusturur:

- `rag_sessions`: SHA-256 token/owner ozeti, TTL ve revoke zamani;
- `rag_messages`: kullanici/asistan turlari ve route ozeti;
- `rag_session_state`: surum kontrollu yapilandirilmis durum;
- `rag_turn_evidence`: turda kullanilan kanit kayitlari.

Mesajlar temizlenene, yeni oturum acilana veya TTL dolana kadar PostgreSQL'de
saklanir. Modele son `RAG_V2_HISTORY_TURNS` tur ayrintili olarak, daha eski
baglam ise sinirli `conversation_summary` alaniyla verilir. Her ikisi de prompt
icinde guvenilmeyen konusma verisi olarak etiketlenir. Dogrulanmis finansal bir
tur structured state'i yenilerken rolling ozeti veya tur sayisini sifirlamaz.
Kanitsiz/rejected finansal tur da acik banka ve urun sorgu baglamini korur,
ancak eski ve uyumsuz offer/source kimliklerini yeni konuya tasimaz.

Daha once kaynaklandirilan kanitlar takip turlarinda session kaynak paketi
olarak yeniden kullanilabilir. Sistem eski JSON snapshot'ini dogrudan modele
vermez; cited chunk kimligini guncel `rag_chunks` kaydindan tekrar yukler ve
banka, urun, offer, kapsam ve tarih filtrelerini yeniden uygular. Fresh retrieval
her turda calismaya devam eder. Varsayilan tek-tur kanit limiti 12 kayittir.

## Indeksleme

Once veritabani-yazmasiz dry-run yapin:

```powershell
.\.venv\Scripts\python.exe -m rag_v2.indexer --scope all --dry-run
```

Migration uygulandiktan ve EVREN/Qdrant anahtarlari ayarlandiktan sonra:

```powershell
.\.venv\Scripts\python.exe -m rag_v2.indexer --scope all
```

`--skip-remote`, yalniz PostgreSQL kayitlarini gunceller. Uretimde dense
arama icin uzak Qdrant upsert'i de tamamlanmalidir. Indeksleyici embedding
boyutunu ilk EVREN cevabindan dogrular, HTTP baglantilarini ve batch'leri
yeniden kullanir, sonra gerekli Qdrant payload indekslerini olusturur.

Bir belgede birden fazla kampanya donemi guvenle ayrilamiyorsa belge
silinmez; `classification_conflict=true`, `review` olarak saklanir. Bu alanlar
denetim icin korunur, ancak varsayilan classification policy kapaliyken belgeyi
otomatik sayisal kullanimdan cekmez. Kampanya donemi, kaynak cumlesi, tarih ve
offer ayrimi kontrolleri yine zorunludur.

Indeksleyici acik Worldpuan/Altin Puan tutarlarini ve kampanya gecerlilik
cumlelerini kanit metniyle birlikte chunk payload'ina baglar. `aktif` kampanya
sorgulari uygulama gununu `date_from=date_to` olarak yollar. PostgreSQL ve
Qdrant ayni acik-aralik kuralini uygular; iki tarih siniri da bilinmeyen belge
aktif sonucuna alinmaz. Bu kurallar degistiginde `--scope current` komutu
`--skip-remote` olmadan yeniden calistirilarak Qdrant payload'i da guncellenir.

## API ve oturum guvenligi

Ilk soru `session_id: null` ile gonderilebilir:

```json
{
  "session_id": null,
  "query": "Ziraat Katilim konut finansmaninda en yuksek tutar nedir?",
  "top_k": 12,
  "use_reranker": false
}
```

`POST /rag/v2/chat` yanitinda uretilen tahmin edilemez `session_id`, sonraki
turlarda ayni JSON alaniyla gonderilir. Tarayici ayrica en az 16 karakterlik
rastgele bir `X-RAG-Client-Id` yollar; backend bunun yalniz SHA-256 ozetini
saklar. Yeni istemciler oturum token'ini URL loglarina sokmayan uclari
kullanmalidir:

- `POST /rag/v2/sessions`: yeni oturum;
- `GET /rag/v2/session`: `X-RAG-Session-Id` basligi ile durum;
- `GET /rag/v2/session/messages`: ayni basliklarla owner-korumali transcript;
- `POST /rag/v2/session/clear`: ayni baslikla mesaj ve baglami temizleme;
- `DELETE /rag/v2/session`: ayni baslikla revoke.

Eski path-parametreli session uclari geriye uyumluluk icin korunur; yeni
frontend bunlari kullanmaz. Uygulamada tam kimlik dogrulama bulunmadigindan
session ve client degerleri bearer sirri gibi korunmalidir. Canonical frontend
ikisini `localStorage` icinde birlikte saklar, eski `sessionStorage` ciftini bir
kez migrate eder ve sayfa acilisinda transcript'i header tabanli uctan yukler.
Bu katman kullanici hesabi yetkilendirmesinin yerini tutmaz.

## Test ve degerlendirme

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py"
.\.venv\Scripts\python.exe -m evaluation.router_compare --output evaluation\router-comparison.local.json
.\.venv\Scripts\python.exe -m evaluation.retrieval_compare --validate-only
.\.venv\Scripts\python.exe -m evaluation.retrieval_compare --output evaluation\retrieval-comparison.local.json
```

Degerlendirme dosyalari `silver_unverified` etiketlidir. Canli PostgreSQL
`rag_chunks`, EVREN, Qdrant veya iki tarafta da ayni etiketli cikti eksikse
runner skor uydurmaz ve ilgili metrigi `unavailable` raporlar. Onceki sistem
ve V2 retrieval metrikleri yalniz ayni 32 kaydin ikisinde de eksiksiz
calistigi kosuda karsilastirilir.

Frontend dogrulamasi canonical `HititFinLex` klasorunde:

```powershell
npm.cmd run test:smoke
npm.cmd run typecheck
npm.cmd run lint
npm.cmd run build
```

PowerShell execution policy `npm.ps1` dosyasini engelliyorsa `npm.cmd`
kullanin.
