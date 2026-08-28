# HititFinLex RAG V2 sonuc raporu

Tarih: 2026-08-28  
Degerlendirme etiketi: `silver_unverified`

Bu rapor, ayni test setinde gercekten olculen sonuclari ve 2026-08-28
production aktivasyon sonucunu birlikte kaydeder. Gold kalite veya basari
artisi iddiasi yapilmaz.

## Incelenen mevcut mimari

- FastAPI ana uygulamasi `api.py`; mevcut `/search`, `/chat`, `/history/*`,
  intake ve review yuzeyleri korunur.
- PostgreSQL 18.6 ve pgvector 0.8.6 kullanilir. Mevcut corpus 771 guncel ve
  2.580 tarihsel belge, guncel `document_chunks.embedding` alani ise
  `vector(1024)` boyutundadir.
- Legacy retrieval, yerel BGE-M3 embedding ile PostgreSQL dense/lexical
  aramasini `hybrid_search.py` uzerinden calistirir.
- RAG V2, EVREN `bge-m3-embed`, takim-izole Qdrant, PostgreSQL full-text
  search ve EVREN `llm-fast` kullanir.
- Aktif frontend ayri `../HititFinLex/` repository'sindeki Next.js 16.3.3 /
  Vinext uygulamasidir; session token ve client owner kimligi yalniz tarayici
  `localStorage` alaninda cift olarak tutulur. Eski `sessionStorage` cifti tek
  seferlik guvenli migration ile tasinir; owner bilgisi olmayan session kabul
  edilmez. Sayfa acilisinda server-side transcript yeniden yuklenir.

## Uygulanan RAG V2 akisi

1. Server-side session state, son 6 kullanici/asistan turu ve sinirli rolling
   konusma ozeti yuklenir. Onceki asistan cevabi EVREN'e baglam olarak verilir,
   ancak finansal kanit veya sistem talimati sayilmaz.
2. EVREN kati JSON-schema router denenir; protokol veya servis hatasinda
   deterministik router ayni semayi uretir.
3. Yeni acik banka, urun, yil, tarih veya konu eski uyumsuz state'i temizler;
   yalniz anlamli eksik alanlar devralinir.
4. Sorgu EVREN BGE-M3 ile 1024 boyutlu embed edilir ve Qdrant'ta dense
   aranir. Banka, kapsam, tarih ve teklif kimligi sert filtredir.
5. PostgreSQL `simple + unaccent` agirlikli `tsvector` ve GIN indexi uzerinde
   parametreli lexical arama yapilir.
6. Sonuclar `dense_weight/(rrf_k+dense_rank) +
   lexical_weight/(rrf_k+lexical_rank)` ile birlestirilir. Varsayilanlar
   `1.0`, `0.5` ve `60` olup environment ile degistirilebilir.
7. Urun sinifi yalniz yumusak sinyaldir. Offer ve document tekillestirmesi,
   banka belirtilmeyen lookup/list/compare sorgularinda banka cesitliligi ve
   Python ile deterministik sayisal siralama uygulanir. Uygun kaydi bulunan
   her banka ayri satirda ozetlenir; kampanyalar ayni offer gibi birlestirilmez.
8. Siniflandirma statusu, confidence ve conflict alanlari denetim metadata'si
   olarak korunur. Varsayilan `RAG_V2_ENFORCE_CLASSIFICATION_POLICY=false`
   ayarinda bu alanlar retrieval, kanit veya session baglamini elemez; kayitta
   veri ve ona ait kanit metni varsa kullanilir. Eski esik politikasi environment
   ile istege bagli yeniden acilabilir.
9. Sayisal liste ve karsilastirmalar normalize birimlerle Python tarafinda
   siralanir ve tek-offer/tek-kaynak maddeleri deterministik uretilir. Diger
   cevaplarda EVREN `llm-fast` yalniz secilmis kanit paketini veri olarak gorur.
10. Bilinmeyen kaynak, kaynaksiz olgusal cumle, desteklenmeyen sayi/birim
    veya teklif baglami karismasi cevabi fail-closed engeller.
11. Finans disi istekler deterministik `chat` intent'iyle retrieval'i atlar;
    EVREN konusma hafizasiyla `conversational` cevap verir. Banka verisi
    isteyen finansal turlar bu dala gecemez ve her turda yeniden aranir.
12. Dogrulanmis finansal turdan sonra yeni state uretilirken rolling ozet ve
    tur sayisi korunur. `Devam et`, `daha ayrintili anlat` ve `ozetle` gibi
    dogal takipler aktif finansal state varsa yeniden retrieval'a yonlendirilir.
    Ilk tur yetersiz olsa bile acik banka/urun baglami tutulur; buna karsilik
    banka, urun veya donem degisiminde eski offer/source baglari temizlenir.
13. Secilen kanitin baslik/bolum urunu acik sorgu urunuyle celisiyorsa sayisal
    cevapta kullanilmaz. Boylece tasit sorusunun konut kampanyasi tutariyla
    cevaplanmasi ve yanlis offer'in sonraki turlari zehirlemesi engellenir.
14. Onceki `verified` turlarda gercekten cite edilen kanit chunk'lari session
    kaynak paketi olarak saklanir. Takip turunda eski JSON kullanilmaz; chunk
    guncel PostgreSQL kaydindan yeniden hydrate edilir ve banka, urun, offer,
    kapsam ve tarih hard filtrelerinden sonra fresh retrieval'a eklenir. Kaynak
    kimlikleri her yeni turda yeniden `[S1]...[Sn]` olarak numaralanir.

Reranker varsayilan olarak kapali kalir ve hem enable hem validation bayragi
olmadan kullanilamaz.

## Parcalama, teklif kimligi ve indeks

- Baslik ve bolum bilgisini koruyan, overlap destekli, navigation ayristiran
  kararlı chunking uygulandi.
- `chunk_id` ve `offer_id` SHA-256 tabanli ve tekrar calistirmada kararlidir.
- Teklif kimligi banka, urun, canonical URL, baslik, fact/content siniri ve
  kampanya donemini kapsar. Ayni URL'deki ayri kampanyalar birlesmez.
- Tutar, oran, vade, ucret, odul, harcama esigi ve kampanya tarih fact'leri
  yalniz ait olduklari chunk ile tasinir.
- Yilsiz Aralik-Ocak araliklari snapshot ile guvenle cozulur. Acikca ters
  veya belirsiz tarih araligi DB kisiti gevsetilmeden `review/conflict` olur.
- Tam indeks: 3.351 belge, 12.380 chunk, 4.363 bagli fact, 0 stale nokta.
- Guncel kaynaklarda acik Worldpuan/Altin Puan ve kampanya tarih ifadeleri
  kaynak cumlesiyle birlikte yuksek kesinlikli fact olarak indekslenir. Aktif
  sorgusu iki ucu bilinmeyen kaydi almaz; tek ucu acik tarihleri acik aralik
  semantigiyle PostgreSQL ve Qdrant'ta ayni bicimde filtreler.
- Qdrant: `green`, 12.380 nokta, 13 payload indexi.

## Cok turlu hafiza ve session semasi

- `rag_sessions`: rastgele session token'inin SHA-256 ozeti, client owner
  ozeti, TTL, son erisim ve revoke zamani.
- `rag_messages`: user/assistant turlari, route, devralinan baglam ve durum.
- `rag_session_state`: surum kontrollu bankalar, urunler, kapsam, yil/tarih,
  offer sirasi, son alanlar, dogrulanmis kaynak/document kimlikleri,
  `broad_bank_context`, rolling konusma ozeti ve tur sayisi.
- `rag_turn_evidence`: her turun kanit snapshot'i.

Token URL'ye konmadan yeni uclarda `X-RAG-Session-Id`, client sahipligi icin
`X-RAG-Client-Id` kullanilir. Varsayilan kayar TTL 2.592.000 saniyedir (30 gun);
model baglami son 6 kullanici/asistan turu, 32.000 karakterlik sinirli
deterministik rolling ozet ve kalici yapilandirilmis state'i kapsar. Oturum
transcript'i varsayilan en fazla 100 tur olarak owner/session kontroluyle
`GET /rag/v2/session/messages` ucundan okunur. Onceki asistan cevaplari EVREN'e
yalniz `untrusted conversation data` olarak verilir; finansal gercek veya
talimat kabul edilmez. Her finansal takip turunda yeniden retrieval ve
dogrulama yapilir. Bankasiz genis sorgular state'te teklif filtresi olarak
daraltilmaz; boylece sonraki `peki tutari?` benzeri soru tekrar tum bankalari
tarar. Expired, owner mismatch, clear ve revoke yollari fail-closed test
edilmistir. Uygulamada hesap kimlik dogrulamasi olmadigi icin bu bearer session
modeli tam user authorization yerine gecmez.

## Migration sonucu

Eklenen idempotent migration seti:

- `db/migrations/0001_base.sql`
- `db/migrations/0002_review_document_baseline.sql`
- `db/migrations/0003_rag_v2.sql`
- `db/migrations/0004_rag_v2_conversation.sql`
- `db/migrations/manifest.json`

Temiz PostgreSQL 18 kurulumunda 0001-0003 uygulandi; tekrar uygulama,
checksum, pgvector, unaccent, `vector(1024)`, trigger, GIN/B-tree indexleri,
session tablolari ve least-privilege siniri gecti. Mevcut verinin izole
kopyasinda 0001/0002 uzerine 0003 upgrade'i de gecti.

Gercek production DB'de `0003` migrator roluyle uygulandi. Migration status
0001-0003 icin `applied`, PostgreSQL 18/pgvector/unaccent/index/trigger smoke
sonucu basarilidir. Production indexleme 3.351 belge, 12.380 chunk ve 4.363
fact ile tamamlandi; stale kayit sayisi sifirdir. Runtime rolunun CRUD yetkisi
dogrulandi ve schema DDL yetkisinin bulunmadigi yeniden test edildi.

`0004`, konusma mesajlarina `conversational` durumunu ekleyen idempotent check
constraint migration'idir ve manifest checksum/testleri gecmistir. Canli DB'de
bu migration henuz uygulanamamistir: eldeki `hititfinlex_app` runtime rolunun
tablo sahipligi/DDL yetkisi yoktur, tablo sahibi `hititfinlex_migrator` icin ise
bu calisma ortaminda parola veya migration URL'i bulunmamaktadir. Uygulama bu
durumu kontrollu algilar; migration uygulanana kadar sohbet mesajini `NULL` DB
status ile saklar, API'de yine `conversational` doner ve konusma hafizasi
calisir. DDL calistirabilecek migrator baglantisinda `python db/migrate.py up`
calistirilmasi kalan tek production schema adimidir.

## Ayni sette retrieval sonucu

32 exact-page `silver_unverified` vaka, reranker kapali:

| Metrik | Legacy | RAG V2 |
| --- | ---: | ---: |
| Recall@1 | 0.812500 | 0.625000 |
| Recall@3 | 0.937500 | 0.718750 |
| Recall@5 | 0.937500 | 0.750000 |
| Recall@10 | 1.000000 | 0.781250 |
| MRR@10 | 0.873698 | 0.677951 |
| nDCG@10 | 0.904173 | 0.702924 |

Bu kosu, classification status/confidence/conflict filtreleri varsayilan olarak
kapaliyken ayni 32 sirali vaka ve ayni corpus snapshot'i uzerinde iki sistemi
birlikte olcmustur. RAG V2 onceki policy-on raporuna gore daha fazla etiketli
sayfayi dondurmus olsa da bu ayri canli kosular arasinda nedensel iyilesme
iddiasi sayilmaz. Ana paired sonucta V2 halen legacy baseline'in altindadir;
basari artisi iddiasi yapilmaz.

## Takip sorusu sonucu

32 senaryo, 66 tur, `silver_unverified`:

| Metrik | Legacy | RAG V2 |
| --- | ---: | ---: |
| Kati standalone query exact | 0.090909 | 0.000000 |
| Standalone context coverage | 0.133333 | 0.983333 |
| Banka devralma | 0.000000 | 1.000000 |
| Urun devralma | 0.000000 | 1.000000 |
| Tarih devralma | 0.000000 | 1.000000 |
| Kapsam devralma | 0.000000 | 0.904762 |
| Tum devralma alanlari | 0.000000 | 0.928571 |
| Konu degisimi temizleme | 0.772727 | 0.909091 |
| Clarification | 0.909091 | 1.000000 |
| Session isolation | 1.000000 | 1.000000 |

Kati exact metriğin sifir olmasi saklanmamistir: V2 semantik olarak gerekli
baglami kapsasa da fixture cumlesini birebir uretmemistir.

Kaynak dogrulugu, sayisal dogruluk ve desteksiz cevap reddetme orani icin
66 turun eksiksiz, insan etiketli paired legacy/V2 answer ciktilari yoktur;
bu metrikler `unavailable` raporlanir. Validator davranislari unit ve
integration testlerinde gecmistir, ancak bunlar gold accuracy skoru diye
sunulmaz.

## Test ve build sonucu

- `python -m unittest discover -s tests -v`: 218 test; 216 passed,
  0 failed, 2 skipped. Skipler yalniz ayri rollback migration testleri icin
  `DATABASE_URL`/`RAG_V2_TEST_DATABASE_URL` ister; production PostgreSQL ile
  session, TTL, izolasyon, transcript ve lexical integration testleri calisip
  gecmistir.
- Ayri legacy/backend hardening suite: 30 passed, 0 failed.
- Birbirinden ayri backend toplam sonucu: 246 passed, 0 failed, 2 skipped.
- Intake/database/review smoke: 3/3 gecti, mutation yapilmadi.
- Canonical `../HititFinLex/` frontend testleri: 23 passed, 0 failed.
- Frontend lint: gecti, 0 error, 0 warning.
- Frontend local TypeScript typecheck: gecti.
- Frontend production build: Next.js 16.3.3 bagimliligi ve Vinext 1.0.0-beta.8
  ile gecti; bounded Vinext build'in 5/5 asamasi tamamlandi.
- Iki repository icin `git diff --check` hata vermedi; yalniz mevcut Windows
  CRLF donusum uyarilari goruldu.
- Backend source/runtime/Git ve frontend hedefli secret hygiene testleri gecti;
  anahtar veya parola bulgusu yoktur.

## Canli servis smoke sonucu

- EVREN model discovery: 10 model.
- `bge-m3-embed`: 1024 boyut dogrulandi.
- `llm-fast`: minimal ve 6 kanitli dogrudan cagri basarili.
- Qdrant: `green`, 12.380 nokta ve 13 payload indexi.
- Izole DB ile iki turlu service smoke her turda retrieval yapti, ikinci tur
  banka/urun/kapsam devraldi ve iki turda da 6 kanit buldu.
- Production DB migration ve tam indekslemeden sonra port 8000 backend prosesi
  kontrollu yeniden baslatildi. `/health=200`; model, NER ve classifier
  readiness alanlari `true`; OpenAPI'de 8 RAG V2 yolu kayitlidir.
- Gercek HTTP lifecycle smoke; session create/get, chat, clear, delete ve
  delete sonrasi 404 kontrollerinin tumunu gecti. Chat her turda retrieval
  yapti ve reranker kullanmadi.
- Finans disi bir selamlama `chat/conversational` dondu, retrieval yapmadi ve
  EVREN dogal cevap uretti. Onceki asistan tanitimina yapilan takip sorusunda
  `assistant_context_used=true` oldu; onceki cikti talimat veya finansal kanit
  olarak kullanilmadi.
- Dokuz turluk canli hafiza testinde ilk turdaki `Benim adim Ali` bilgisi, ilk
  tur son 6 turluk yakin pencereden cikmasina ragmen dokuzuncu turda dogru
  hatirlandi. `conversation_turn_count` 1'den 9'a monoton ilerledi ve transcript
  ucu 9 mesaji geri dondurdu. Bu test eski ozet-sifirlama hatasini dogrudan
  kapsar.
- Kaynak paketi aktivasyonu sonrasi canli iki turlu Ziraat Katilim testinde ilk
  cevap `verified` dondu. Takip turu fresh retrieval'i yeniden calistirdi,
  onceki cited chunk referansini buldu ve ayni chunk'i guncel PostgreSQL
  kaydindan basariyla hydrate etti. Fresh aramada ayni chunk zaten bulundugu icin
  cift kopya modele verilmedi; iki tur da `verified` dondu.
- `Devam et ve daha ayrintili anlat` takibi aktif konut finansmani state'inde
  `lookup` olarak kaldi ve `retrieval_performed=true` ile yeni kanit aradi;
  erken genel sohbet dalina dusmedi.
- `Vakif Katilim tasit finansmaninda tutar nedir?` sorgusu artik yalniz tasit
  kanitlarini dondurur ve `Kentsel Donusum` konut kampanyasindaki 3.000.000 TL
  tutari cevap veya session offer'i olarak kullanmaz. Mevcut tasit sayfalarinda
  guvenli tek bir azami finansman tutari bulunmadigi icin bu soru durust bicimde
  `insufficient_evidence` kalir; yanlis urunle cevap uretilmez. Banka ve tasit
  urun baglami sonraki takip icin yine state'e yazilir.
- Relevance duzeltmesinden sonra `Vakif Katilim guncel tasit finansmani
  vadesi kac ay?` sorusu tek urun+alan uyumlu kanitla `verified` dondu. Takip
  sorusu `Peki 400.000 TL ve altinda kac ay?` onceki banka/urun/offer
  baglamini devraldi ve 48 ay sonucunu `[S1]` ile dogruladi.
- `Turkiye Finans is yeri finansmaninda 60 ay vadede ekspertiz ucreti nedir?`
  sorusu 16.500 TL sonucuyla `verified` dondu.
- `Ziraat Katilim konut finansmaninda en yuksek tutar nedir?` sorusu `review`
  statuslu ve conflict metadata'li kaydi dislamadi; tek teklif kanitindan Python
  tarafinda 6.000.000 TL sonucunu `[S1]` ile `verified` dondu. Diagnostics'te
  classification policy kapali ve withheld kayit sayisi sifirdir.
- `Aktif kampanyalardaki alisveris puanlarini karsilastir` sorgusu bugunun
  kampanya araligi ve reward fact filtresiyle Kuveyt Turk icin 4.000 TL Altin
  Puan, Albaraka icin 1.250 TL Worldpuan olmak uzere iki banka satirini
  deterministik siralanmis `[S1]`/`[S2]` kanitlariyla `verified` dondu. Dort
  kanit icinde `accepted`, `required` ve `review` statuslari birlikte kullanildi;
  withheld kayit sayisi sifirdir.
- `Konut finansmaninda en uzun vadeli secenekleri karsilastir` sorgusu 11 kaniti
  yil/ay/gun icin ortak olcekte siraladi ve Turkiye Finans 120 ay, Ziraat
  Katilim Kentsel Donusum 10 yil, Vakif Katilim Kentsel Donusum 10 yil olmak
  uzere uc bankayi kaynakli `verified` dondurdu. Kanitlarda `accepted`,
  `required` ve `review` statuslari birlikte kullanildi; withheld kayit sayisi
  sifirdir.
- Banka belirtilmeden `Konut finansmani vadelerini getir` sorgusu bankalara
  gore kapsamli arama yapar. Kaniti ve ilgili yapilandirilmis fact'i bulunan her
  banka ayri satirda verilir; kaynagi olmayan banka icin deger uydurulmaz.
  `Peki tutarlari ne kadar?` gibi takiplerde urun baglami devralinir fakat tum
  bankalarda her tur yeniden retrieval yapilir.
- Context clear smoke 200 dondu ve state'teki banka/urun/ozet/tur sayisini
  sifirladi.
- Backend port 8000 ve canonical `../HititFinLex/scripts/start-production.mjs`
  production frontend port 3000 yeniden baslatildi; ikisi de HTTP 200 donuyor.
  Port 3000 listener'inin canonical start script'inden geldigi process
  komutuyla dogrulandi. Frontend bundle RAG V2 session protokolunu ve backend
  adresini kullaniyor.

## Degisen ve eklenen ana dosyalar

- Backend RAG V2: `rag_v2/api_router.py`, `chunking.py`, `database.py`,
  `evidence.py`, `identity.py`, `indexer.py`, `models.py`, `providers.py`,
  `retrieval.py`, `routing.py`, `service.py`, `sessions.py`, `settings.py`,
  `validation.py`.
- API/uyumluluk: `api.py`, `hybrid_search.py`, `api_security.py`,
  `requirements.txt`, `requirements-ci.txt`, `.env.example`.
- DB: `db/migrate.py`, `db/provision.py`, `db/runtime_schema.py`,
  `db/README.md`, `db/migrations/0004_rag_v2_conversation.sql` ve migration
  manifesti dahil migration seti.
- Degerlendirme: `evaluation/multiturn_scenarios.silver_unverified.json`,
  `retrieval_cases.silver_unverified.json`, `router_compare.py`,
  `retrieval_compare.py`, `rag_v2_metrics.py`, `evaluate_rag_v2.py`,
  `secret_hygiene.py`, `evaluation/README.md`.
- Testler: `tests/test_migrations.py`, `test_provision.py`,
  `test_rag_v2_core.py`, `test_rag_v2_evaluation.py`,
  `test_rag_v2_indexing.py`, `test_rag_v2_relevance.py`,
  `test_rag_v2_routing_regression.py`, `test_rag_v2_sessions.py`,
  `test_retrieval_compare.py`, `test_router_compare.py`,
  `test_secret_hygiene.py`, `test_backend_hardening.py`.
- Dokumantasyon: `README.md`, `RAG_V2_KURULUM.md`, bu rapor.
- Kanonik frontend: `../HititFinLex/app/rag-v2.ts`,
  `../HititFinLex/app/page.tsx`, `../HititFinLex/app/globals.css` ve
  `../HititFinLex/tests/rag-v2.test.mjs`,
  `../HititFinLex/tests/frontend-safety.test.mjs`. Tasarim/responsive iskelet
  degistirilmedi; session cifti kalici saklamaya alindi, server-side transcript
  hydration eklendi ve yeni/temizle davranislari korundu.

## Kalan teknik riskler

1. `silver_unverified` set gold degildir. Varsayilan akista `accepted`, `review`,
   `required`, `verified`, confidence ve conflict metadata'si kanit kullanimini
   engellemez. Bu tercih kapsami artirir; ancak cevabin kaynak metnine sadik
   oldugunu dogrulamak, kaynak kaydinin gercek dunyada hatasiz oldugunu garanti
   etmez. Eski siniflandirma esikleri gerektiğinde environment ile acilabilir.
2. EVREN ortak altyapi oldugu icin gecici answer provider hatalari gorulebilir;
   retry/timeout ve fail-closed calisir, ancak operasyonel izleme gerekir.
3. Session token + client owner hash izolasyonu hesap tabanli authorization
   degildir. Bearer cifti `localStorage`'da oldugu icin bir XSS acigi bu degerlere
   erisebilir. Kayar TTL 30 gunle sinirlidir; gercek kullanici hesaplari eklenirse
   session sahipligi auth subject ve tercihen HttpOnly cookie ile baglanmalidir.
4. `0004` canli DB'de beklemektedir. Runtime uyumluluk fallback'i veri kaybini
   onler, ancak DB check constraint'inin tam semantik duruma gelmesi icin tablo
   sahibi/migrator credential ile migration uygulanmalidir.
5. Starlette TestClient, kurulu `httpx` icin deprecation warning veriyor;
   test sonucu etkilenmedi fakat bagimlilik guncellemesinde ele alinmalidir.
6. Qdrant koleksiyonu `green` ve nokta sayisi dogrudur; ancak servis
   `indexed_vectors_count=0` raporladigi icin mevcut boyutta dense arama brute
   force calisiyor olabilir. Dogruluk etkilenmedi, buyumede gecikme izlenmelidir.
7. Banka belirtilmeyen sorularda sistem her bankayi arar, fakat kaynakta ilgili
   fact'i olmayan banka icin cevap uydurmaz. Tum bankalari her soruda doldurmak
   ancak eksik veya ayristirilamayan kaynak/fact kayitlarinin tamamlanmasiyla
   mumkundur.
