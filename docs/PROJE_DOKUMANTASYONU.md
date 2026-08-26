# Proje Dokümantasyonu

**Takım:** HititFinLex · **Yarışma:** TEKNOFEST 2026 Yapay Zeka Dil Ajanları
Yarışması, 2. Senaryo (`#BilisimVadisi2026`)

Bu belge, şartnamenin "6. Tespit Edilmesi Gerekenler" bölümünde istenen
proje dokümantasyonu maddelerini karşılar.

## İçindekiler

- [Tarihsel arşiv katmanı](#tarihsel-arşiv-katmanı)
- [Sistem mimarisi ve veri akışı](#sistem-mimarisi-ve-veri-akışı)
- [Kullanılan NLP yaklaşımı](#kullanılan-nlp-yaklaşımı)
- [Veri seti](#veri-seti)
- [Veri ön işleme adımları](#veri-ön-i̇şleme-adımları)
- [Model ve kural yapısı](#model-ve-kural-yapısı)
- [Karşılaştırma yaklaşımı](#karşılaştırma-yaklaşımı)
- [Çalıştırma talimatları](#çalıştırma-talimatları)
- [Karşılaşılan problemler ve çözüm yaklaşımları](#karşılaşılan-problemler-ve-çözüm-yaklaşımları)
- [Model çıktı örnekleri](#model-çıktı-örnekleri)
- [Model performans değerlendirmesi](#model-performans-değerlendirmesi)

## Tarihsel arşiv katmanı

Güncel veriye ek olarak sistem, belgelerin zaman içindeki değişimini de
saklar: `archive_*.py` script'leri (bakım/toplu iş, `api.py` tarafından
çalışma zamanında import edilmez) belgeleri arşive taşır ve denetler;
`historical_search_v28.py` bu arşiv üzerinde hibrit arama yapar. Frontend
bunu `/history/overview`, `/history/search`, `/history/comparison`
(`as_of` tarih parametresiyle) ve `/history/chat` uçları üzerinden
"son 1 ay / 3 ay / 6 ay / 1 yıl / tüm arşiv" seçenekleriyle sunar — bkz.
kök `README.md#özellikler`.

## Sistem mimarisi ve veri akışı

Uçtan uca akış: **web scraping → veri ön işleme → NER + sınıflandırma →
finansal bilgi çıkarımı (fact extraction) → PostgreSQL/pgvector →
hibrit arama + RAG → dashboard/chatbot arayüzü**. Bileşenlerin ayrıntılı
şeması ve teknoloji yığını için bkz. [kök `README.md`](../README.md#mimari)
ve [`backend/README.md`](../backend/README.md#mimari).

## Kullanılan NLP yaklaşımı

Sistem üç katmanlı bir NLP yaklaşımı kullanır:

1. **Metin sınıflandırma** — `dbmdz/bert-base-turkish-cased` taban modeli
   üzerine fine-tune edilmiş iki ayrı sınıflandırıcı
   (`train_classifier.py`, `train_product_v2.py`):
   - **Kampanya sınıflandırıcı** (`classifier_campaign_v1`): bir metnin
     kampanya duyurusu olup olmadığını (`EVET`/`HAYIR`) belirler.
   - **Ürün sınıflandırıcı** (`classifier_product_v2`): metni 13 ürün
     türünden birine atar (`KONUT_FINANSMANI`, `TASIT_FINANSMANI`,
     `IHTIYAC_FINANSMANI`, `KART_KAMPANYASI`, `KATILMA_HESABI` vb. — tam
     liste `backend/models/*_training_summary.json` içinde).
2. **Adlandırılmış varlık tanıma (NER)** — `ner_v4` modeli (yine
   `dbmdz/bert-base-turkish-cased` tabanlı, BIO etiketleme şemasıyla
   token-classification), metin içinden `KAR_PAYI_ORANI`,
   `FINANSMAN_TUTARI`, `VADE_SURESI`, `TAHSIS_UCRETI`, `EKSPERTIZ_UCRETI`,
   `IPOTEK_TESIS_UCRETI`, `INDIRIM_ORANI`, `ODUL_TUTARI`,
   `TAKSIT_SAYISI` gibi 17 varlık türünü span olarak çıkarır
   (`train_ner.py`, `predict_ner.py`).
3. **Kural tabanlı bağlam doğrulama** — NER'in ham çıktısı, `%2,05` gibi
   sayısal ifadelerin hangi bağlamda geçtiğine bakan Türkçe'ye özel kural
   kümeleriyle (`fact_context_rules.py`, `fact_surface_rules.py`,
   `coverage_rules_v27.py`) süzülür. Örneğin `VADE_SURESI` etiketli bir
   span, metinde "erken ödeme", "ilk taksit" veya "cayma tazminatı"
   bağlamında geçiyorsa dışlanır (`excluded_early_payment_context` vb.);
   `FINANSMAN_TUTARI`, "toplam geri ödeme tutarı" bağlamında geçiyorsa
   dışlanır (`excluded_total_repayment_context`). Bu, saf model çıktısının
   üzerine konan, yanlış pozitifleri azaltan bir güvenlik katmanıdır.

Her varlık türü için ayrı bir **otomatik kabul güven eşiği**
(`AUTO_THRESHOLDS`, `fact_context_rules.py:7-25`) tanımlıdır — örn.
`KAR_PAYLASIM_ORANI` için 0.95, `EKSPERTIZ_UCRETI` için 0.55. Eşiğin
altında kalan veya kural katmanınca dışlanan çıkarımlar otomatik
reddedilmez, **insan inceleme kuyruğuna** düşer (`review_service.py`,
`human_review_v1` iş akışı) — bkz. dashboard'daki "Veri kalitesi" görünümü.

## Veri seti

Bkz. [`backend/README.md#veri-seti`](../backend/README.md#veri-seti). Ham
metinler BDDK'nın katılım bankacılığı kuruluşları listesindeki
([bddk.org.tr/Kurulus/Liste/77](https://www.bddk.org.tr/Kurulus/Liste/77))
bankaların resmî web sitelerinden toplanmıştır; etiketli eğitim/doğrulama
setleri `backend/data/` altında bu repoyla birlikte paylaşılmıştır.

## Veri ön işleme adımları

- **Metin normalizasyonu**: Türkçe'ye özgi karakterler (`ı`/`İ`) ile
  Unicode aksan işaretleri ayrıştırılıp kaldırılır (`unicodedata.NFKD` +
  combining-mark temizliği), fazla boşluklar tekilleştirilir ve metin
  `casefold()` ile küçük harfe çevrilir (`fact_context_rules.fold_text`).
  Bu, "Kâr Payı" / "kar payi" gibi yazım farklılıklarının aynı şekilde
  eşleşmesini sağlar.
- **Sayısal format normalizasyonu**: `%2,05`, `% 2.05`, `2.05 %` gibi
  farklı yüzde gösterimleri ve `500 TL` / `500₺` / `500 Türk Lirası` gibi
  para birimi gösterimleri tek bir standart değere indirgenir (şartname
  madde 5.6 gereksinimine karşılık gelir).
- **Chunk'lama**: Belgeler `intake_service.py` içinde `document_chunks`
  tablosuna, her biri kendi `content`, `token_count` ve `content_hash`
  (SHA-256) değeriyle parçalanarak yazılır; aynı içerik tekrar
  işlenmişse `content_hash` üzerinden çift kayıt engellenir
  (`intake_duplicate_gate: record_hash_first_v1`, bkz. `/health`).
- **İkili indeksleme**: Her chunk için hem `BAAI/bge-m3` ile 1024
  boyutlu yoğun (dense) vektör embedding'i hem de PostgreSQL
  `tsvector` (`to_tsvector('simple', ...) || to_tsvector('turkish', ...)`)
  seyrek (sparse) metin indeksi üretilir — hibrit aramanın (BM25 benzeri +
  vektör) temelini oluşturur (`hybrid_search.py`).

## Model ve kural yapısı

| Bileşen | Yaklaşım | Girdi → Çıktı |
| --- | --- | --- |
| Kampanya sınıflandırıcı | Fine-tuned BERT (ikili sınıflandırma) | Belge metni → `EVET`/`HAYIR` |
| Ürün sınıflandırıcı | Fine-tuned BERT (13 sınıflı) | Belge metni → ürün türü kodu |
| NER (`ner_v4`) | Fine-tuned BERT + BIO token classification | Belge metni → 17 varlık türü span'i |
| Bağlam kuralları | Deterministik, regex/anahtar-kelime tabanlı | (etiket, kanıt metni) → kabul/red + gerekçe kodu |
| Kapsam kuralları (`coverage_rules_v27.py`) | Deterministik | Ürün türüne göre hangi alanların matrise gireceğini belirler |
| Hibrit arama | BM25 benzeri (`tsvector`) + kosinüs benzerliği (`pgvector`) | Sorgu → ilgili chunk'lar |
| RAG cevap üretimi | Ollama `qwen3.5:9b` (yerel LLM) | Sorgu + getirilen chunk'lar → kaynak gösteren cevap |

## Karşılaştırma yaklaşımı

`extract_comparison_facts.py`, NER + kural katmanından geçen her kabul
edilmiş varlığı `comparison_facts` tablosuna normalize edilmiş bir
`(fact_type, normalized_value, evidence_text, confidence)` kaydı olarak
yazar. `coverage_rules_v27.py`, ürün türüne göre karşılaştırma matrisinde
hangi `fact_type`'ların satır olarak üretileceğini belirler (örn. kart
kampanyalarında tarih/harcama eşiği/indirim/puan; finansman ürünlerinde
tutar/oran/vade/kâr payı) — **veride hiç bulunmayan alanlar matrise
eklenmez**. Frontend'in `/comparison` uç noktası, seçilen ürün türü için
bu fact'leri bankalara göre pivotlayıp kanıt metnine kadar izlenebilir bir
tablo döndürür (bkz. kök `README.md#api-sözleşmesi`).

## Çalıştırma talimatları

Bkz. kök [`README.md#kurulum`](../README.md#kurulum) (frontend) ve
[`backend/README.md#kurulum`](../backend/README.md#kurulum) (backend) —
adım adım kurulum, ortam değişkenleri ve doğrulama testleri orada
ayrıntılıdır.

## Karşılaşılan problemler ve çözüm yaklaşımları

> **Takımın doldurması gerekiyor.** Geliştirme sürecinde karşılaşılan
> teknik zorluklar ve çözüm yaklaşımları (örn. Türkçe'ye özgü karakter
> normalizasyonu ihtiyacı, PostgreSQL locale/pgvector kurulum sorunları,
> yanlış pozitif NER çıkarımlarını azaltmak için bağlam kurallarının
> eklenmesi vb.) buraya, ekibin kendi deneyiminden yazılmalıdır — bu
> içerik kod tabanından çıkarılamaz.

## Model çıktı örnekleri

Örnek bir NER isteği ve beklenen çıktı (şartname madde 15.4'teki test
metniyle):

```text
Girdi: "Konut finansmani kapsaminda 500.000 TL finansman 120 ay vade ve
%2,79 kar payi orani ile sunuluyor. Tahsis ucreti 2.500 TL, ekspertiz
ucreti 8.000 TL ve ipotek tesis ucreti 3.000 TL'dir."

Beklenen etiketler: FINANSMAN_TUTARI (500.000 TL), VADE_SURESI (120 ay),
KAR_PAYI_ORANI (%2,79), TAHSIS_UCRETI (2.500 TL),
EKSPERTIZ_UCRETI (8.000 TL), IPOTEK_TESIS_UCRETI (3.000 TL)
```

Swagger UI üzerinden (`http://127.0.0.1:8000/docs`) `POST /ner` veya
`POST /analyze` uç noktalarıyla canlı olarak denenebilir.

> **Takımın eklemesi önerilir:** gerçek veri setinden birkaç `/comparison`
> ve `/chat` çıktı ekran görüntüsü/örneği (demo videosunda da gösterilecek).

## Model performans değerlendirmesi

**Sınıflandırıcılar** — test seti üzerinde ölçülen metrikler
(`backend/models/classifier_training_summary.json`,
`classifier_product_v2_training_summary.json`):

| Model | Test Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- |
| Kampanya sınıflandırıcı (`campaign_v1`) | 0.9912 | 0.9908 | 0.9912 |
| Ürün sınıflandırıcı (`product_v2`) | 0.9123 | 0.8759 | 0.9109 |

**NER (`ner_v4`)** — `train_ner.py`, `seqeval` kütüphanesiyle entity-level
precision/recall/F1 (`classification_report`, satır 218-229) hesaplar ve
en iyi checkpoint'i F1'e göre seçer (`metric_for_best_model: "f1"`).
Nihai test seti metrikleri ayrı bir özet dosyası olarak
kaydedilmediğinden, tam sayılar için `train_ner.py`'nin eğitim
konsolu çıktısı/logu referans alınmalıdır.

> **Takımın eklemesi gerekiyor:** `ner_v4` için nihai
> precision/recall/F1 değerleri (varsa eğitim logundan alınıp buraya
> eklenmelidir).
