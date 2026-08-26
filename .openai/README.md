# HititFinLex Frontend

Katılım bankacılığına özel; ürün keşfi, banka karşılaştırması, kaynaklı RAG
asistanı ve veri kalitesi ekranlarını içeren Next.js arayüzüdür.

## Gereksinimler

- Node.js 20 veya daha yeni bir sürüm
- Çalışan HititFinLex API V2.7
- PostgreSQL veritabanı ve mevcut `.env` ayarları

## Windows'ta çalıştırma

Önce backend'i ayrı bir CMD penceresinde başlatın:

```cmd
cd /d C:\Users\PC\Documents\katilim_finans_app
.venv\Scripts\activate
set PYTHONUTF8=1
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Ardından frontend klasöründe ikinci bir CMD açın:

```cmd
cd /d C:\Users\PC\Documents\HititFinLex
npm install
npm run dev
```

Tarayıcı adresi:

```text
http://localhost:3000
```

Frontend varsayılan olarak `http://127.0.0.1:8000` API adresini kullanır.
Farklı bir backend adresi gerekiyorsa `.env.local` dosyasına şunu ekleyin:

```env
NEXT_PUBLIC_API_BASE_URL=https://api-adresiniz.example
```

## Başlıca özellikler

- PostgreSQL verileriyle güncellenen genel dashboard
- Arama, çoklu filtre, güven eşiği, sıralama ve sayfalama
- Kanıt metinleri ve ham kaynak içeren belge ayrıntısı
- Ürün türüne göre değişen karşılaştırma alanları
- Her banka için birden fazla belgeyi birleştiren karşılaştırma sütunları
- BGE-M3, PostgreSQL ve Qwen tabanlı kaynaklı RAG asistanı
- Sınıflandırma, NER kapsamı ve inceleme kuyruğu görünümü
- Genel kapsam ve ürün karşılaştırmasına uygun kapsamın ayrı izlenmesi

Kart kampanyalarında tarih, harcama eşiği, indirim, ödül, puan ve taksit;
finansman ürünlerinde amaç, tür, tutar, oran, vade, kâr payı ve masraf;
yatırım ürünlerinde yatırım aracı ve işlem kanalı otomatik olarak öne alınır.
Veride hiç bulunmayan alanlar matris satırı olarak üretilmez.
