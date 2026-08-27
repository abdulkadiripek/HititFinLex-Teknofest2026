# Sürümlü aktarım paketleri

`scripts/artifact-manifest.mjs`, veritabanı yedeği, model klasörleri ve
kurulum kilit dosyaları için taşınabilir bir SHA-256 manifesti üretir. Manifest
her dosyanın boyutunu ve özetini, ayrıca klasörün deterministik paket özetini
tutar. Sembolik bağlantılar kabul edilmez; böylece paket dışında bir dosyaya
sessizce başvurulamaz.

## Paket oluşturma

Önce aktarılacak dosyaları aynı `transfer/` klasöründe toplayın. Gerçek model
ağırlıkları ve yedekler Git'e eklenmez.

```powershell
npm run artifacts -- create `
  --release api-v1.3.0 `
  --output transfer/manifest.json `
  database@postgresql-18_pgvector-0.8.6_schema-0002=transfer/katilim_finans.backup `
  ner@ner-v4=transfer/hititfinlex-model-ner-v4.zip `
  campaign@classifier-campaign-v1=transfer/hititfinlex-model-classifier-campaign-v1.zip `
  product@classifier-product-v2=transfer/hititfinlex-model-classifier-product-v2.zip
```

Linux/macOS'ta aynı komut ters eğik çizgi yerine `\` satır devamıyla
çalıştırılabilir. `--output` mevcut bir dosyanın üzerine yazmaz.

## Paketi doğrulama

Aktarımdan önce ve sonra şu komut aynı sonucu vermelidir:

```cmd
npm run artifacts -- verify transfer/manifest.json
```

Doğrulama başarılı olmadan `pg_restore` çalıştırmayın ve model klasörlerini
üretim yoluna taşımayın. Manifestteki sürüm değerleri şu bileşenlerle
eşleşmelidir:

| Paket | Sürüm kimliği |
| --- | --- |
| API/transfer seti | `api-v1.3.0` |
| Veritabanı | `postgresql-18_pgvector-0.8.6_schema-0002` |
| NER | `ner-v4` |
| Kampanya sınıflandırıcı | `classifier-campaign-v1` |
| Ürün sınıflandırıcı | `classifier-product-v2` |

Model arşivlerinin kesin adları, kurulum hedefleri ve arşiv içi kök klasörleri
[`model-package-spec.json`](model-package-spec.json) içinde makinece okunabilir
biçimde sabitlenmiştir. Arşivden çıkarılan klasörü bu hedefin üstüne sessizce
yazmayın: önce manifesti doğrulayın, yeni klasörü ayrı bir geçici hedefte açın
ve uygulamanın model yükleme testinden sonra atomik olarak devreye alın.

Kaynak kurulumunun kilitleri (`package-lock.json`, `backend/requirements.txt`)
`source-manifest.json` içinde ayrıca sürümlenip doğrulanır.

## Doğrulanmış yerel release varlıkları

27 Ağustos 2026'da oluşturulan gerçek model ZIP'lerinin boyut ve SHA-256
değerleri [`model-release-manifest.json`](model-release-manifest.json), gerçek
güncel ve tarihsel PostgreSQL custom-format yedeklerinin kayıtları ise
[`database-release-manifest.json`](database-release-manifest.json) içindedir.
Bu manifestlerde `download_url: null`, dosyanın henüz kimliği doğrulanmış bir
release hedefine yüklenmediği anlamına gelir. Upload başarıyla bitmeden URL
tahmin etmeyin; sonrasında URL'yi doldurup indirilen baytları aynı SHA-256 ile
yeniden doğrulayın.
