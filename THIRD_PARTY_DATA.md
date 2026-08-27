# Üçüncü taraf veri ve model bildirimi

Bu bildirim hukuki danışmanlık değildir; deponun hangi parçalarının proje
lisansından ayrı haklara tabi olabileceğini görünür kılar.

## Banka web içeriği

`dataset/` ve `backend/data/` içindeki kayıtlar, BDDK'nın katılım bankaları
listesinde yer alan kuruluşların kamuya açık web sayfalarından türetilmiştir.
Tarihsel kayıtların bir bölümü Internet Archive Wayback Machine yakalamalarına
ait kaynak URL'leri taşır. Kaynak URL, banka, toplama/yakalama tarihi ve içerik
parmak izi veri satırlarında korunur.

Apache-2.0 lisansı; HititFinLex tarafından yazılan kodu, şemayı,
dokümantasyonu ve lisanslanabilir ölçüde seçim/düzenleme katkılarını kapsar.
Banka sayfalarındaki özgün metin, görsel, marka ve ticari adların mülkiyetini
projeye devretmez. Veri setini yeniden dağıtan veya farklı amaçla kullanan kişi
kaynak sitelerin koşullarını, robots kurallarını, kişisel veri ve fikrî mülkiyet
yükümlülüklerini ayrıca değerlendirmelidir.

## Haricî hizmet ve kaynaklar

- Kuruluş kapsamı: BDDK katılım bankaları listesi.
- Tarihsel yakalama adresleri: Internet Archive Wayback Machine.
- Ham sayfalar: ilgili bankaların resmî alan adları; her kaydın
  `kaynak_url`/`arsiv_url` alanı asıl kaynağı gösterir.

Internet Archive veya banka içeriğinin bu depoda anılması, bu kuruluşların
HititFinLex'i desteklediği ya da onayladığı anlamına gelmez.

## Modeller

Depo model ağırlıklarını içermez. Eğitim/çalıştırma sırasında kullanılan temel
modeller kendi upstream lisanslarına tabidir:

- `dbmdz/bert-base-turkish-cased` — NER ve sınıflandırma tabanı.
- `BAAI/bge-m3` — embedding modeli.
- `qwen3.5:9b` — Ollama üzerinden yerel cevap üretimi.

Bir transfer paketine model ağırlığı eklemeden önce ilgili model kartını ve
lisansı kaydedin. `scripts/artifact-manifest.mjs` bütünlük ve sürüm kaydı
sağlar; kullanım hakkı vermez.
