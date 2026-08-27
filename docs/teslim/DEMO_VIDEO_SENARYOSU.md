# HititFinLex demo videosu senaryosu

Bu senaryo, `HititFinLex_Demo_Videosu.mp4` içindeki yaklaşık 80 saniyelik
teknik akışın konuşma metnidir. Videodaki ekran görüntüleri yerel çalışan
frontend'in canlı API'ye bağlı hâlinden alınır; sabit finansal örnek veri
kullanılmaz.

## Zaman akışı

| Süre | Görüntü | Anlatım |
| --- | --- | --- |
| 00:00–00:08 | Açılış | “HititFinLex, katılım bankacılığı ürünlerini güncel ve tarihsel resmî kaynaklarıyla birlikte sunan yerel bir karar destek platformudur.” |
| 00:08–00:20 | Genel bakış | “Sistem panoramasında API, yerel GPU ve modellerin hazırlık durumu ile 771 güncel ve 2.580 tarihsel belgenin kapsamı izlenir.” |
| 00:20–00:32 | Ürün kataloğu | “Akıllı katalog; banka, ürün, dönem, güven ve bilgi kapsamı filtreleriyle toplam 3.351 kaydı keşfetmeyi sağlar.” |
| 00:32–00:46 | Karşılaştırma | “Karşılaştırma matrisi finansal koşulları banka bazında yan yana getirir. Her değer kendi belge ve kanıt cümlesine bağlıdır; eksik alan uydurulmaz.” |
| 00:46–01:00 | Akıllı asistan | “BGE-M3 hibrit arama, ilgili resmî metinleri getirir; yerel Qwen modeli yalnız bu bağlamdan, benzersiz dipnotlarla cevap üretir.” |
| 01:00–01:12 | Veri kalitesi | “Silver ve model çıkarımı olan kayıtlar doğrulanmış gibi sunulmaz. Güven, kapsam ve insan inceleme durumu kullanıcıya açıkça gösterilir.” |
| 01:12–01:20 | Kapanış | “HititFinLex: yerel, kaynaklı, tarihsel ve denetlenebilir katılım finansı karar desteği.” |

## Yeniden üretim

Backend `127.0.0.1:8000`, frontend `127.0.0.1:3000` üzerinde çalışırken:

```powershell
node scripts/capture-demo-screenshots.mjs
powershell -ExecutionPolicy Bypass -File scripts/generate-teslim-sunumu.ps1
```

İkinci komut ana sunumu PPTX ve PDF olarak, demo akışını PPTX ve MP4 olarak
`docs/teslim/` altına yazar. MP4 sessizdir; metinler videonun içinde yer
alır. Seslendirme istenirse yukarıdaki anlatım aynı zaman kodlarıyla
eklenebilir.

## Kayıt güvenliği

- Ekranda `.env`, veritabanı parolası veya yönetim API anahtarı açılmamalıdır.
- Finansal çıktıların “model çıkarımı / doğrulanmadı” işareti kadrajda
  korunmalıdır.
- Video açıklamasına “finansal tavsiye değildir” notu eklenmelidir.
- Yayımlanan videonun erişim bağlantısı teslim formunda ayrıca sınanmalıdır.
