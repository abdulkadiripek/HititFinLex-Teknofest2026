# HititFinLex teslim paketi

Bu klasör, şartname kapsamındaki sunum ve demo materyallerinin sürümlü
kopyasını içerir.

## Dosyalar

- `HititFinLex_Teknofest2026_Sunumu.pptx` — düzenlenebilir ana sunum.
- `HititFinLex_Teknofest2026_Sunumu.pdf` — ana sunumun sabit PDF çıktısı.
- `HititFinLex_Demo_Videosu.mp4` — gerçek uygulama ekranlarından üretilen,
  yaklaşık 80 saniyelik altyazılı teknik demo.
- `HititFinLex_Demo_Akisi.pptx` — demo videosunun düzenlenebilir kaynak akışı.
- `ekran-goruntuleri/` — çalışan uygulamadan alınan kaynak görüntüler.
- `DEMO_VIDEO_SENARYOSU.md` — zaman kodlu konuşma ve yeniden üretim metni.
- `SHA256SUMS.txt` — dört ikili teslim dosyasının SHA-256 özeti.

## Son teslim kontrolü

- [ ] `main` dalının CI kontrolleri başarılı.
- [ ] `BilisimVadisi2026` etiketi doğrulanmış son teslim commit'ini gösteriyor.
- [ ] GitHub'da PPTX, PDF ve MP4 dosyaları indirilebiliyor.
- [ ] Yarışma sistemine girilen GitHub etiketi/ref'i anonim tarayıcıda açılıyor.
- [ ] Yarışma sistemine girilen demo video bağlantısı anonim tarayıcıda oynuyor.
- [x] Video/sunum üzerinde kişisel anahtar, parola veya yerel `.env` değeri yok.
- [x] Veri setinin silver/kural tabanlı durumu ve finansal tavsiye olmadığı açık.

> GitHub deposuna dosya eklemek, yarışma platformundaki form alanlarını
> otomatik olarak doldurmaz. Son iki bağlantı teslim hesabından ayrıca
> girilmeli ve gizli pencerede sınanmalıdır.
