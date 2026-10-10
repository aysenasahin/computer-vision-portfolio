# 04 - D1 Maskelerinin Betimleyici Karşılaştırması

Durum: gerçek dosyalarda geometri doğrulandı, üç bölgenin tam hacim ölçümleri ve görseller üretildi. Bu, anatomik kapsamları eşleştirilmemiş iki maskenin betimleyici karşılaştırmasıdır; başarı değerlendirmesi değildir.

## Gerçek çalıştırma

Girdi yolları `outputs/03_reference_inspection/D1/reference_inspection.json` içindeki inputs alanından alındı. Kullanılan tahmin başarılı `D1_resume_20261010_180001_c35ec40e` çalışmasına aittir. Proje kökünden mevcut .venv ile şu çağrı çalıştırıldı:

```powershell
.\.venv\Scripts\python.exe experiments/04_mask_comparison/compare_masks.py --ct outputs/02_aorta_segmentation/D1/ct.nii.gz --prediction outputs/02_aorta_segmentation/D1_resume_20261010_180001_c35ec40e/masks/aorta.nii.gz --reference data/raw/avt/Dongyang/D1/D1.seg.nrrd --output-dir outputs/04_mask_comparison/D1
```

Betik yolları komut satırından alır, yalnızca mevcut SimpleITK, NumPy ve Matplotlib bağımlılıklarını kullanır. 3D binary 0/1 maskeleri kabul eder; farklı katman/etiket yapısını örtük olarak birleştirmez. Geometri uyuşmazsa doğrudan dizi karşılaştırması yapmadan hata raporu yazıp durur. Model, sklearn veya eğitim kodu import edilmez.

## Geometri

Boyut: CT ve her iki maske 512 × 666 × 251. Tahminin CT'ye geometri farkları sıfır. Referansın en büyük origin farkı 2,188e-5 mm, spacing farkı 8,474e-8 mm; direction farkı 0. Boyut tam eşit, spacing 1e-5 mm, origin 1e-4 mm, direction 1e-6 mutlak tolerans ve sıfır göreli toleransla uyum doğrulandı. Yeniden örnekleme veya maske düzeltme yapılmadı.

## Renkler ve ölçümler

| Bölge | Renk | Voxel | Hacim (mL) |
| --- | --- | ---: | ---: |
| İki maskede de bulunan | Yeşil | 153.546 | 232,821189 |
| Yalnız otomatik tahminde | Turuncu | 23.036 | 34,929395 |
| Yalnız referansta | Mavi | 35.895 | 54,427446 |

Hacimler ortak CT grid voxel hacmi olan 0,001516296025 mL üzerinden hesaplandı. Referansın kendi NRRD spacing'iyle önceki deneyde hesaplanan hacimden çok küçük yuvarlama farkı oluşması beklenir. Hacimler damar duvarı hacmi olarak yorumlanamaz.

Tutarlılık kontrolleri geçti:

- 153.546 + 23.036 = 176.582 tahmin voxeli.
- 153.546 + 35.895 = 189.441 referans voxeli.
- Aksiyel CSV'deki 251 kesitin her bölge için toplamı tam hacim sayısını verdi.

## Görseller ve kesit seçimi

`outputs/04_mask_comparison/D1/` altında:

- `axial_comparison_gallery.png`: 12 aksiyel kesit.
- `comparison_coronal.png`: koronal karşılaştırma.
- `comparison_sagittal.png`: sagittal karşılaştırma.
- `comparison_report.json`: yollar, renklerin RGB değerleri, geometri farkları/toleransları, sayılar, hacimler, seçim yöntemi, LPS kesit indeksleri ve fiziksel düzlem konumları.
- `axial_region_counts.csv`: tüm 251 aksiyel kesit için fiziksel S koordinatı, üç bölgenin voxel sayıları ve iki maske toplamı.

Aksiyel seçim: maskelerin birleşiminin pozitif olduğu kesit listesinde eşit aralıklı 9 sıra, ayrıca üç bölgenin ayrı ayrı en büyük kesit alanı; tekrarlar çıkarıldı. Seçilen LPS z indeksleri: 20, 49, 77, 106, 134, 163, 164, 189, 192, 196, 220, 249. Bu seçim tüm hacmin anatomik incelemesi yerine geçmez.

Koronal/sagittal seçim: maskelerin birleşiminin en büyük kesit alanı. Koronal y=311 (LPS P=155,159378 mm), sagittal x=265 (LPS L=1,302444 mm). Aksiyel fiziksel konumlar JSON'da kaydedildi. Görseller LPS yönünde, mm eksenleri ve eşit fiziksel ölçekle üretildi; CT görüntüleme penceresi depolanan değerlerde -200 ile 600. Renk açıklamaları görsellerde bulunur. PNG'lerin görüntülenebildiği kontrol edildi; anatomik doğruluk değerlendirmesi yapılmadı.

## Kapsam farkı

Referans tek birleşik segment/etikettir; kullanıcı Slicer ön incelemesinde referansın daha fazla damar dalı içerdiğini bildirdi. Bu gözlem uzman doğrulaması değildir. Turuncu ve mavi bölgeler otomatik olarak yanlış pozitif/yanlış negatif olarak adlandırılamaz; dal kapsamı, uç sınırlar ve anotasyon farkları ayrıştırılmamıştır.

Dice/IoU veya başarı yüzdesi hesaplanmadı. Referans tahmine göre kırpılmadı, dallar budanmadı, maskeler değiştirilmedi. Bu karşılaştırmada model çıkarımı veya eğitim yapılmadı. Ham veri ve önceki kayıtlar korunmuştur; yeni çıktılar Git dışında tutulur.

## Kullanıcının ilettiği görsel ön inceleme

Seçilmiş görsellerin kullanıcı tarafından bildirilen ön incelemesinde aksiyel 106, 134, 189 ve 192 kesitlerinde ana aortada belirgin ortak alan görülmüştür. Referansta boyun ve alt ekstremite yönündeki ek damar dalları vardır. Sagittal görüntüde ana damarın alt bitiş sınırı farklıdır. Aksiyel 163–164 kesitlerinde kalbe yakın başlangıç bölgesinde ve sagittal görüntüde bazı ana aorta kenarlarında yalnız tahmine ait alanlar görülmüştür.

Bunlar uzman anatomik doğrulaması veya bütün kesitlerin incelenmesi değildir; ölçülmüş başarı olarak sunulmaz. Kapsam farkı nedeniyle bu bölgeler hata sınıfları olarak yorumlanmaz. Tüm hacimde uzman incelemesi ve anatomik kapsamı bağımsız biçimde eşleştirilmiş referans gereksinimi devam etmektedir. Ayrıntılı yerel not `outputs/04_mask_comparison/D1/visual_review_tr.md` içindedir; outputs ve veri içeren görseller ilk yayına dahil edilmez.
