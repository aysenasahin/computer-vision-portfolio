# 02 - D1 Aorta Segmentasyonu

Durum: hazır model çıkarımı tamamlandı; mevcut maske teknik kontrollerden geçti. Anatomik görsel değerlendirme henüz bekliyor.

## Veri ve kapsam

Kaynak AVT Dongyang D1 CT: `data/raw/avt/Dongyang/D1/D1.nrrd`. Doğrulanmış NIfTI: `outputs/02_aorta_segmentation/D1/ct.nii.gz`. Dönüşümde voxel değerleri birebir aynı ve geometri tolerans içindedir; son çıkarımda dönüşüm tekrarlanmadı. `D1.seg.nrrd` veri setinin referansıdır; modele verilmedi, karşılaştırılmadı ve Dice/IoU hesaplanmadı. Model eğitimi veya klinik doğrulama yapılmadı.

## Gerçek çıkarım sonucu

Normal PowerShell'de proje kökünden çalıştırılan komut:

```powershell
.\.venv\Scripts\python.exe .\experiments\02_aorta_segmentation\resume_aorta_inference.py
```

Yeni çalışma klasörü: `outputs/02_aorta_segmentation/D1_resume_20261010_180001_c35ec40e`.

- TotalSegmentator 2.18.0, nnU-Net 2.8.1, PyTorch 2.14.1+cpu, scikit-learn 1.9.1.
- CPU, total, aorta ve fast ayarları kullanıldı.
- Tam import ön kontrolü: çıkış kodu 0; 40,374 saniye.
- Çıkarım alt süreci: çıkış kodu 0; 446,221 saniye. TotalSegmentator kendi raporunda 436,13 saniye bildirdi; ölçümlerin kapsamı farklıdır.
- Çıktı: `masks/aorta.nii.gz`; durum: `inference_completed_mask_file_present`.
- Orijinal run_metadata.json, terminal_output.txt ve totalsegmentator_report.json korunmuştur. Maske kontrolü ayrı mask_check.json içindedir.

Çalıştırıcı mevcut doğrulanmış NIfTI'yi kullanır, import veya çıkarım hatasında durur, her denemeyi ayrı klasöre kaydeder. Model cache'i `outputs/02_aorta_segmentation/model_cache` altındadır.

## Gerçek teknik kontrol

Model yeniden çalıştırılmadan aşağıdaki komut çalıştırıldı:

```powershell
.\.venv\Scripts\python.exe experiments/02_aorta_segmentation/check_aorta_mask.py --run-dir outputs/02_aorta_segmentation/D1_resume_20261010_180001_c35ec40e
```

Betik mevcut SimpleITK, NumPy ve Matplotlib bağımlılıklarını kullanır; TotalSegmentator veya sklearn import etmez.

| Kontrol | Gözlenen sonuç |
| --- | --- |
| Okunabilir / boş | Okunabilir; boş değil; değerler sonlu |
| Veri tipi | uint8 |
| Etiketler | 0: 85.412.410 voxel; 1: 176.582 voxel |
| Pozitif voxel | 176.582 |
| Voxel hacmi | 1,516296025 mm³ |
| Segmentasyon hacmi | 267,750584723 mL |
| Bağlı bileşen | 1; 6-komşuluk, yüz bağlantısı |
| CT ve maske boyutu | 512 × 666 × 251; fark 0 |
| Spacing | 0,7109375 × 0,7109375 × 2,999999285 mm; fark 0 |
| Origin, LPS | (-187,095993042; -65,942184448; -1193,419921875) mm; fark 0 |
| Direction | Birim matris; fark 0 |

Boyut tam eşitlikle; spacing 1e-5 mm, origin 1e-4 mm, direction 1e-6 mutlak toleransla karşılaştırıldı. Göreli tolerans 0. Tüm farklar sıfır. Geometri doğrulanmadan overlay yapılmaz. Yeniden örnekleme, bileşen silme veya maske düzeltme yapılmadı.

## Görseller ve Slicer

Çalışma klasöründe `mask_check.json`, `overlay_axial.png`, `overlay_coronal.png`, `overlay_sagittal.png` bulunur. Her düzlemde en büyük pozitif maske kesit alanı seçildi: LPS z=188, y=308, x=265. LPS yönleri ve fiziksel en-boy oranı korunmuştur; eksenler mm ve anatomik yönlerle etiketlidir. Pencere depolanan CT değerlerinde -200 ile 600 arasıdır. PNG'lerin üretildiği ve görüntülenebildiği kontrol edildi; anatomik değerlendirme henüz bekliyor.

Slicer'da CT olarak `outputs/02_aorta_segmentation/D1/ct.nii.gz`, maske olarak yeni çalışma klasöründeki `masks/aorta.nii.gz` dosyasını açın. Aynı fiziksel griddedirler.

## Önceki engel ve sınırlamalar

İlk çıkarım Code Integrity / _openmp_helpers import hatasıyla model başlamadan durdu. `outputs/02_aorta_segmentation/D1` altındaki başarısızlık kayıtları korunmuştur. Salt okunur teşhis VerifiedAndReputableDesktop politikasını belirledi; raporlar `outputs/diagnostics` altındadır. Normal PowerShell'de import kontrolü ve ardından çıkarım başarılı oldu. Önceki engel Windows genelinde çıkarımın imkânsız olduğunu göstermez; bağlamlar arasındaki farkın kesin nedeni henüz belirlenmedi. Güvenlik ayarları değiştirilmedi.

Teknik kontrol anatomik doğruluğu kanıtlamaz; tüm hacmin Slicer'da anatomik görsel değerlendirmesi bekliyor. Hacim, pozitif model-maskesi voxellerinin hacmidir; damar duvarı hacmi veya duvar kalınlığı değildir. Referansın damar ağacıyla modelin aorta sınıfının kapsamı henüz eşleştirilmedi. Dice/IoU, eğitim veya klinik doğrulama yapılmadı. Veri lisansı/atıf koşulları yeniden dağıtım öncesinde doğrulanmalıdır. Veri, model ağırlıkları ve yerel çıktılar Git dışındadır.
