# 01 - Hacim İncelemesi

Durum: tamamlandı (yalnızca D1 CT geometrisi ve örnek kesitler)

## Amaç

SimpleITK ile bir 3B NRRD görüntüsünü okumak; boyut, voxel aralığı, fiziksel yönelim ve yoğunluk aralığını raporlamak; yön bilgisine göre aksiyel, koronal ve sagittal merkez kesitlerini kaydetmek. Kaynak görüntü okunur, görselleştirme için yönlendirilmiş kopyası kullanılır; kaynak voxel değerleri değiştirilmez.

## Veri

- Yerel dosya: `data/raw/avt/Dongyang/D1/D1.nrrd`.
- Veri seti adı/alt kümesi: AVT, Dongyang, D1 (çalışma alanında kullanıcı tarafından sağlandı).
- Eşlik eden `D1.seg.nrrd`, veri setinin sağladığı referans segmentasyondur. Bu deneyde okunmadı, model çıktısı değildir ve değerlendirme için kullanılmadı.
- Arşiv, yayın, kesin veri seti sürümü ve lisans bilgileri bu çalışma alanındaki dosyalarda bulunmadığından bağımsız olarak doğrulanamadı. Bu nedenle görselleri/veriyi yeniden dağıtmadan önce kaynak koşullarını doğrulayın.

## Kurulum

Proje kökünde PowerShell kullanın. `.venv` henüz yoksa:

```powershell
py -3.12 -m venv .venv
```

Deney bağımlılıklarını proje ortamına yükleyin:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Doğrulanan sürümler: Python 3.12.0, SimpleITK 2.5.6, NumPy 2.5.3 ve Matplotlib 3.11.2. `requirements.txt` doğrudan bağımlılık sürümlerini sabitler.

## Çalıştırma

Proje kökünden, girdi ve çıktı klasörlerini komut satırında belirtin:

```powershell
.\.venv\Scripts\python.exe experiments\01_volume_inspection\inspect_volume.py `
	--input data\raw\avt\Dongyang\D1\D1.nrrd `
	--output-dir outputs\01_volume_inspection
```

Betik `report.json`, `axial.png`, `coronal.png` ve `sagittal.png` üretir. Kaynak NRRD dosyasını değiştirmez. Düzlemleri SimpleITK ile LPS yönelimine çevirerek seçer; NumPy dizi sırası `z, y, x` olsa da anatomik isimleri yalnızca indeks numarasından çıkarmaz. PNG'lerde eksen aralıkları voxel spacing'e göre mm cinsindendir ve fiziksel en-boy oranı korunur. Kontrast, üç merkez kesitin birleşik 1. ve 99. yüzdelik değerleriyle sınırlandırılır; JSON'daki minimum/maksimum bu görüntüleme penceresinden bağımsız, tüm kaynak hacme aittir.

## Gerçek Sonuçlar

D1 görüntüsü başarıyla okundu ve çıktılar `outputs/01_volume_inspection/` altında oluşturuldu.

| Ölçüm | Gerçek değer |
| --- | --- |
| Görüntü boyutu, x-y-z | 512 x 666 x 251 voxel |
| NumPy dizi şekli, z-y-x | 251 x 666 x 512 |
| Voxel spacing, x-y-z | 0.7109375 x 0.7109375 x 2.9999992 mm |
| Origin, LPS | (-187.096, -65.9421875, -1193.4199) mm |
| Direction | Birim matris; LPS yönelimi |
| Piksel veri tipi | 16-bit signed integer |
| Tüm hacimde yoğunluk min-max | -1024 .. 3071 |
| Girdi dosyası boyutu | 75,184,331 bayt |
| Kaydedilen merkez kesitler | axial z=125; coronal y=333; sagittal x=256 |

Ölçümlerin tam hassasiyeti ve örnek indeksleri JSON raporundadır. PNG'ler görsel olarak kontrol edildi; bu, klinik doğrulama veya segmentasyon değerlendirmesi değildir.

## Sınırlamalar

- Yalnızca tek bir görüntü ve her düzlemden bir merkez kesit incelendi.
- Referans segmentasyon karşılaştırılmadı; otomatik segmentasyon, model eğitimi ve klinik doğrulama yapılmadı.
- Veri kaynağının tam atıf bilgisi, sürümü ve lisansı yerel dosyalardan doğrulanamadı.
- Görseller ve JSON `outputs/` altında yerel deney çıktısıdır ve Git dışında tutulur. Yayınlamadan önce veri koşullarını ve görüntülerdeki olası tanımlayıcı bilgileri ayrıca inceleyin.
