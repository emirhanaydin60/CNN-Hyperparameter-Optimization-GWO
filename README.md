# CNN Hyperparameter Optimization with GWO

Bu proje, Fashion-MNIST veri seti üzerinde Grey Wolf Optimizer (GWO) kullanarak bir CNN modelinin hiperparametrelerini optimize eden, sonuçları akademik raporlamaya uygun şekilde kaydeden deney paketidir.

## Proje Amacı

Amaç, optimize edilmemiş bir baseline CNN ile GWO ile optimize edilmiş CNN modelini karşılaştırmak ve optimizasyonun test doğruluğuna etkisini ölçmektir. Tüm arama süreci validation accuracy üzerinden yürütülür; test seti yalnızca final değerlendirmede kullanılır.

## Veri Seti

- Veri seti: Fashion-MNIST
- Toplam eğitim verisi: 60.000 örnek
- Bölünme: 55.000 train, 5.000 validation, 10.000 test

## CNN Mimarisi

Kullanılan model basit bir CNN’dir:

- 1. Conv2d + ReLU + MaxPool2d
- 2. Conv2d + ReLU + MaxPool2d
- Flatten
- Fully connected katman
- Dropout
- Çıkış katmanı

Optimizasyona açık hiperparametreler:

- `filter_size`
- `filters`
- `dilation`
- `final_neurons`
- `dropout`

## GWO Algoritması

GWO, her aday CNN konfigürasyonunu kısa süreli bir eğitimle değerlendirir. Fitness değeri validation accuracy’dir. Her iterasyonda kurtların pozisyonları alpha, beta ve delta rehberliğinde güncellenir. En iyi çözüm validation accuracy’ye göre seçilir.

## Konfigürasyon Yönetimi

Tüm deney parametreleri merkezi olarak `config.py` içinde tutulur veya JSON üzerinden override edilebilir.

Önemli parametreler:

- `population_size`
- `iteration_count`
- `search_epochs`
- `final_epochs`
- `batch_size`
- `learning_rate`
- `subset_size`
- `random_seed`
- `patience`
- `runs`

## Çalıştırma Komutları

Tek koşu:

```bash
python train.py --population-size 6 --iteration-count 10 --search-epochs 2 --final-epochs 20
```

Çoklu koşu:

```bash
python train.py --runs 3 --population-size 6 --iteration-count 10 --search-epochs 2 --final-epochs 20
```

JSON konfigürasyon ile:

```bash
python train.py --config my_config.json
```

## Sonuç Dosyaları

Çalışma sonunda `results/` içinde aşağıdaki dosyalar otomatik oluşturulur:

- `best_model.pth`
- `best_config.txt`
- `final_results.txt`
- `statistics.txt`
- `timing.txt`
- `convergence.txt`
- `confusion_analysis.txt`
- `summary.csv`
- `search_history.csv`
- `search_space.json`
- `config_used.json`
- `run.log`
- `global_best.png`
- `local_bests.png`
- `accuracy_curve.png`
- `loss_curve.png`
- `confusion_matrix.png`

## Dosya Açıklamaları

- `final_results.txt`: deney parametreleri, baseline/GWO doğrulukları, improvement değerleri ve zaman bilgileri
- `statistics.txt`: çoklu koşularda accuracy ve improvement için mean/std özetleri
- `timing.txt`: optimizasyon, baseline eğitim, final eğitim ve toplam deney süreleri
- `convergence.txt`: GWO yakınsama özeti
- `confusion_analysis.txt`: en çok karışan sınıflar ve sınıf bazlı doğruluk analizi
- `summary.csv`: her koşu için kısa karşılaştırma tablosu
- `search_history.csv`: her fitness değerlendirmesinin kayıtları
- `search_space.json`: arama uzayı tanımı
- `config_used.json`: deneyde kullanılan konfigürasyon
- `run.log`: deney sürecinin log kaydı

## Notlar

- Eğitim sırasında seed sabitlenir; random, numpy, torch ve CUDA tarafı aynı seed mekanizmasına bağlıdır.
- Final eğitimde train + validation birleşik veri kullanılır.
- Confusion matrix normalize edilmiş olarak kaydedilir.
- Baseline model sabit hiperparametrelerle çalışır; yalnızca GWO tarafında hiperparametreler optimize edilir.
