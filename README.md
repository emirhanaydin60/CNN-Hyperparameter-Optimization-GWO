# CNN Hyperparameter Optimization with GWO

Bu proje, CIFAR-10 veri seti üzerinde çalışan sabit bir HybridCNN mimarisinin hiperparametrelerini Grey Wolf Optimizer (GWO) ile optimize eden, sonuçları akademik raporlamaya uygun şekilde kaydeden deney paketidir.

## Proje Amacı

Amaç, sabit HybridCNN mimarisi üzerinde baseline deney ile GWO ile optimize edilmiş modeli karşılaştırmak ve optimizasyonun test doğruluğuna etkisini ölçmektir. Tüm arama süreci validation accuracy üzerinden yürütülür; test seti yalnızca final değerlendirmede kullanılır.

## Veri Seti

- Veri seti: CIFAR-10
- Görüntü boyutu: 32x32 RGB
- Giriş kanalı: 3
- Sınıf sayısı: 10
- Bölünme: 45.000 train, 5.000 validation, 10.000 test

## CNN Mimarisi

Kullanılan model sabit bir HybridCNN’dir:

- Stem: Conv + BatchNorm + ReLU
- Residual Block
- Inception-Lite Block
- Residual Block
- SE Attention Block
- Global Average Pooling
- Dense + Dropout + Output Layer

GWO tarafından optimize edilen hiperparametreler:

- `kernel_size`
- `base_filters`
- `dilation`
- `final_neurons`
- `dropout`
- `learning_rate`
- `batch_size`
- `se_ratio`

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
python train.py --population-size 12 --iteration-count 15 --search-epochs 6 --final-epochs 20
```

Çoklu koşu:

```bash
python train.py --runs 3 --population-size 12 --iteration-count 15 --search-epochs 6 --final-epochs 20
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
- `diversity_analysis.txt`
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
- `diversity_analysis.txt`: iterasyon bazlı unique solution ve fitness özeti
- `confusion_analysis.txt`: en çok karışan sınıflar ve sınıf bazlı doğruluk analizi
- `summary.csv`: her koşu için kısa karşılaştırma tablosu
- `search_history.csv`: her fitness değerlendirmesinin kayıtları ve değerlendirme süreleri
- `search_space.json`: arama uzayı tanımı
- `config_used.json`: deneyde kullanılan konfigürasyon
- `run.log`: deney sürecinin log kaydı

## Notlar

- Eğitim sırasında seed sabitlenir; random, numpy, torch ve CUDA tarafı aynı seed mekanizmasına bağlıdır.
- Final eğitimde train + validation birleşik veri kullanılır.
- Confusion matrix normalize edilmiş olarak kaydedilir.
- Baseline model sabit hiperparametrelerle çalışır; yalnızca GWO tarafında hiperparametreler optimize edilir.
