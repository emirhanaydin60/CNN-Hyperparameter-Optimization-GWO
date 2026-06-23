# CNN Hyperparameter Optimization Research Framework

Bu proje, CIFAR-10 üzerinde çalışan sabit HybridCNN mimarisiyle GWO, PSO, WOA ve RAO algoritmalarını aynı koşullarda karşılaştıran araştırma çerçevesidir.

## Proje Amacı

Amaç, tüm algoritmaları aynı veri bölünmesi, aynı seed stratejisi, aynı search space, aynı training stratejisi ve aynı evaluation prosedürü ile çalıştırarak adil bir karşılaştırma üretmektir.

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

Metaheuristic algoritmalar tarafından optimize edilen hiperparametreler:

- `shared_conv_kernel_size`
- `base_filters`
- `dilation`
- `final_neurons`
- `dropout`
- `learning_rate`
- `batch_size`
- `se_ratio`

## Metaheuristic Algoritmalar

Desteklenen algoritmalar:

- GWO
- PSO
- WOA
- RAO

Hepsi aynı `BaseOptimizer` arayüzüne bağlıdır. Varsayılan adil karşılaştırma ayarı `population_size = 8` ve `iteration_count = 15` şeklindedir.

## Konfigürasyon Yönetimi

Tüm deney parametreleri merkezi olarak `config.py` içinde tutulur veya JSON üzerinden override edilebilir.

Önemli parametreler:

- `optimizers`
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

Tek bir algoritma çalıştırmak için:

```bash
python train.py --optimizer gwo
```

Tüm algoritmaları çalıştırmak için:

```bash
python train.py
```

JSON konfigürasyon ile:

```bash
python train.py --config my_config.json
```

İkinci aşama analiz için:

```bash
python analyze_results.py --results-dir results --dataset cifar10
```

## Sonuç Dosyaları

Çalışma sonunda sonuçlar `results/CIFAR10/<ALGORITHM>/run_XX/` altında tutulur. Her koşuda `summary.json` üretilir; algoritma özetleri ise `results/CIFAR10/<ALGORITHM>/summary.json` içinde saklanır.

Karşılaştırma çıktıları `results/CIFAR10/comparison/` içinde üretilir:

- `overall_summary.json`
- `performance_comparison.csv`
- `runtime_comparison.csv`
- `stability_comparison.csv`
- `ranking_table.csv`
- `accuracy_comparison.png`
- `runtime_plot.png`
- `convergence_comparison.csv`
- `convergence_plot.png`
- `boxplot_accuracy.png`
- `boxplot_runtime.png`

## Dosya Açıklamaları

- `summary.json`: tek koşunun tüm metrikleri, süreleri, en iyi hiperparametreleri ve convergence geçmişi
- `results/CIFAR10/<ALGORITHM>/summary.json`: algoritma bazlı çoklu-run özeti
- `results/CIFAR10/comparison/overall_summary.json`: tüm algoritmalar için birleştirilmiş özet
- `performance_comparison.csv`: tüm koşular için satır bazlı performans tablosu
- `runtime_comparison.csv`: algoritma bazlı runtime özeti
- `stability_comparison.csv`: mean/std stabilite özeti
- `ranking_table.csv`: test doğruluğuna göre sıralama

`population_size = 8` ve `iteration_count = 15` varsayılan olarak korunmuştur; bu, mevcut GWO 8 kurt x 15 iterasyon koşusuyla adil karşılaştırmayı sağlar.

Yeni algoritma eklemek için `optimizers/` altına yeni sınıf ekleyip `config.py` içindeki `optimizers` listesine dahil etmek yeterlidir.

## Notlar

- Eğitim sırasında seed sabitlenir; random, numpy, torch ve CUDA tarafı aynı seed mekanizmasına bağlıdır.
- Final eğitimde train + validation birleşik veri kullanılır.
- Confusion matrix normalize edilmiş olarak kaydedilir.
- Baseline model sabit hiperparametrelerle çalışır; yalnızca GWO tarafında hiperparametreler optimize edilir.
