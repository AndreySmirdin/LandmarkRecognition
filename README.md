# LandmarkRecognition

В этом репозитории содеражится исходный код системы по распознаванию достопримечательностей.

Структура репозитория:
* В папке `nets` должны содержатся питоновские файлы с решениями для запуска из консоли.
* В папке `src` код, который переиспользуется разными решениями.
* Помимо этого в корне находятся ноутбуки с экспериментами.

Инструкции по запуску решения из консоли:

```
PYTHONPATH=$PYTHONPATH:`pwd`

# Для запуска тренировки:
python nets/center_loss.py -job train -train_path /mnt/hdd/1/imageData/train/russianDataCleanAdded -test_path /mnt/hdd/1/imageData/index/russianDataCleanAdded -train_again

# Для запуска классификации:
python nets/center_loss.py -job eval -train_path /mnt/hdd/1/imageData/train/russianDataCleanAdded -test_path /mnt/hdd/1/imageData/index/russianDataCleanAdded -epoch 40
```

Помимо этого в репозиториии содержатся следующие файлы с экспериментами:
* `SimpleClassifier.ipynb` классификация с помощью сети, без дополнительных функция ошибки.
* `CenterLoss.ipynb` использование функции ошибки Center loss.
* `ArcFace.ipynb` использование функций ошибок ArcFace loss и CosFace loss.
* `GeoSorter.ipynb` выделение достопримечательностей, относящихся к определенному географическому региону.
* `DELF.ipynb` замеры точности работы DELF на различных датасетах.
* `Backbones_charts.ipynb` построение графиков для сравнения разных архитектур сетей.


