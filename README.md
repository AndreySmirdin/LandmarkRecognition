# LandmarkRecognition

В этом репозитории содеражится исходный код системы по распознаванию достопримечательностей.

Структура репозитория:
* В папке `nets` должны содержатся питоновские файлы с решениями для запуска из консоли.
* В папке `src` код, который переиспользуется разными решениями.
* Помимо этого в корне находятся ноутбуки с экспериментами.

**Инcnрукции по запуску**

```
PYTHONPATH=$PYTHONPATH:`pwd`

# Для запуска тренировки:
python nets/center_loss.py -job train -train_path /mnt/hdd/1/imageData/train/russianDataCleanAdded -test_path /mnt/hdd/1/imageData/index/russianDataCleanAdded -train_again

# Для запуска классификации:
python nets/center_loss.py -job eval -train_path /mnt/hdd/1/imageData/train/russianDataCleanAdded -test_path /mnt/hdd/1/imageData/index/russianDataCleanAdded -epoch 40
```

Датасеты должны быть в виде:
```
dataset
-- class1
   -- file1.jpg
   -- file2.jpg
-- class2
   -- file2.jpg
```
Для обучения модели необходимо передать пути до датасетов с тестовыми и валидационными данными `-train_path, -test_path`
Опционально можно указать размер батча `-b` и `-train_again` чтобы обучение началось заного, а не с предыдущего сохранения.

Для классификации можно указать номер эпохи, которую загрузить. Иначе будет использоваться самое последнее сохранение.


**Другие эксперименты**:
* `SimpleClassifier.ipynb` классификация с помощью сети, без дополнительных функция ошибки.
* `CenterLoss.ipynb` использование функции ошибки Center loss.
* `ArcFace.ipynb` использование функций ошибок ArcFace loss и CosFace loss.
* `GeoSorter.ipynb` выделение достопримечательностей, относящихся к определенному географическому региону.
* `DELF.ipynb` замеры точности работы DELF на различных датасетах.
* `Backbones_charts.ipynb` построение графиков для сравнения разных архитектур сетей.


