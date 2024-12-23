import os
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow import data as tf_data

## Отфильтровать поврежденые изображения
def filter_damaged_images(path):
  num_skipped = 0 # счетчик поврежденных файлов
  for folder_name in os.listdir(path): # перебираем папки
      folder_path = os.path.join(path, folder_name) # склеиваем путь
      for fname in os.listdir(folder_path): # получаем список файлов в папке
          fpath = os.path.join(folder_path, fname) # получаем путь до файла
          try:
              fobj = open(fpath, "rb") # пытаемся открыть файл для бинарного чтения (rb)
              is_jfif = b"JFIF" in fobj.peek(10) # получаем первые 10 байт из файла и ищем в них бинарный вариант строки JFIF
          finally:
              fobj.close() # Закрываем файл

          if not is_jfif: # Если не нашли JFIF строку
              # Увеличиваем счетчик
              num_skipped += 1
              # Удаляем поврежденное изображение
              os.remove(fpath)

  print(f"Удалено изображений: {num_skipped}")

# example us
# filter_damaged_images(f'{path}/brain-tumor-mri-dataset')

#img_augmentation[0] Вращение на 15%-base
#img_augmentation[1] Зум на 10%-base
#img_augmentation[2] Яркость на 20%-base
#img_augmentation[3,4] Сдвиг на 10%-base
#img_augmentation[5] Отражение
#img_augmentation[6] Контрастность на 10%

img_augmentation = [0.15, 0.1, 0.2, 0.1,0.1, 0, 0.1]

def preparation_data(path_data,
                  validation_split=0.2,
                  BATCH_SIZE=32,
                  IMAGE_SIZE=(300, 300),
                  img_augmentation = None
                  ):

  report = {}
  report ['BATCH_SIZE'] = BATCH_SIZE
  report ['IMAGE_SIZE'] = IMAGE_SIZE

  train_ds, val_test_dataset = keras.utils.image_dataset_from_directory(
      path_data, # путь к папке с данными
      validation_split=validation_split, # 70% for trening and 30% for time set (temp)
      subset="both", # указываем, что необходимо вернуть кортеж из обучающей и проверочной выборок ("training", "validation" или "both")
      seed=42,  # воспроизводимость результата генерации (результаты с одинаковым числом - одинаковы),
      shuffle=True, # перемешиваем датасет
      image_size=IMAGE_SIZE, # размер генерируемых изображений
      batch_size=BATCH_SIZE, # размер мини-батча
    )
  dataset_size = tf.data.experimental.cardinality(val_test_dataset).numpy()

  # Деление (50% на 50%)
  test_size = dataset_size // 2
  test_ds = val_test_dataset.take(test_size)
  control_ds = val_test_dataset.skip(test_size)

  # print(f"Количество батчей в train_ds: {len(train_ds)}")
  # print(f"Количество батчей в test_ds: {len(test_ds)}")
  # print(f"Количество батчей в control_ds: {len(control_ds)}")
  report['train_ds'] = len(train_ds)
  report['test_ds'] = len(test_ds)
  report['control_ds'] = len(control_ds)


  # Определяем список имен классов
  CLASS_LIST = sorted(os.listdir(path_data))

  # Определяем количества классов
  CLASS_COUNT = len(CLASS_LIST)

  # Вывод результата
  # print(f'Количество классов: {CLASS_COUNT}')
  # print(f'Метки классов: {CLASS_LIST}')
  report['CLASS_COUNT'] = CLASS_COUNT
  report['CLASS_LIST'] = CLASS_LIST
  
  # с augmentation
  if img_augmentation:
    img_augmentation_layers = [
        layers.RandomRotation(img_augmentation[0]),  # Вращение на 15%
        layers.RandomZoom(img_augmentation[1]),  # Зум на 10%
        layers.RandomBrightness(img_augmentation[2]),  # Яркость на 20%
        layers.RandomTranslation(img_augmentation[3], img_augmentation[4]),  # Сдвиг на 10%
        layers.RandomFlip(img_augmentation[5]),  # Отражение
        layers.RandomContrast(img_augmentation[6])  # Контрастность на 10%
    ]

    def img_augmentation_func(images):
        for layer in img_augmentation_layers:
            images = layer(images)
        return images
  
  else:
    def img_augmentation_func(images):
        return images

  # def img_augmentation(images):
  #   for layer in img_augmentation_layers:
  #       images = layer(images)
  #   return images

    # Применяем img_augmentation к обучающей выборке
  train_ds = train_ds.map(
      lambda img, label: (img_augmentation_func(img), tf.one_hot(tf.cast(label, tf.int32), CLASS_COUNT)),
      num_parallel_calls=tf_data.AUTOTUNE
  )
  test_ds = test_ds.map(
      lambda img, label: (img, tf.one_hot(tf.cast(label, tf.int32), CLASS_COUNT)),
      num_parallel_calls=tf_data.AUTOTUNE
  )
  control_ds = control_ds.map(
      lambda img, label: (img, tf.one_hot(tf.cast(label, tf.int32), CLASS_COUNT)),
      num_parallel_calls=tf_data.AUTOTUNE
  )

  # Предварительная выборка примеров
  train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
  test_ds = test_ds.prefetch(tf_data.AUTOTUNE)
  control_ds = control_ds.prefetch(tf_data.AUTOTUNE)


  return train_ds, test_ds, control_ds, report