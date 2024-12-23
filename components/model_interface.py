import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from .preparation_data import preparation_data

class ModelInterface:
    def __init__(self, input_shape, num_classes, batch_size,validation_split=0.2, drop_rate=0.2, learning_rate=1e-3):
        """
        Класс для работы с предобученными моделями.

        :param input_shape: Размер входных изображений (H, W, C).
        :param num_classes: Количество классов для классификации.
        :param learning_rate: Шаг обучения для оптимизатора.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.validation_split=validation_split
        self.learning_rate = learning_rate
        self.drop_rate = drop_rate
        self.model = self.build_model()


    def build_model(self):
        """
        Создание модели с предобученной EfficientNetV2B0.
        """
        print(self.input_shape)

        inputs = layers.Input(shape=(self.input_shape[0], self.input_shape[1], 3))

        base_model = EfficientNetV2B0(include_top=False, input_tensor=inputs, weights="imagenet")
        print(base_model)

        # Заморозка предобученных весов
        base_model.trainable = False

        # Замена верхних слоев
        x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.drop_rate, name="top_dropout")(x)
        outputs = layers.Dense(self.num_classes, activation="softmax", name="pred")(x)

        # Компилируем модель
        model = models.Model(inputs, outputs, name="EfficientNet")
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return model


    def unfreeze_model(self, trainable_layers=10, new_learning_rate=1e-3):
        """
        Разморозка последних слоев модели для дообучения.

        :param trainable_layers: Количество размораживаемых слоев с конца.
        :param new_learning_rate: Новый шаг обучения для оптимизатора.
        """
        for layer in self.model.layers[-trainable_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        optimizer = tf.keras.optimizers.Adam(learning_rate=new_learning_rate)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


    def prepare_data(self, path_df, img_augmentation=None):
        """
        Подготовка данных для обучения.

        :param path_df: Путь к данным.
        :param img_augmentation: Список параметров аугментации.
        :return: Тренировочный и тестовый набор данных.
        """

        train_ds, val_ds, test_ds, report = preparation_data(
            path_df,
            validation_split=self.validation_split,
            BATCH_SIZE=self.batch_size,
            IMAGE_SIZE=self.input_shape,
            img_augmentation=img_augmentation,
        )
        print('ok!')
        return train_ds, val_ds, test_ds, report


    def train_model(self, train_ds, val_ds, epochs=10):
        """
        Обучение модели.

        :param train_ds: Тренировочный набор данных.
        :param val_ds: Валидационный набор данных.
        :param epochs: Количество эпох обучения.
        """
        history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        return history


    def evaluate_model(self, test_ds):
        """
        Оценка модели на тестовых данных.

        :param test_ds: Тестовый набор данных.
        :return: Метрики оценки.
        """
        results = self.model.evaluate(test_ds)
        return results


    def predict(self, path_img):
        """
        Предсказание на новых данных.

        :param images: Набор изображений для предсказания.
        :return: Предсказанные классы.
        """
        # import matplotlib.pyplot as plt

        # Загружаем картинку
        img_SB = load_img(path_img, target_size=self.input_shape[:2])  # (height, width)
        # plt.imshow(img_SB)

        img_array_SB = keras.utils.img_to_array(img_SB) # Преобразуем картинку в тензор
        img_array_SB = keras.ops.expand_dims(img_array_SB, 0)  # Создание дополнительного измерения для батча

        # Предсказание
        predictions = self.model.predict(img_array_SB)

        # plt.title(f"Предсказание: %s\n Заболевания: {nama_img} \n Вероятность: %2.1f%%" %
        #  (CLASS_LIST[keras.ops.argmax(predictions)],
        #   keras.ops.max(predictions)*100)  ) # Вывод метки
        # plt.axis("off")
        # predictions = self.model.predict(images)
        return predictions
