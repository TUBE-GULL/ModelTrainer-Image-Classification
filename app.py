# Импортируем библиотеку Streamlit
import streamlit as st
import pandas as pd
import keras
import numpy as np
from tensorflow.keras.models import save_model, load_model
from components.preparation_data import filter_damaged_images, preparation_data
from components.model_interface import ModelInterface
from PIL import Image

# Для предсказания
path_model = 'models/'

db_model = {
    'BaseModel': ((300, 300), ['glioma', 'meningioma', 'notumor', 'pituitary']),
}

# Заголовок страницы
st.title("Model Classification of Brain Tumors")
model_name = st.selectbox('Выберите модель для предсказания:', list(db_model.keys()))
uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])


if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    # Отображаем изображение
    st.image(image, caption="Загруженное изображение для определения диагноза", use_column_width=True)
    st.success("Изображение успешно загружено!")

    # Функция для предсказания
    def img_predictions(uploaded_image):
        # Отображаем индикатор выполнения
        with st.spinner('Модель работает, пожалуйста подождите...'):
            # Преобразуем изображение в формат RGB (удаляем альфа-канал, если есть)
            if uploaded_image.mode != 'RGB':
                    uploaded_image = uploaded_image.convert('RGB')

            # Изменяем размер изображения под модель
            img_resized = uploaded_image.resize(db_model[model_name][0])
            # Преобразуем картинку в тензор
            img_array_SB = keras.utils.img_to_array(img_resized)
            img_array_SB = keras.ops.expand_dims(img_array_SB, 0) # Создание дополнительного измерения для батча

            # Загружаем модель
            model = load_model(path_model + model_name + '.keras')
            # Получаем предсказания
            predictions = model.predict(img_array_SB)

            # Получаем метку и вероятность класса с максимальной вероятностью
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class = db_model[model_name][1][predicted_class_index]
            probability = np.max(predictions) * 100

            # Выводим результат
            st.success("Модель успешно завершила предсказание!")
            st.write(f"Предсказанный диагноз: {predicted_class}")
            st.write(f"Вероятность: {probability:.2f}%")

    # Вызываем функцию предсказания
    img_predictions(image)
else:
    st.info("Пожалуйста, загрузите изображение для предсказания.")



# код для обучения модели !
with st.expander("Обучения модели на своих данных"):

    # Заголовок страницы
    st.title("Обучения модели на своих данных")

    # Ввод названия модели
    st.write("Введите название модели:")
    name_model = st.text_input("Название модели:", "")

    # Проверка и обработка ввода для названия модели
    if name_model:
        st.write(f"Вы ввели название модели: {name_model}")

    # Ввод количества эпох
    # st.write("Введите количество эпох:")
    # epochs = st.text_input("Количество эпох:", "")


    # Обработка ввода количества эпох (преобразование в целое число)
    epochs = st.slider('Выберите количество эпох:', min_value=1, max_value=200, value=1, step=1)

    # if epochs:
    #     try:
    #         epochs = int(epochs)  # Преобразуем в целое число
    #         st.write(f"Вы ввели количество эпох: {epochs}")
    #     except ValueError:
    #         st.error("Введите корректное количество эпох (целое число).")

    # # Ввод размера батча
    st.write("Введите размер батча:")
    batch_size = st.text_input("Размер батча:", "")

    # Обработка ввода для размера батча
    if batch_size:
        try:
            batch_size = int(batch_size)
            st.write(f"Вы ввели размер батча: {batch_size}")
        except ValueError:
            st.error("Введите корректный размер батча (целое число).")

    # Ввод количества классов
    st.write("Введите количество классов в ваших данных:")
    class_count = st.text_input("Количество классов:", "")

    # Обработка ввода для количества классов
    if class_count:
        try:
            class_count = int(class_count)
            st.write(f"Вы ввели количество классов: {class_count}")
        except ValueError:
            st.error("Введите корректное количество классов (целое число).")

    # Ввод размера изображения
    st.write("Введите размер изображения для обучения:")
    image_size_horizontal = st.text_input("Размер изображения в ширену:", "")

    # Обработка ввода для размера изображения (преобразование в целое число)
    if image_size_horizontal:
        try:
            image_size = int(image_size_horizontal)
            st.write(f"Вы ввели размер изображения: {image_size}")
        except ValueError:
            st.error("Введите корректный размер изображения (целое число).")

    image_size_vertical = st.text_input("Размер изображения в горизонталь:", "")

    # Обработка ввода для размера изображения (преобразование в целое число)
    if image_size_vertical:
        try:
            image_size = int(image_size_vertical)
            st.write(f"Вы ввели размер изображения: {image_size}")
        except ValueError:
            st.error("Введите корректный размер изображения (целое число).")
    # Ввод пропорции разделения данных
    st.write("Введите, как разделить данные:")
    validation_split = st.slider("Как разделить данные (например, 80/20(0.2) или 70/30 (0.3)):", min_value=0.05, max_value=0.50, value=0.2, step=0.05)
    
    
    # Обработка ввода для разделения данных
    # if validation_split:
    #     try:
    #         # Разделяем строку на части и проверяем корректность
    #         split_values = validation_split.split('/')
    #         if len(split_values) == 2:
    #             train_split = int(split_values[0])
    #             val_split = int(split_values[1])
    #             st.write(f"Вы ввели пропорцию для разделения данных: {train_split}% для тренировки и {val_split}% для валидации")
    #         else:
    #             st.error("Введите корректную пропорцию в формате 'X/Y'.")
    #     except ValueError:
    #         st.error("Введите корректную пропорцию разделения данных.")
      

    # Введение и описание для пользователя

    # Чекбокс для активации/деактивации параметров
    activate_augmentation = st.checkbox('Активировать настройки аугментации изображений')

    if activate_augmentation:
        # st.write("Настройка параметров аугментации изображений:")
        
        # Если чекбокс активен, показываем параметры аугментации
        rotation = st.slider('Вращение (в нормализованном диапазоне от 0 до 1):', min_value=0.0, max_value=1.0, value=0.15, step=0.01)
        # Преобразуем нормализованное значение в угол в градусах
        rotation_angle = rotation * 360  # Если максимальное значение для вращения 360 градусов
        st.write(f"Угол вращения: {rotation_angle} градусов")
        
        zoom = st.slider('Зум:', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        brightness = st.slider('Яркость:', min_value=0.0, max_value=2.0, value=1.2, step=0.01)
        horizontal_shift = st.slider('диапазон сдвига по горизонтали:', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        vertical_shift = st.slider('диапазон сдвига по вертикали:', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        flip = st.checkbox('Отражение изображения', value=False)
        contrast = st.slider('Контрастность:', min_value=0.0, max_value=2.0, value=1.1, step=0.01)
        
        img_augmentation = [rotation, zoom, brightness, horizontal_shift, vertical_shift, flip, contrast]
    else:
        # Если чекбокс не активен, показываем сообщение
        st.write("Параметры аугментации изображений деактивированы.")
        img_augmentation = None
 
    import os
    import shutil
    import zipfile
    import streamlit as st
    from tensorflow.keras.models import save_model
    import pandas as pd
    
    # Основной функционал
    if __name__ == "__main__":
        # Загрузка ZIP архива
        uploaded_file = st.file_uploader("Загрузите ZIP-архив с данными", type=["zip"])
    
        if uploaded_file is not None:
            # Создание временной папки для распаковки архива
            temp_dir = "uploaded_data"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
    
            # Распаковка архива
            try:
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                    st.write(f"Архив успешно распакован в папку: {temp_dir}")
    
                # Список файлов в папке
                files = os.listdir(temp_dir)
                st.write("Содержимое архива:")
                st.write(files)
    
                # Пример обработки CSV файлов (если они есть)
                for file in files:
                    file_path = os.path.join(temp_dir, file)
                    if file.endswith(".csv"):
                        df = pd.read_csv(file_path)
                        st.write(f"Данные из файла {file}:")
                        st.dataframe(df)
    
            except zipfile.BadZipFile:
                st.error("Ошибка: Некорректный ZIP файл.")
                shutil.rmtree(temp_dir, ignore_errors=True)
                st.stop()
    
        # Настройки модели
        activate_unfreeze_model = st.checkbox('Разморозка последних слоев модели для дообучения:')
        unfreeze_model = st.slider('Разморозить слоев:', 0, 20, 1) if activate_unfreeze_model else 0
    
        # Кнопка для обучения
        if st.button("Начать обучение"):
            if uploaded_file is None:
                st.error("Загрузите данные для обучения!")
            else:
                try:
                    # Сохраняем настройки в словарь
                    settings = {
                        'name_model': str(name_model),
                        'epochs': int(epochs),
                        'BATCH_SIZE': int(batch_size),
                        'CLASS_COUNT': int(class_count),
                        'IMAGE_SIZE': (int(image_size_horizontal),int(image_size_vertical)),
                        'validation_split': validation_split,
                        'img_augmentation': img_augmentation,
                    }

                    
                    # Инициализация модели
                    new_model = ModelInterface(settings['IMAGE_SIZE'], settings['CLASS_COUNT'], settings['BATCH_SIZE'], validation_split=settings["validation_split"])
    
                    # Разморозка слоев модели
                    if activate_unfreeze_model:
                        new_model.unfreeze_model(trainable_layers=unfreeze_model)
    
                    # Подготовка данных и обучение
                    train_ds, test_ds, control_ds, report = new_model.prepare_data(temp_dir, img_augmentation=settings["img_augmentation"])
                    print('date ok!')
                    history = new_model.train_model(train_ds, test_ds, epochs=settings["epochs"])
                    result = new_model.evaluate_model(control_ds)
    
                    # Сохранение модели
                    path_model = './models/'
                    if not os.path.exists(path_model):
                        os.makedirs(path_model)
                    save_path = os.path.join(path_model, f"{settings['name_model']}.keras")
                    save_model(new_model, save_path)
    
                    # Вывод результатов
                    st.success("Обучение завершено!")
                    st.write(f"Точность: {result[0]:.2f}, Потери: {result[1]:.2f}")
    
                except Exception as e:
                    st.error(f"Произошла ошибка: {e}")
    
                # Удаление временной папки
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    st.info("Временные данные удалены.")
    
    
    