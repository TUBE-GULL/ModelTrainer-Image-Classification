<h1 align="center">ModelTrainer-Image-Classification</h1>

<h2 align="center">Used Libraries</h2>
<div align="center">
 <a href="https://www.python.org" target="_blank" rel="noreferrer" style="display: inline-block;"> 
   <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="60" height="60"/>
 </a>

 <a href="https://numpy.org/" target="_blank" rel="noreferrer" style="display: inline-block;">
   <img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original-wordmark.svg" title="Numpy" alt="Numpy" width="60" height="60"/> 
 </a>

 <a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer" style="display: inline-block;"> 
   <img src="https://github.com/devicons/devicon/blob/master/icons/tensorflow/tensorflow-original.svg" title="tensorflow" alt="tensorflow" width="60" height="60"> 
 </a>

 <a href="https://keras.io/" target="_blank" rel="noreferrer" style="display: inline-block;">
   <img src="https://github.com/devicons/devicon/blob/master/icons/keras/keras-original.svg" title="keras" alt="keras" width="60" height="60"> 
 </a>

 <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer" style="display: inline-block;">
   <img src="https://github.com/devicons/devicon/blob/master/icons/pandas/pandas-original.svg" title="Pandas" alt="Pandas" width="60" height="60"/> 
 </a>

 <a href="https://streamlit.io/" target="_blank" rel="noreferrer" style="display: inline-block;">
   <img src="https://user-images.githubusercontent.com/7164864/217935870-c0bc60a3-6fc0-4047-b011-7b4c59488c91.png" title="streamlit" alt="streamlit" width="60" height="60"/> 
 </a>

</div>

<h2 align="center">Instructions</h2>

```bash
"/start" - streamlit run streamlit_app.py 
```

## Installation of Dependencies

You can install the required dependencies either manually or using the `requirements.txt` file.

###  Install Manually
```bash
pip install tensorflow
pip install numpy
pip install pandas
pip install streamlit

````

### Install using the `requirements.txt` file
```bash
pip install -r requirements.txt

````







## Описание программы
Это веб-приложение на базе Streamlit, предназначенное для загрузки изображений, их классификации с помощью нейронных сетей и обучения собственных моделей на пользовательских данных. Программа позволяет:

1. Классификация изображений:

    - Загружать изображения в форматах PNG, JPG, JPEG.
    - Выбирать одну из заранее обученных моделей (например, для классификации опухолей мозга).
    - Получать предсказания с указанием вероятности для каждого класса (например, glioma, meningioma, pituitary и т.д.).
    - Просматривать результат предсказания на загруженном изображении.

2. Обучение модели на собственных данных:

    - Загружать ZIP-архив с данными, содержащими изображения для обучения.
    - Настроить параметры модели, включая количество эпох, размер батча, количество классов и размеры изображений.
    - Использовать аугментацию данных, включая вращение, масштабирование, изменение яркости, сдвиги по горизонтали и вертикали и другие.
    - Разморозить последние слои модели для дообучения.
    - Обучить модель с использованием переданных данных и настроек, а затем сохранить обученную модель в файл.
3. Визуализация данных:

    - Отображение изображений с их результатами классификации.
    - Вывод статистики об обучении модели (точность, потери).
    - Поддержка обучения и тестирования модели с возможностью проверки результата на контролирующем наборе данных.




## Ключевые компоненты:
1. Модель интерфейса:
    - Использует класс ModelInterface, который включает в себя методы для подготовки данных, создания и обучения модели, а также для оценки её качества.

2. Предсказание:
    - После загрузки изображения, модель классифицирует его, выводя прогнозируемый класс и вероятность этого прогноза.

3. Обучение:
    - Позволяет обучить модель с нуля, включая настройку гиперпараметров, а также применение аугментации данных для улучшения качества модели.

4. Работа с архивами:
    - Поддержка загрузки и обработки данных из ZIP-архивов.

5. Использование моделей:
    - Приложение поддерживает выбор различных обученных моделей и их использование для предсказаний.




## Инструкции по использованию:
1. Загрузите изображение для классификации с помощью кнопки "Загрузите изображение".
2. Выберите модель для предсказания из выпадающего списка.
3. Нажмите на кнопку "Начать обучение", чтобы обучить модель на собственных данных, настроив параметры обучения, включая количество эпох, размер батча и другие.
4. Загрузите архив с данными (например, изображения и метки) для обучения.
5. Подтвердите использование аугментации данных для улучшения качества модели.




## Ожидаемые результаты:
- После загрузки изображения и выполнения предсказания, вы получите метку класса и вероятность для данного изображения.
- После обучения модели, вы сможете увидеть точность и потери на тестовых данных, а также сохранить обученную модель для дальнейшего использования.



Эта программа предназначена для пользователей, которые хотят классифицировать изображения с использованием нейронных сетей или обучить собственные модели для решения задач классификации.



