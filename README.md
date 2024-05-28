# Тема проекта: Распознавание лиц (Face Recognition)

## Структура проекта
```
.
├── Dockerfile
├── README.md
├── face_recognition
│   ├── __init__.py
│   ├── classification.py
│   ├── custom_dataset.py
│   ├── model.py
│   └── train_model.py
├── infer.py
├── requirements.txt
└── train.py
```

## 1. Формулировка задачи
<hr>
Проект по CV на основе моего финального проекта по курсу Deep Learning School. <br>
Целью данного проекта является создание модели по распознаванию лиц людей. <br>
Ее можно использовать,например, для верификации сотрудников компании на проходном пункте,
или отслеживание посещаемости студентов в аудитории.
<br>
<br>

## 2. Данные для обучения
<hr>

В качестве обучающей выборки будет использован датасет [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
<br>
В нем содержится несколько выборок для обучения: с 60 000 классами и 1000 класса.<br>
В данном проекте будет использован датасет с 1000 классов. Его должно быть достаточно для получения приемлемого качества модели.
<br>
<br>

## 3. Подход к моделированию:
<hr>
В данном проекте планируется использовать модель InceptionResnetV1.<br>
Модель будет написана на языке python с помощью библиотеки pytorch. <br>
Модель будет обучаться с помощью лосс функции <a href="https://arxiv.org/abs/1801.09414">CosFace</a>, которая распределяет эмбеддинги лиц по разным углам.<br>
После обучения для всех людей, которых мы хотим детектировать, будет сохранен словарь с их эмбеддингами.
<br>
<br>

## 4. Способ предсказания:
<hr>

Сам прогноз будет производиться следующим образом:
 * для каждого лица считается его средний вектор эмбеддингов по доступным изображениям и записывается в словарь
 * для прогноза конкретного лица оно подается в модель, и берется его эмбеддинг
 * считаются угол между этим вектором и всеми, записанными в словарь
 * в качестве прогноза берется вектор, имеющий наименьший угол
 * таким образом, можно легко добавлять новые лица, просто добавив его эмбеддинга словарь
 * и при этом нет необходимости заново переобучать модель

Таким образов в продакшене должна храниться готовая модель и словарь с эмбеддингами. <br>
 * модель должна быть статичной, и должен быть обеспечен лишь ее запуск
 * словарь с эмбеддингами должен иметь возможность обновления
