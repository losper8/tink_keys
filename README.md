# tink_keys

Архитектура модели описанна в файле keys_tink.py 

Обучение происходило на 10 эпохах, батч сайз = 2

датасет взят из чата Huawei NLP Course 2023

обучалось в течение 3.5 часов на видеокарте rtx 3060

![Пример изображения](https://github.com/losper8/tink_keys/blob/master/photo_2023-08-31_12-26-52.jpg)
## Установка

1. Склонируйте репозиторий:

```
git clone https://github.com/losper8/tink_keys.git
```

2. Установите необходимые зависимости:

```
conda env create --file environment.yml
```
```
conda activate имя_среды
```
3. Запуск

Запустите команду для запуска проекта:
```
python inference.py
```

