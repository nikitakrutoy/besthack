# Best Hack

[Презентация](https://slides.com/gpov1/deck/fullscreen)

Краткое описание:

Апроксимируем зависимоть Fa от v линейной регрессией.

Апроксимируем зависимоть Wx и Wz от Y методом опроных векторов с rbf-ядром.

Выполнеяем итеративный алгоритм:

  1. Делаем шаг по направлению скорости
  2. Пересчитываем ускорением
  3. Пересчитываем V
  4. Повторяем 1-3 пока Y > 0
 
Получаем черную коробку, которую можно оптимизировать при помощи байесовской оптимизации по  x, z и alpha


Запуск
```
pip install -r requirements.txt
```
Для обучения
```
python model.py --mode 0 --input_path ./data --output_path ./models
```
Для предсказания
```
python model.py x, z, alpha --mode 1 --input_path ./models/model.pickle --mass 100 --y 1400 --step_size 100
```

Для оптимизации
```
python model.py --mode 2 --input_path ./models/model.pickle --mass 100 --y 1400 --step_size 100
```


