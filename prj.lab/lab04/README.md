# misis2024s-21-02-suchoruchenkov-m-e

# Отчет по 4 ЛР

### Цель

Изучить и реализовать различные методы бинаризации изображений. Рассмотреть возможные варианты создания GUI для измененния параметров бинаризации, а также реализовать функционал для детектирования объектов изображения с возможностью оценки качества детектирования.

### Ход работы

В ходе выполнения лабораторной работы была написана программа, которая генерирует тестовое изображение с заданными параметрами и реализует несколько вариантов бинаризации. Также был предусмотрен режим визуальной настройки параметров бинаризации через GUI, реализован функционал для детектирования объектов на изображении и для оценки качества работы алгоритмов.

1. Изображение с параметрами генерации countCircles=10, minRadius=5, maxRadius=15, minContrast=50, maxContrast=255, blur=4, noise=4:

![example](assets/example.png "example")

Пример работы GUI:

Бинаризация исходного изображения методом Бернсена:

![bernsen](assets/bernsen.png "bernsen")

Детектирование при параметрах radius = 14, CMin = 40:

![bernsenDetection](assets/bernsenDetection.png "bernsenDetection")

![bernsenkQuality](assets/bernsenQuality.png "bernsenQuality")

Пример работы GUI:

Бинаризация исходного изображения методом Ниблека:

![niblack](assets/niblack.png "niblack")

Детектирование при параметрах Radius = 24, k Value = 9, d Value = 18:

![niblackDetection](assets/niblackDetection.png "niblackDetection")

![niblackQuality](assets/niblackQuality.png "niblackQuality")

2. Изображение с параметрами генерации countCircles=5, minRadius=5, maxRadius=15, minContrast=25, maxContrast=200, blur=3, noise=5:

![example2](assets/example2.png "example2")

Детектирование при параметрах radius = 14, CMin = 40, бинаризация Бернсена:

![bernsenDetection2](assets/bernsenDetection2.png "bernsenDetection2")

![bernsenkQuality2](assets/bernsenQuality2.png "bernsenQuality2")

Детектирование при параметрах Radius = 8, k Value = 9, d Value = 7, бинаризация Ниблека:

![niblackDetection2](assets/niblackDetection2.png "niblackDetection2")

![niblackQuality2](assets/niblackQuality2.png "niblackQuality2")

3. Изображение с параметрами генерации countCircles=7, minRadius=5, maxRadius=20, minContrast=40, maxContrast=180, blur=3, noise=7:

![example3](assets/example3.png "example3")

Детектирование при параметрах radius = 10, CMin = 30, бинаризация Бернсена:

![bernsenDetection3](assets/bernsenDetection3.png "bernsenDetection3")

![bernsenkQuality3](assets/bernsenQuality3.png "bernsenQuality3")

Детектирование при параметрах Radius = 20, k Value = 9, d Value = 23, бинаризация Ниблека:

![niblackDetection3](assets/niblackDetection3.png "niblackDetection3")

![niblackQuality3](assets/niblackQuality3.png "niblackQuality3")

### Работа консольного приложения

Для запуска консольного приложения, нужно найти исполняемый файл в вашей директории и прописать в консоль:

```.\lab04.exe <countCircles> <minRadius> <maxRadius> <minContrast> <maxContrast> <blur> <noise> <method("bernsen", "niblack", "detection")>```

### Вывод

Как можем заметить, объекты детектируются лучше, при бинаризации методом Ниблека.