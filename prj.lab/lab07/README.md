# misis2024s-21-02-suchoruchenkov-m-e

# Отчет по 7 ЛР

### Цель

Изучить и реализовать метод сегментации изображений. Рассмотреть возможные варианты оценки качества сегментации с использованием эталонной разметки и реализовать функционал для этого.

### Ход работы

В ходе лабораторной работы была написана программа, которая генерирует тестовое изображение из json с заданными параметрами и производит Edge-Based сегментацию объектов изображения с последующей оценкой метрик. 

Edge-Based сегментация - этот метод основывается на том, что границы объектов в изображениях обычно соответствуют резким изменениям интенсивности, цвета или текстуры.

Используемые метрики:

Intersection over Union (IoU): Эта метрика измеряет степень перекрытия между сегментированной областью и истинной областью объекта. Значение IoU варьируется от 0 до 1, где 1 означает идеальное совпадение. IoU является одной из наиболее распространенных метрик для оценки качества сегментации.

Precision (точность): Эта метрика показывает, какая доля пикселей, отнесенных к сегментированной области, действительно принадлежит объекту. Высокое значение Precision означает, что алгоритм сегментации редко ошибочно включает пиксели фона в сегментированную область.

Recall (полнота): Эта метрика показывает, какая доля пикселей объекта была правильно отнесена к сегментированной области. Высокое значение Recall означает, что алгоритм сегментации обнаруживает большую часть объекта.

F1 Score (F-мера): позволяет оценить баланс между точностью и полнотой сегментации. Высокое значение F1 Score указывает на хорошую производительность алгоритма сегментации.

### Работа программы

Для запуска программы, нужно найти исполняемый файл в вашей директории и прописать в консоль:

```.\lab07.exe```

### Вывод

Тестовые изображения с различными параметрами генерации:

1. size - 200x200, background color = 100, blur = 2.0, noise = 3.0:

![example1](assets/example1.png "example1")

2. size - 375x375, background color = 85, blur = 2.0, noise = 3.0:

![example2](assets/example2.png "example2")

3. size - 475x475, background color = 65, blur = 2.0, noise = 5.0:

![example3](assets/example3.png "example3")

4. size - 525x525, background color = 200, blur = 4.0, noise = 5.0:

![example4](assets/example4.png "example4")

5. size - 525x525, background color = 25, blur = 4.0, noise = 5.0:

![example5](assets/example5.png "example5")

Соответствующие результаты сегментации:

1.
![example1](assets/example1.png "example1") ![result1](assets/result1.png "result1")

![quality1](assets/quality1.png "quality1")

2.
![example2](assets/example2.png "example2") ![result2](assets/result2.png "result2")

![quality2](assets/quality2.png "quality2")

3.
![example3](assets/example3.png "example3") ![result3](assets/result3.png "result3")

![quality3](assets/quality3.png "quality3")

4.
![example4](assets/example4.png "example4") ![result4](assets/result4.png "result4")

![quality4](assets/quality4.png "quality4")

5.
![example5](assets/example5.png "example5") ![result5](assets/result5.png "result5")

![quality5](assets/quality5.png "quality5")