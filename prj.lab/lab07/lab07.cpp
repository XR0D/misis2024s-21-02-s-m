#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

struct Image {
    cv::Mat src_image;
    cv::Mat result;
    cv::Mat result_for_estimation;
    cv::FileStorage json;
    int weight;
    int height;
    int color;
    int blur;
    double noise;
};

// Функция для добавления гауссовского шума к изображению
void generateNoise(Image& img) {
    double mean = 0;
    cv::Mat noise(img.src_image.size(), CV_8SC1);
    cv::randn(noise, cv::Scalar{ mean }, cv::Scalar{ img.noise });
    img.src_image += noise;
}

// Функция для генерации изображения на основе данных из JSON-файла
void generateImage(Image& img) {
    img.json.open("../prj.lab/lab07/input.json", cv::FileStorage::READ);
    if (!img.json.isOpened()) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return;
    }

    cv::FileNode data = img.json["data"];
    cv::FileNode objects = data["objects"];
    cv::FileNode background = data["background"];
    cv::FileNode size_of_image = background["size"];

    img.weight = (int)size_of_image[0];
    img.height = (int)size_of_image[1];
    img.color = (int)background["color"];
    img.blur = (int)background["blur"];
    img.noise = (int)background["noise"];

    img.src_image = cv::Mat(img.weight, img.height, CV_8U, cv::Scalar(img.color));

    // Рисование кругов на изображении
    for (cv::FileNodeIterator it = objects.begin(); it != objects.end(); ++it) {
        cv::FileNode obj = *it;
        cv::FileNode p = obj["p"];

        int x = (int)p[0];
        int y = (int)p[1];
        int radius = (int)p[2];
        int contrast = (int)obj["c"];

        cv::Point center(x, y);
        cv::circle(img.src_image, center, radius, contrast, cv::FILLED);
    }

    // Размытие изображения с помощью гауссова фильтра
    cv::Mat img_for_bluring;
    img.src_image.copyTo(img_for_bluring);
    cv::GaussianBlur(img_for_bluring, img.src_image, cv::Size(5, 5), img.blur);

    // Добавление шума к изображению
    generateNoise(img);

    img.json.release();
}

// Функция для оценки качества сегментации
void results(Image& img) {
    cv::Mat resultBW;
    cv::cvtColor(img.result, resultBW, cv::COLOR_BGR2GRAY);

    // Вычисление пересечения и объединения областей
    cv::Mat intersection = (img.src_image & resultBW);
    cv::Mat unionArea = (img.src_image | resultBW);

    // Вычисление площадей пересечения и объединения
    double intersectionArea = cv::countNonZero(intersection);
    double unionAreaValue = cv::countNonZero(unionArea);

    // Вычисление метрик качества сегментации
    double IoU = intersectionArea / unionAreaValue; // Intersection over Union
    double precision = intersectionArea / cv::countNonZero(resultBW); // Precision
    double recall = intersectionArea / cv::countNonZero(img.src_image); // Recall
    double f1Score = 2 * (precision * recall) / (precision + recall); // F1 Score

    std::cout << "Intersection over Union: " << IoU << std::endl;
    std::cout << "Precision: " << precision << std::endl;
    std::cout << "Recall: " << recall << std::endl;
    std::cout << "F1: " << f1Score << std::endl;
}

// Функция для выполнения сегментации изображения
void segmentation(Image& img) {
    cv::Mat edges, thresh;

    // Вычисление порога по методу Оцу
    double otsu_threshold = cv::threshold(img.src_image, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Определение порогов для фильтра Кэнни
    double low_threshold = otsu_threshold * 0.5;
    double high_threshold = otsu_threshold * 1.5;

    // Применение фильтра Кэнни для выделения границ
    cv::Canny(img.src_image, edges, low_threshold, high_threshold);

    // Поиск контуров на изображении границ
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Преобразование исходного изображения в BGR для рисования контуров
    cv::cvtColor(img.src_image, img.result, cv::COLOR_GRAY2BGR);

    // Создание изображения для оценки качества сегментации
    img.result_for_estimation = cv::Mat(img.weight, img.height, CV_8UC3, cv::Scalar(img.color));

    // Цвет для рисования контуров
    cv::Scalar color(0, 255, 0); // Зеленый

    // Рисование контуров на изображениях
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::drawContours(img.result, contours, static_cast<int>(i), color, cv::FILLED);
        cv::drawContours(img.result_for_estimation, contours, static_cast<int>(i), color, cv::FILLED);
    }

    results(img);
}

int main() {
    Image img;

    // Генерация изображения
    generateImage(img);
    cv::imwrite("example5.png", img.src_image); // Сохранение исходного изображения

    // Сегментация изображения
    segmentation(img);
    cv::imwrite("result5.png", img.result); // Сохранение результата сегментации


    cv::Mat concated, bgr_source_image;
    cv::cvtColor(img.src_image, bgr_source_image, cv::COLOR_GRAY2BGR);
    cv::hconcat(bgr_source_image, img.result, concated);
    cv::imshow("Result", concated);
    cv::waitKey(0);

    return 0;
}