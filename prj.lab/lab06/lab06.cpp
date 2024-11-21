#include <iostream>
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <limits>
#include <algorithm>

std::vector<cv::Vec3f> groundTruths; // Вектор для хранения истинных значений

// Функция для бинаризации изображения методом Ниблака
cv::Mat niblackBinary(const cv::Mat& image, const int radius, const double k, const int d) {
    cv::Mat binary;

    constexpr int max = static_cast<int>(std::numeric_limits<uchar>::max());
    cv::Mat kernel{ cv::Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_32FC1) };
    cv::circle(kernel, cv::Point{ radius, radius }, radius, cv::Scalar{ 1.0f }, -1);
    kernel /= cv::sum(kernel)[0]; // Нормализация ядра
    cv::Mat tmp{ image.clone() };
    tmp.convertTo(tmp, CV_32FC1, 1.0 / max);
    cv::Mat mean{ tmp.clone() };
    cv::filter2D(tmp, mean, -1, kernel); // Среднее значение
    cv::multiply(tmp, tmp, tmp);
    cv::Mat meanSq{ tmp.clone() };
    cv::filter2D(tmp, meanSq, -1, kernel); // Среднее значение квадратов
    cv::multiply(mean, mean, tmp);
    cv::sqrt(meanSq - tmp, tmp); // Стандартное отклонение
    image.convertTo(binary, CV_32FC1, 1.0 / max);
    binary = binary > (mean + k * tmp + static_cast<double>(d) / max); // Бинаризация
    binary.convertTo(binary, CV_8UC1, max);

    return binary;
}

// Функция для оценки детекций
void evaluateDetections(const std::vector<cv::Vec3f>& detections, const std::vector<cv::Vec3f>& groundTruths, double iouThreshold, int& TP, int& FP, int& FN) {
    TP = 0; FP = 0; FN = 0;

    std::vector<bool> matched(groundTruths.size(), false);

    for (const auto& detection : detections) {
        cv::Point detectCenter(cvRound(detection[0]), cvRound(detection[1]));
        int detectRadius = cvRound(detection[2]);

        for (size_t i = 0; i < groundTruths.size(); ++i) {
            const auto& truth = groundTruths[i];
            cv::Point truthCenter(cvRound(truth[0]), cvRound(truth[1]));
            int truthRadius = cvRound(truth[2]);

            double distance = cv::norm(truthCenter - detectCenter);
            double radiusSum = truthRadius + detectRadius;

            // Вычисляем Intersection over Union (IoU)
            double iou = (distance <= radiusSum) ? 1.0 : 0.0;

            if (iou >= iouThreshold && !matched[i]) {
                TP++;
                matched[i] = true; // Отметить, что это истинное значение уже найдено
                break; // Прерываем, когда находим соответствие для текущей детекции
            }
        }
    }
    FP = detections.size() - TP;
    FN = groundTruths.size() - std::count(matched.begin(), matched.end(), true);
}

// Функция для анализа FROC
void frocAnalysis(const std::vector<cv::Vec3f>& detections, const std::vector<cv::Vec3f>& groundTruths, const std::vector<double>& thresholds) {
    std::vector<int> TPCounts;
    std::vector<int> FPCounts;

    for (double threshold : thresholds) {
        int TP, FP, FN;
        evaluateDetections(detections, groundTruths, threshold, TP, FP, FN);
        TPCounts.push_back(TP);
        FPCounts.push_back(FP);
    }

    // Вывод результатов для FROC
    std::cout << "FROC Analysis Results:" << std::endl;
    std::cout << "Threshold\tTP\tFP" << std::endl;
    for (size_t i = 0; i < thresholds.size(); ++i) {
        std::cout << thresholds[i] << "\t" << TPCounts[i] << "\t" << FPCounts[i] << std::endl;
    }
}

// Функция для генерации изображения с кругами
cv::Mat generateImage(int countCircles, int minRadius, int maxRadius, int minContrast, int maxContrast, int blur) {
    int side_length = 4 * maxRadius * (countCircles - 1);
    cv::Mat image(side_length, side_length, CV_8UC1, cv::Scalar(0));

    int radius = minRadius;
    int contrast = minContrast;
    for (int i = side_length / 10; i <= (side_length - side_length / 10); i += (side_length - side_length / 5) / (countCircles - 1)) {
        for (int j = side_length / 10; j <= (side_length - side_length / 10); j += (side_length - side_length / 5) / (countCircles - 1)) {
            cv::Point center(i, j);
            cv::circle(image, center, radius, cv::Scalar(contrast), -1);
            groundTruths.push_back(cv::Vec3f(i, j, radius));
            contrast += (maxContrast - minContrast) / countCircles;
        }
        contrast = minContrast;
        radius += (maxRadius - minRadius) / countCircles;
    }

    cv::GaussianBlur(image, image, cv::Size(blur * 2 + 1, blur * 2 + 1), 0);
    return image; // Возвращаем сгенерированное изображение
}

// Функция для обработки и отображения изображений
void processAndDisplay(const cv::Mat& image, int minRadius, int maxRadius) {
    // Применяем метод Ниблака для бинаризации
    cv::Mat binaryImage = niblackBinary(image, 24, 0.90, 18);

    // Поиск кругов в бинарном изображении
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(binaryImage, circles, cv::HOUGH_GRADIENT, 1,
        (binaryImage.rows - binaryImage.rows / 5) / (10 - 1),
        100, 10,
        minRadius, maxRadius);

    // Рисуем найденные круги
    cv::Mat detectedCirclesImage;
    cv::cvtColor(image, detectedCirclesImage, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < circles.size(); i++) {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        cv::circle(detectedCirclesImage, center, 3, cv::Scalar(0, 255, 0), -1, 1);
        cv::circle(detectedCirclesImage, center, radius, cv::Scalar(0, 0, 255), 2, 1);
    }

    // Анализируем результаты FROC
    std::vector<double> thresholds = { 0.1, 0.2, 0.3, 0.4, 0.5 }; // Примеры порогов IoU
    frocAnalysis(circles, groundTruths, thresholds);

    // Отображаем изображения
    cv::imshow("Original Image", image);
    cv::imshow("Binary Image", binaryImage);
    cv::imshow("Detected Circles", detectedCirclesImage);
    cv::waitKey(0);
}

int main(int argc, char* argv[]) {
    // Значения по умолчанию
    int countCircles = 10;
    int minRadius = 5;
    int maxRadius = 15;
    int minContrast = 50;
    int maxContrast = 255;
    int blur = 4;

    if (argc >= 7) {
        countCircles = std::stoi(argv[1]);
        minRadius = std::stoi(argv[2]);
        maxRadius = std::stoi(argv[3]);
        minContrast = std::stoi(argv[4]);
        maxContrast = std::stoi(argv[5]);
        blur = std::stoi(argv[6]);
    }

    // Генерация изображения
    cv::Mat image = generateImage(countCircles, minRadius, maxRadius, minContrast, maxContrast, blur);

    // Обработка и отображение результатов
    processAndDisplay(image, minRadius, maxRadius);

    return 0;
}