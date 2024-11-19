#include <opencv2/opencv.hpp>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// Функция для нормализации вектора
cv::Vec3f normalize(const cv::Vec3f& v) {
    return v / cv::norm(v, cv::NORM_L2);
}

// Функция для проекции цветов изображения на плоскость
std::vector<cv::Point2f> projectColors(const cv::Mat& src) {
    std::vector<cv::Point2f> result;

    // Определяем нормализованные векторы для проекции на плоскость
    const cv::Vec3f ox = normalize(cv::Vec3f(-1.0f, 1.0f, 0.0f));
    const cv::Vec3f oy = normalize(cv::Vec3f(-1.0f, -1.0f, 2.0f));

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            const auto& color = src.at<cv::Vec3f>(i, j);
            const float alpha = 1.5f / (color.dot(cv::Vec3f(1.0f, 1.0f, 1.0f)));
            if (std::isinf(alpha)) {
                continue;
            }

            // Рассчитываем проекцию цвета на плоскость
            const auto proj = alpha * color - cv::Vec3f(0.5f, 0.5f, 0.5f);
            result.emplace_back(proj.dot(ox), proj.dot(oy));
        }
    }
    return result;
}

// Функция для создания 2D-изображения из проекций
cv::Mat getProj(const std::vector<cv::Point2f>& points, int size) {
    const int height = static_cast<int>(std::sqrt(3.0) / 2 * size);
    cv::Mat result = cv::Mat::zeros(height, size, CV_16UC1);

    // Заполняем матрицу проекциями
    for (const auto& p : points) {
        const int x = static_cast<int>((p.x + 0.75f * std::sqrt(2.0f)) / (1.5f * std::sqrt(2.0f)) * (size - 1));
        const int y = static_cast<int>((p.y + 0.25f * std::sqrt(6.0f)) / (0.75f * std::sqrt(6.0f)) * (height - 1));

        // Увеличиваем значение в матрице
        ++result.at<ushort>(height - y - 1, x);
    }

    // Нормализация значений для получения изображения плотности
    double maxVal = 0.0;
    cv::minMaxLoc(result, nullptr, &maxVal);
    result.convertTo(result, CV_32FC1, 1.0 / maxVal);

    return result;
}

int main(int argc, char* argv[]) {
    std::string input = "C:/Users/Леново/Source/Repos/misis2024s-21-02-suchoruchenkov-m-e/prj.lab/lab08/rgb1.png";
    std::string output = "C:/Users/Леново/Source/Repos/misis2024s-21-02-suchoruchenkov-m-e/prj.lab/lab08/result1.png";
    int size = 256;
    if (argc >= 4) {
        input = argv[1];
        output = argv[2];
        size = std::stoi(argv[3]);
    }

    cv::Mat img = cv::imread(input);
    img.convertTo(img, CV_32FC3, 1.0f / std::numeric_limits<uchar>::max());

    // Проекция цветов и получение результата
    const auto points = projectColors(img);
    // Получение изображения проекций
    cv::Mat result = getProj(points, size);

    result.convertTo(result, CV_8UC1, std::numeric_limits<uchar>::max());

    if (!cv::imwrite(output, result)) {
        std::cerr << "Ошибка при сохранении результата в " << output << std::endl;
        return -1;
    }

    cv::imshow("Original Image", img);
    cv::imshow("Density Projection Result", result);
    cv::waitKey(0);

    return 0;
}