#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <random>
#include <opencv2/opencv.hpp>
#include <map>


cv::Mat outputImage(std::string name, cv::Mat& image, bool show = true) {
    cv::Mat image_clone = image.clone();
    double minVal, maxVal;

    cv::minMaxLoc(image_clone, &minVal, &maxVal); // Находим минимальное и максимальное значение
    double a = 255.0 / (maxVal - minVal);
    double b = (-minVal * 255.0 / (maxVal - minVal));
    image_clone.convertTo(image_clone, CV_8UC1, a, b);

    if (show) {
        cv::imshow(name, image_clone);
    }

    return image_clone;
}

//аддитивный шум со смещением
void addNoise(cv::Mat& image, float* noise) {
    std::random_device device;
    std::mt19937 generator(device());

    // диапазон генерации случайных чисел [min, max]
    double min = 0.0;
    double max = 1.0;

    double stddev = static_cast<double>(noise[0]);
    double shift = static_cast<double>(noise[1]);

    std::uniform_real_distribution<double> distribution(min, max);

    int rows = image.rows;
    int cols = image.cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            //генерация равномерного распределённых случайных величин
            double u = distribution(generator);
            double v = distribution(generator);

            //генерация случайной нормально распредлённой величины, преобразование Бокса-Мюллера
            double rand_numb_box_muller = sqrt(-2 * log(v)) * cos(2 * M_PI * u);

            image.at<uchar>(i, j) = cv::saturate_cast<uchar>(image.at<uchar>(i, j) + (stddev * rand_numb_box_muller) + shift);
        }
    }
}

cv::Mat testImage(int color_background, int nobj, float* radius, std::vector<float>& contrast, float* blur, std::string file_name) {
    //открываем запись в json
    cv::FileStorage js(file_name + ".json", cv::FileStorage::WRITE);

    int size_contrast = contrast.size();
    //отступ
    int indentation = 20;
    int size_radius = 2;

    ////шаг радиуса
    float step = (radius[size_radius - 1] - radius[0]) / nobj;
    int w = indentation;
    for (int i = 0; i < nobj; i++) {
        w += static_cast<int>(2 * (radius[0] + i * step) + indentation + 0.5);
    }

    int h = indentation;
    for (int i = 0; i < size_contrast; i++) {
        h += static_cast<int>(2 * (radius[size_radius - 1]) + indentation + 0.5);
    }


    //запись размеров картинки в json
    js << "h" << h;
    js << "w" << w;

    js << "array" << "[";
    cv::Mat image(h, w, CV_8UC1, cv::Scalar(color_background));
    float sum_distance_h = 0;
    for (int i = 0; i < size_contrast; i++) {
        float radius_max = radius[size_radius - 1];
        sum_distance_h += indentation + radius_max;
        float sum_distance_w = 0;
        for (int j = 0; j < nobj; j++) {
            js << "{";

            float radius_circlej = radius[0] + step * j;
            sum_distance_w += indentation + radius_circlej;
            cv::circle(image, cv::Point(sum_distance_w, sum_distance_h), radius_circlej, cv::Scalar(contrast[i] * 255 + color_background), -1, cv::LINE_8);
            js << "circle" << "[" << sum_distance_w << sum_distance_h << radius_circlej << "]";
            js << "contrast" << contrast[i];
            js << "quality" << std::string();
            js << "}";

            sum_distance_w += radius_circlej;
        }

        sum_distance_h += radius_max;
    }

    //блюр всей картинки
    cv::blur(image, image, cv::Size(blur[0], blur[1]));

    //запись в json
    js.release();

    //сохранение картинки
    cv::imwrite(file_name + ".jpg", image);
    return image;
}

cv::Mat createImageFromFile(std::string file_name, int color_background, float* blur) {
    cv::FileStorage js(file_name + ".json", cv::FileStorage::READ);

    int w, h;

    if (js["w"].empty()) {
        std::cerr << "Field 'w' is empty" << std::endl;
        js.release();
    }
    else {
        w = static_cast<int>(js["w"]);
    }

    if (js["h"].empty()) {
        std::cerr << "Field 'h' is empty" << std::endl;
        js.release();
    }
    else {
        h = static_cast<int>(js["h"]);
    }

    cv::Mat image(h, w, CV_8UC1, cv::Scalar(color_background));

    cv::FileNode js_array;
    if (js["array"].empty()) {
        std::cerr << "Field 'array' is empty" << std::endl;
        js.release();
    }
    else if (js["array"].type() != cv::FileNode::SEQ) {
        std::cerr << "Invalid file format! Not array" << std::endl;
        js.release();
    }
    else {
        js_array = js["array"];
    }

    for (cv::FileNodeIterator i = js_array.begin(); i != js_array.end(); i++) {
        cv::FileNode objectNode = *i;
        double x, y, radius, contrast;

        if (objectNode["circle"][0].empty()) {
            std::cerr << "Field 'x' is empty" << std::endl;
            js.release();
        }
        else {
            x = static_cast<double>(objectNode["circle"][0]);
        }

        if (objectNode["circle"][1].empty()) {
            std::cerr << "Field 'y' is empty" << std::endl;
            js.release();
        }
        else {
            y = static_cast<double>(objectNode["circle"][1]);
        }

        if (objectNode["circle"][2].empty()) {
            std::cerr << "Field 'radius' is empty" << std::endl;
            js.release();
        }
        else {
            radius = static_cast<double>(objectNode["circle"][2]);
        }

        if (objectNode["contrast"].empty()) {
            std::cerr << "Field 'contrast' is empty" << std::endl;
            js.release();
        }
        else {
            contrast = static_cast<double>(objectNode["contrast"]);
        }

        cv::circle(image, cv::Point(x, y), radius, cv::Scalar(contrast * 255 + color_background), -1, cv::LINE_8);
    }

    cv::blur(image, image, cv::Size(blur[0], blur[1]));

    //закрытие в json
    js.release();

    //сохранение картинки
    cv::imwrite(file_name + ".jpg", image);

    return image;
}

//запись в config.json
void createConfig(int color_background, int nobj, float* radius, std::vector<float>& contrast, float* blur, float* noise, float* binary, std::string file_name = "../prj.lab/lab04/results/config") {
    cv::FileStorage js(file_name + ".json", cv::FileStorage::WRITE);

    int size_contrast = contrast.size();

    js << "color_background" << color_background;
    js << "nobj" << nobj;

    js << "radius" << "[";
    for (int i = 0; i < 2; i++) {
        js << radius[i];
    }
    js << "]";

    js << "contrast" << "[";
    for (int i = 0; i < size_contrast; i++) {
        js << contrast[i];
    }
    js << "]";

    js << "blur" << "[";
    for (int i = 0; i < 2; i++) {
        js << blur[i];
    }
    js << "]";

    js << "noise" << "[";
    for (int i = 0; i < 2; i++) {
        js << noise[i];
    }
    js << "]";

    js << "binary" << "[";
    for (int i = 0; i < 2; i++) {
        js << binary[i];
    }
    js << "]";

    js.release();
}

void readFromConfig(int& color_background, int& nobj, float* radius, std::vector<float>& contrast, float* blur, float* noise, float* binary, std::string file_name = "../prj.lab/lab04/results/config") {
    cv::FileStorage js(file_name + ".json", cv::FileStorage::READ);

    color_background = js["color_background"];
    nobj = js["nobj"];

    cv::FileNode arr = js["radius"];
    for (int i = 0; i < 2; i++) {
        radius[i] = arr[i];
    }

    arr = js["contrast"];
    int size = arr.size();
    for (int i = 0; i < size; i++) {
        contrast.push_back(arr[i]);
    }

    arr = js["blur"];
    for (int i = 0; i < 2; i++) {
        blur[i] = arr[i];
    }

    arr = js["noise"];
    for (int i = 0; i < 2; i++) {
        noise[i] = arr[i];
    }

    arr = js["binary"];
    for (int i = 0; i < 2; i++) {
        binary[i] = arr[i];
    }
}

//локальная бинаризация алгоримт Брэдли-Рота
void BradleyThresHold(cv::Mat& image, float* binary, std::string fileName = "") {
    cv::Mat res = image.clone();
    int width = image.cols;
    int height = image.rows;

    //const int S = width/8;
    const float S = binary[0];
    int s2 = S / 2;
    //const float t = 0.15;
    const float t = binary[1];
    unsigned long* integral_image = 0;
    long sum = 0;
    int count = 0;
    int index;
    int x1, y1, x2, y2;

    //рассчитываем интегральное изображение 
    integral_image = new unsigned long[width * height * sizeof(unsigned long*)];

    for (int i = 0; i < width; i++) {
        sum = 0;
        for (int j = 0; j < height; j++) {
            index = j * width + i;
            sum += image.at<uchar>(j, i);
            if (i == 0)
                integral_image[index] = sum;
            else
                integral_image[index] = integral_image[index - 1] + sum;
        }
    }

    //  находим границы для локальные областей
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            index = j * width + i;

            x1 = i - s2;
            x2 = i + s2;
            y1 = j - s2;
            y2 = j + s2;

            if (x1 < 0)
                x1 = 0;
            if (x2 >= width)
                x2 = width - 1;
            if (y1 < 0)
                y1 = 0;
            if (y2 >= height)
                y2 = height - 1;

            count = (x2 - x1) * (y2 - y1);

            sum = integral_image[y2 * width + x2] - integral_image[y1 * width + x2] -
                integral_image[y2 * width + x1] + integral_image[y1 * width + x1];
            if ((long)(image.at<uchar>(j, i) * count) < (long)(sum * (1.0 - t)))
                res.at<uchar>(j, i) = 0;
            else
                res.at<uchar>(j, i) = 255;
        }
    }

    delete[] integral_image;

    image = res;

    if (!fileName.empty()) {
        cv::imwrite(fileName + ".jpg", image);
    }
}

//алгоритм Бернерса
void BernsenThreshHold(cv::Mat& image, float* binary, std::string fileName = "") {
    int rows = image.rows;
    int cols = image.cols;
    int aperturesize = static_cast<int>(binary[0]);
    int r = static_cast<int>(binary[1] * 255);

    cv::Mat binarizedimage = image.clone();

    for (int x = 0; x < rows; x++) {
        for (int y = 0; y < cols; y++) {
            int minVal = image.at<uchar>(x, y);
            int maxVal = image.at<uchar>(x, y);

            // Step 1: Find Min and Max within the aperture
            for (int i = -aperturesize / 2; i <= aperturesize / 2; i++) {
                for (int j = -aperturesize / 2; j <= aperturesize / 2; j++) {
                    int newX = x + i;
                    int newY = y + j;

                    if (newX >= 0 && newX < rows && newY >= 0 && newY < cols) {
                        int currVal = image.at<uchar>(newX, newY);
                        minVal = std::min(minVal, currVal);
                        maxVal = std::max(maxVal, currVal);
                    }
                }
            }

            // Step 2: Calculate Avg
            int avg = (minVal + maxVal) / 2;
            int threshold = avg;
            if (maxVal - minVal < r) {
                threshold = 128;
            }

            // Step 3: Binarize the current pixel
            if (image.at<uchar>(x, y) >= threshold) {
                binarizedimage.at<uchar>(x, y) = 255;  // set pixel to white
            }
            else {
                binarizedimage.at<uchar>(x, y) = 0;    // set pixel to black
            }
        }
    }

    image = binarizedimage;

    if (!fileName.empty()) {
        cv::imwrite(fileName + ".jpg", image);
    }
}


//алгоритм Ниблека
void NiblackThreshHold(cv::Mat& image, float* binary, std::string fileName = "") {
    int rows = image.rows;
    int cols = image.cols;
    int aperturesize = static_cast<int>(binary[0]);
    float k = binary[1];

    cv::Mat binarizedimage = cv::Mat::zeros(rows, cols, CV_8UC1);

    for (int x = 0; x < rows; x++) {
        for (int y = 0; y < cols; y++) {
            float mean, stddev;
            float sum = 0;
            int count = 0;
            // mean и stddev
            for (int i = -aperturesize / 2; i <= aperturesize / 2; i++) {
                for (int j = -aperturesize / 2; j <= aperturesize / 2; j++) {
                    int newX = x + i;
                    int newY = y + j;

                    if (newX >= 0 && newX < rows && newY >= 0 && newY < cols) {
                        sum += image.at<uchar>(newX, newY);
                        count += 1;
                    }
                }
            }

            mean = sum / count;

            float sum_stddev = 0;
            for (int i = -aperturesize / 2; i <= aperturesize / 2; i++) {
                for (int j = -aperturesize / 2; j <= aperturesize / 2; j++) {
                    int newX = x + i;
                    int newY = y + j;

                    if (newX >= 0 && newX < rows && newY >= 0 && newY < cols) {
                        sum_stddev += (image.at<uchar>(newX, newY) - mean) * (image.at<uchar>(newX, newY) - mean);
                    }
                }
            }

            stddev = sqrt(sum_stddev / count);

            int threshhold = static_cast<int>(mean + k * stddev);

            if (image.at<uchar>(x, y) >= threshhold) {
                binarizedimage.at<uchar>(x, y) = 255;
            }
            else {
                binarizedimage.at<uchar>(x, y) = 0;
            }
        }
    }

    image = binarizedimage;

    if (!fileName.empty()) {
        cv::imwrite(fileName + ".jpg", image);
    }
}

void detectBlobsConnected(cv::Mat& binary_image, float threshold_radius) {
    cv::Mat image_clone;
    cv::Mat stats;
    cv::Mat centroids;
    cv::connectedComponentsWithStats(binary_image, image_clone, stats, centroids);

}

// Функция для обнаружения блобов с использованием алгоритма LoG
void detectBlobsLoG(std::vector<cv::Point>& centers, std::vector<float>& radii, cv::Mat& image, float min_sigma, float max_sigma, float step_sigma, int pyr_depth, float min_radius, float max_radius, int threshold) {

    std::vector<cv::Mat> pyramid;
    pyramid.push_back(image.clone());

    for (int i = 1; i < pyr_depth; i++) {
        cv::Mat pyr_prev = pyramid[i - 1];
        cv::Mat pyr_cur;
        cv::pyrDown(pyr_prev, pyr_cur);
        pyramid.push_back(pyr_cur);
    }

    //сохраняем все ядра и соответсвующие им сигмы
    struct LoG_kernels {
        cv::Mat kernel;
        float sigmak;
    };

    std::vector<LoG_kernels> kernels;
    for (float sigma = min_sigma; sigma <= max_sigma; sigma += step_sigma) {
        //Получение одномерного ядра гаусса
        int kernel_size = static_cast<int>(6 * sigma) / 2 - 1; //согласование 63 максимальный размер

        if (kernel_size % 2 == 0) {
            kernel_size += 1;
        }

        std::cout << kernel_size << std::endl;
        cv::Mat kernel1D = cv::getGaussianKernel(kernel_size, sigma, CV_32F);

        // Создание квадратного симметричного ядра гаусса
        cv::Mat kernel2D = kernel1D * kernel1D.t();
        cv::Mat laplacian;
        cv::Laplacian(kernel2D, laplacian, CV_32F, kernel_size);

        LoG_kernels kernel_cur;
        kernel_cur.kernel = laplacian;
        kernel_cur.sigmak = sigma;
        kernels.push_back(kernel_cur);
    }

    //нахождение подозрительных окружностей
    struct Blob {
        cv::Point center;
        float radius;
        float response;
        int flag;
    };

    std::vector<Blob> blobList;

    for (int i = 0; i < pyr_depth; i++) {
        for (float k = 0; k < kernels.size(); k++) {
            cv::Mat laplacian = kernels[k].kernel;
            float sigma = kernels[k].sigmak;
            float radius = sigma * sqrt(2.0);

            cv::Mat image_clone;

            cv::filter2D(pyramid[i], image_clone, CV_32F, laplacian);

            outputImage("gg", image_clone);

            cv::Mat erode;
            cv::Mat image_compare;
            cv::erode(image_clone, erode, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * radius + 4, 2 * radius + 4)));
            cv::compare(image_clone, erode, image_compare, cv::CMP_EQ);
            //outputImage("image_compare", image_compare);


            std::vector<cv::Point> idxpoints;

            cv::Mat image_temp;
            cv::Mat stats;
            cv::Mat centroids;
            int numLabels = cv::connectedComponentsWithStats(image_compare, image_temp, stats, centroids);

            for (int i = 1; i < numLabels; i++) {
                if (stats.at<int>(i, cv::CC_STAT_AREA) > threshold) {
                    std::cout << stats.at<int>(i, cv::CC_STAT_AREA) << std::endl;
                    cv::Mat componentMask = image_temp == i;
                    image_compare.setTo(cv::Scalar::all(1), componentMask);
                }
            }

            //outputImage("image_compare12", image_compare);

            cv::findNonZero(image_compare, idxpoints);

            for (int j = 0; j < idxpoints.size(); j++) {
                float temp_radius = pow(2, i) * radius;
                //if (temp_radius >= min_radius && temp_radius <= max_radius) {
                float normalize_response = sigma * sigma * image_clone.at<float>(idxpoints[j]);
                Blob blob;
                blob.center = pow(2, i) * idxpoints[j];
                blob.radius = temp_radius;
                blob.response = normalize_response;
                blob.flag = 1;
                blobList.push_back(blob);
                //}
            }
        }
    }


    cv::Mat clear = image.clone();

    cv::cvtColor(image, clear, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < blobList.size(); i++) {
        if (blobList[i].flag == 1) {
            cv::circle(clear, blobList[i].center, blobList[i].radius, cv::Scalar(0, 0, 255), 3);
        }
    }

    cv::imshow("image_kddk", clear);



    for (int i = 0; i < blobList.size(); i++) {
        for (int j = i + 1; j < blobList.size(); j++) {
            int xi = blobList[i].center.x;
            int yi = blobList[i].center.y;
            int xj = blobList[j].center.x;
            int yj = blobList[j].center.y;
            float dist = sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj));

            if (dist < (blobList[i].radius + blobList[j].radius)) {
                if (blobList[i].response < blobList[j].response) {
                    blobList[j].flag = 0;
                }
                else {
                    blobList[i].flag = 0;
                }
            }
        }
    }

    for (int i = 0; i < blobList.size(); i++) {
        if (blobList[i].flag == 1) {
            centers.push_back(blobList[i].center);
            radii.push_back(blobList[i].radius);
        }
    }
}

void detectBlobsDoG(std::vector<cv::Point>& centers, std::vector<float>& radii, cv::Mat& image, float min_sigma, float max_sigma, float step_sigma, int pyr_depth, float min_radius, float max_radius) {
}


int main(int argc, char** argv) {
    int color_background = 30;
    int nobj = 10;
    float radius[2] = { 8.0, 150.0 };
    std::vector<float> contrast{ 0.4, 0.45, 0.57, 0.6, 0.7 };
    float blur[2] = { 5.0, 7.2 };
    float noise[2] = { 6, 2 };
    float binary[2] = { 3.0, 0.15 };

    //createConfig(color_background, nobj, radius, contrast, blur, noise, binary);

    // int color_background;
    // int nobj;
    // float radius[2];
    // std::vector<float> contrast;
    // float blur[2];
    // float noise[2];
    // float binary[2];
    //readFromConfig(color_background, nobj, radius, contrast, blur, noise, binary);

    cv::Mat image = testImage(color_background, nobj, radius, contrast, blur, "../prj.lab/lab04/results/final2");
    addNoise(image, noise);


    //cv::Mat image1 = createImageFromFile("../prj.lab/lab04/results/final2", 150, blur);
    //addNoise(image1, noise);

    //NiblackThreshHold(image1, binary, "../prj.lab/lab04/results/final2_binary");

    cv::imshow("image", image);
    //cv::imshow("image1", image1);

    std::vector<cv::Point> centers;
    std::vector<float> radii;
    // Обнаружение блобов с использованием алгоритма LoG
    detectBlobsLoG(centers, radii, image, 6.5, 8, 0.4, 1, 11.0, 70.0, 1);

    cv::Mat clear;

    cv::cvtColor(image, clear, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < centers.size(); i++) {
        cv::circle(clear, centers[i], radii[i], cv::Scalar(0, 0, 255), 3);
    }

    cv::imshow("image_clone", clear);

    cv::waitKey(0);
    return 0;
}