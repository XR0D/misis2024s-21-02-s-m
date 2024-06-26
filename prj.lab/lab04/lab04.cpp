#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <random>
#include <map>
#include <windows.h>
#include <opencv2/opencv.hpp>

int binary1 = 5;
int binary2 = 1;

void createFolder(std::string folder_name) {
    struct stat info;
    // Проверка существования папки
    DWORD fAttrib = GetFileAttributesA(folder_name.c_str());
    if (fAttrib != INVALID_FILE_ATTRIBUTES && (fAttrib & FILE_ATTRIBUTE_DIRECTORY)) {
        std::cout << "Folder exists: " << folder_name << std::endl;
    } else {
        // Создание папки
        if (CreateDirectoryA(folder_name.c_str(), NULL)) {
            std::cout << "Folder create: " << folder_name << std::endl;
        } else {
            std::cerr << "Folder create error: " << GetLastError() << std::endl;
        }
    }
}

//корректный вывод cv::Mat 
cv::Mat outputImage(std::string name, cv::Mat& image, std::string folder_name = "", bool save = false, bool show = true) {
    cv::Mat image_clone = image.clone();
    double minVal, maxVal;

    cv::minMaxLoc(image_clone, &minVal, &maxVal); // Находим минимальное и максимальное значение
    double a = 255.0 / (maxVal - minVal);
    double b = (-minVal * 255.0 / (maxVal - minVal));
    image_clone.convertTo(image_clone, CV_8UC1, a, b);

    if (show) {
        cv::imshow(name, image_clone);
    }

    if (save) {
        cv::imwrite(folder_name + name + ".jpg", image_clone);
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

//генерация тестовой картинки и запись параметров в js
cv::Mat testImage (int color_background, int nobj, float* radius, std::vector<float>& contrast, float* blur, std::vector<cv::Point>& centers, std::vector<float>& radii) { 
        int size_contrast = contrast.size();
        //отступ
        int indentation = 150;
        int size_radius = 2;

        ////шаг радиуса
        float step = (radius[size_radius - 1] - radius[0]) / (nobj - 1);
        int w = indentation;

        //расчёт центров
        for (int i = 0; i < nobj; i++) {
                  w += static_cast<int>(2*(radius[0] + i*step) + indentation + 0.5);
        } 

        int h = indentation;
        for (int i = 0; i < size_contrast; i++) {
                  h += static_cast<int>(2*(radius[size_radius - 1]) + indentation + 0.5);
        } 

        //создание картинки
        cv::Mat image(h, w, CV_8UC1, cv::Scalar(color_background));
        float sum_distance_h = 0;
        for (int i = 0; i < size_contrast; i++) {
                  float radius_max = radius[size_radius - 1];
                  sum_distance_h += indentation + radius_max;
                  float sum_distance_w = 0;
                  for (int j = 0; j < nobj; j++) {

                            float radius_circlej = radius[0] + step*j;
                            sum_distance_w += indentation + radius_circlej;

                            cv::circle(image, cv::Point(sum_distance_w, sum_distance_h), radius_circlej, cv::Scalar(contrast[i] * 255 + color_background), -1, cv::LINE_8);

                            centers.push_back(cv::Point(sum_distance_w, sum_distance_h));
                            radii.push_back(radius_circlej);

                            sum_distance_w += radius_circlej;
                  }

                  sum_distance_h += radius_max;
        }
        
        //блюр всей картинки
        cv::GaussianBlur(image, image, cv::Size(), blur[0], blur[1]);

        //сохранение картинки
        return image;
}

//создаем тестовую картинку из json
cv::Mat createImageFromFile (std::string file_name, int color_background, float* blur, std::vector<cv::Point>& centers, std::vector<float>& radii) {
    cv::FileStorage js(file_name + ".json", cv::FileStorage::READ);

    int w, h;

    if (js["w"].empty()) {
        std::cerr << "Field 'w' is empty" << std::endl;
        js.release();
    } else {
        w = static_cast<int>(js["w"]);
    }

    if (js["h"].empty()) {
        std::cerr << "Field 'h' is empty" << std::endl;
        js.release();
    } else {
        h = static_cast<int>(js["h"]);
    }

    cv::Mat image(h, w, CV_8UC1, cv::Scalar(color_background));

    cv::FileNode js_array;
    if (js["array"].empty()) {
        std::cerr << "Field 'array' is empty" << std::endl;
        js.release();
    } else if (js["array"].type() != cv::FileNode::SEQ) {
        std::cerr << "Invalid file format! Not array" << std::endl;
        js.release();
    } else {
        js_array = js["array"];
    }

    for (cv::FileNodeIterator i = js_array.begin(); i != js_array.end(); i++) {
        cv::FileNode objectNode = *i;
        double x, y, radius, contrast;

        if (objectNode["circle"][0].empty()) {
            std::cerr << "Field 'x' is empty" << std::endl;
            js.release();
        } else {
            x = std::stod(objectNode["circle"][0]);
        }

        if (objectNode["circle"][1].empty()) {
            std::cerr << "Field 'y' is empty" << std::endl;
            js.release();
        } else {
            y = std::stod(objectNode["circle"][1]);
        }

        if (objectNode["circle"][2].empty()) {
            std::cerr << "Field 'radius' is empty" << std::endl;
            js.release();
        } else {
            radius = std::stod(objectNode["circle"][2]);
        }

        if (objectNode["contrast"].empty()) {
            std::cerr << "Field 'contrast' is empty" << std::endl;
            js.release();
        } else {
            contrast = std::stod(objectNode["contrast"]);
        }

        cv::circle(image, cv::Point(x, y), radius, cv::Scalar(contrast * 255 + color_background), -1, cv::LINE_8);
        centers.push_back(cv::Point(x, y));
        radii.push_back(radius);
    }

    cv::GaussianBlur(image, image, cv::Size(), blur[0], blur[1]);

    //закрытие в json
    js.release();

    std::cout << "Successfully generate from json" << std::endl;

    return image;
}

//запись в config.json
void createConfig(int color_background, int nobj, float* radius, std::vector<float>& contrast, float* blur, float* noise, float* binary, float iou_threshold, float step, std::string file_name = "../prj.lab/lab04/results/config") {
    cv::FileStorage js(file_name + ".json", cv::FileStorage::WRITE);

    int size_contrast = contrast.size();
    
    js << "color_background" << color_background;
    js << "nobj" << nobj;
    
    std::ostringstream ss;
    std::string floatString;
    
    js << "radius" << "[";
    for (int i = 0; i < 2; i++) {
        ss << std::fixed << radius[i]; // Установка формата вывода
        floatString = ss.str(); 
        js << floatString;
         ss.str("");
    }
    js << "]";

    js << "contrast" << "[";
    for (int i = 0; i < size_contrast; i++) { 
        ss << std::fixed << contrast[i]; // Установка формата вывода
        floatString = ss.str(); 
        js << floatString;
        ss.str("");
    }
    js << "]";

    js << "blur" << "[";
    for (int i = 0; i < 2; i++) {
        ss << std::fixed << blur[i]; // Установка формата вывода
        floatString = ss.str(); 
        js << floatString; 
        ss.str("");
    }
    js << "]";

    js << "noise" << "[";
    for (int i = 0; i < 2; i++) { 
        ss << std::fixed << noise[i]; // Установка формата вывода
        floatString = ss.str(); 
        js << floatString;
        ss.str("");
    }
    js << "]";

    js << "binary" << "[";
    for (int i = 0; i < 2; i++) {
        ss << std::fixed << binary[i]; // Установка формата вывода
        floatString = ss.str(); 
        js << floatString; 
        ss.str("");
    }
    js << "]";

    ss << std::fixed << iou_threshold; // Установка формата вывода
    floatString = ss.str(); 
    js << "iou_threshold" << floatString; 

    ss.str("");

    ss << std::fixed << step; // Установка формата вывода
    floatString = ss.str(); 
    js << "step" << floatString;

    js.release();

    std::cout << "The config Successfully written" << std::endl;
}

//чтение из конфига
void readFromConfig(int& color_background, int& nobj, float* radius, std::vector<float>& contrast, float* blur, float* noise, float* binary, float& iou_threshold, float& step, std::string file_name = "../prj.lab/lab04/results/config") {
    cv::FileStorage js(file_name + ".json", cv::FileStorage::READ);

    color_background = static_cast<int>(js["color_background"]);
    nobj = static_cast<int>(js["nobj"]);

    cv::FileNode arr = js["radius"];
    for (int i = 0; i < 2; i++) {
        radius[i] = std::stod(arr[i]);
    }

    arr = js["contrast"];
    int size = arr.size();
    for (int i = 0; i < size; i++) {
        contrast.push_back(std::stod(arr[i]));
    }

    arr = js["blur"];
    for (int i = 0; i < 2; i++) {
        blur[i] = std::stod(arr[i]);
    }

    arr = js["noise"];
    for (int i = 0; i < 2; i++) {
        noise[i] = std::stod(arr[i]);
    }

    arr = js["binary"];
    if (arr.empty()) {
            std::cerr << "Field 'binary' is empty" << std::endl;
    } else {
        for (int i = 0; i < 2; i++) {
            binary[i] = std::stod(arr[i]);
        }
    }

    iou_threshold = std::stod(js["iou_threshold"]);
    step = std::stod(js["step"]);

    std::cout << "Successfully read from config" << std::endl;
}

//создаем Json с результатами генерации и качества 
void makeJson(std::string file_name, cv::Size size_image, std::vector<cv::Point>& centers, std::vector<float>& radii, std::vector<float>& contrast, std::vector<float>& iou, std::vector<int>& froc) {
    //открываем запись в json
    cv::FileStorage js(file_name + ".json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

    //запись размеров картинки в json
    js << "h" << size_image.height;
    js << "w" << size_image.width;

    js << "array" << "[";
    int j = -1;
    float floatValue;
    std::ostringstream ss;
    std::string floatString;
    for (int i = 0; i < centers.size(); i++) {

        if (radii[i] == radii[0]) {
            j++;
        }

        js << "{";

        ss << std::fixed << centers[i].x; // Установка формата вывода
        floatString = ss.str();
        js << "circle" << "[" << floatString;

        ss.str("");

        ss << std::fixed << centers[i].y; // Установка формата вывода
        floatString = ss.str();
        js <<  floatString;

        ss.str("");

        ss << std::fixed << radii[i]; // Установка формата вывода
        floatString = ss.str();
        js << floatString << "]";

        ss.str("");

        ss << std::fixed << contrast[j]; // Установка формата вывода
        floatString = ss.str();
        js << "contrast" << floatString;

        ss.str("");

        ss << std::fixed << iou[i]; // Установка формата вывода
        floatString = ss.str();
        js << "IoU" << floatString;

        ss.str("");

        js << "}";
    }

    js << "]";

    js << "tp" << froc[0];
    js << "fp" << froc[1];
    js << "fn" << froc[2];

    //запись в json
    js.release();

    std::cout << "The generate json file Successfully written" << std::endl;
}

//создаем Json с результатами генерации и качества 
void makeResJson(std::string file_name, cv::Size size_image, std::vector<cv::Point>& centers, std::vector<float>& radii, std::vector<float>& contrast, std::vector<float>& iou, float iou_threshold, std::vector<int>& froc) {
    //открываем запись в json
    cv::FileStorage js(file_name + ".json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

    //запись размеров картинки в json
    js << "h" << size_image.height;
    js << "w" << size_image.width;

    js << "array" << "[";
    int j = -1;
    float floatValue;
    std::ostringstream ss;
    std::string floatString;
    for (int i = 0; i < centers.size(); i++) {

        if (radii[i] == radii[0]) {
            j++;
        }

        js << "{";

        ss << std::fixed << centers[i].x; // Установка формата вывода
        floatString = ss.str();
        js << "circle" << "[" << floatString;

        ss.str("");

        ss << std::fixed << centers[i].y; // Установка формата вывода
        floatString = ss.str();
        js <<  floatString;

        ss.str("");

        ss << std::fixed << radii[i]; // Установка формата вывода
        floatString = ss.str();
        js << floatString << "]";

        ss.str("");

        ss << std::fixed << contrast[j]; // Установка формата вывода
        floatString = ss.str();
        js << "contrast" << floatString;

        ss.str("");

        ss << std::fixed << iou[i]; // Установка формата вывода
        floatString = ss.str();
        js << "IoU" << floatString;

        ss.str("");

        js << "}";
    }

    js << "]";

    js << "tp" << froc[0];
    js << "fp" << froc[1];
    js << "fn" << froc[2];

    ss << std::fixed << iou_threshold; // Установка формата вывода
    floatString = ss.str();
    js << "threshold" << floatString;

    //запись в json
    js.release();

    std::cout << "The generate json file Successfully written" << std::endl;
}

float IoU(cv::Point center1, float radius1, cv::Point center2, float radius2) {
    int x1 = center1.x;
    int y1 = center1.y;
    int x2 = center2.x;
    int y2 = center2.y;

    float d = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    float intersection = 0;
    float s1;
    float s2;
    //первая вложена во вторую
    if (radius2 - radius1 >= d) {
        intersection = M_PI * radius1 * radius1;
    } else if (radius1 - radius2 >= d) {
        intersection = M_PI * radius2 * radius2;
    } else if (d <= radius1 + radius2) {
        float f1 = 2 * acos(((radius1 * radius1) - (radius2 * radius2) + (d * d)) / (2 * radius1 * d));
        float f2 = 2 * acos(((radius2 * radius2) - (radius1 * radius1) + (d * d)) / (2 * radius2 * d));

        s1 = radius1 * radius1 * (f1 - sin(f1)) / 2;
        s2 = radius2 * radius2 * (f2 - sin(f2)) / 2;  

        intersection = s1 + s2;
    }

    s1 = M_PI * radius1 * radius1;
    s2 = M_PI * radius2 * radius2;

    float uni = s1 + s2 - intersection;

    float iou = intersection / uni;

    return iou;
}

//локальная бинаризация алгоримт Брэдли-Рота
void BradleyThreshHold(cv::Mat& image, float* binary, std::string fileName = "") {
    cv::Mat res = image.clone();
    int width = image.cols;
    int height = image.rows;  

    //const int S = width/8;
    const float S = binary[0];
    int s2 = S/2;
    //const float s = 0.15;
    const float s = binary[1];
    unsigned long* integral_image = 0;
    long sum=0;
    int count=0;
    int index;
    int x1, y1, x2, y2;

    //рассчитываем интегральное изображение 
    integral_image = new unsigned long [width*height*sizeof(unsigned long*)];
    
    for (int i = 0; i < width; i++) {
      sum = 0;
      for (int j = 0; j < height; j++) {
        index = j * width + i;
        sum += image.at<uchar>(j, i);
        if (i==0)
    	    integral_image[index] = sum;
        else
    	    integral_image[index] = integral_image[index-1] + sum;
      }
    }
    
//  находим границы для локальные областей
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++) {
        index = j * width + i;

        x1=i-s2;
        x2=i+s2;
        y1=j-s2;
        y2=j+s2;

        if (x1 < 0)
    	    x1 = 0;
        if (x2 >= width)
    	    x2 = width-1;
        if (y1 < 0)
    	    y1 = 0;
        if (y2 >= height)
    	    y2 = height-1;

        count = (x2-x1)*(y2-y1);

        sum = integral_image[y2*width+x2] - integral_image[y1*width+x2] -
    				  integral_image[y2*width+x1] + integral_image[y1*width+x1];
        if ((long)(image.at<uchar>(j, i)*count) < (long)(sum*(1.0-s)))
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

void BradleyThreshHoldGui(int , void* data) {
    cv::Mat image = *static_cast<cv::Mat*>(data);

    float binary[2] = {static_cast<float>(binary1), static_cast<float>(binary2) / 100.0f};

    BradleyThreshHold(image, binary);
    cv::imshow("Bradley", image);
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
            } else {
                binarizedimage.at<uchar>(x, y) = 0;    // set pixel to black
            }
        }
    }

    image = binarizedimage;

    if (!fileName.empty()) {
        cv::imwrite(fileName + ".jpg", image);
    }
}

void BernsenThreshHoldGui(int , void* data) {
    cv::Mat image = *static_cast<cv::Mat*>(data);

    float binary[2] = {static_cast<float>(binary1), static_cast<float>(binary2) / 100.0f};

    BernsenThreshHold(image, binary);
    cv::imshow("Bernsen", image);
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
            } else {
                binarizedimage.at<uchar>(x, y) = 0;
            }
        }
    }        

    image = binarizedimage;

    if (!fileName.empty()) {
        cv::imwrite(fileName + ".jpg", image);
    }
}

void NiblackThreshHoldGui(int , void* data) {
    cv::Mat image = *static_cast<cv::Mat*>(data);

    float binary[2] = {static_cast<float>(binary1), static_cast<float>(binary2) / 100.0f};

    NiblackThreshHold(image, binary);
    cv::imshow("Niblack", image);
}

void detectBlobsConnected(std::vector<cv::Point>& centers, std::vector<float>& radii, cv::Mat& binary_image, float threshold_area_min, float threshold_area_max) {
    cv::Mat image_clone;
    cv::Mat stats;
    cv::Mat centroids;
    cv::connectedComponentsWithStats(binary_image, image_clone, stats, centroids);

    for (int i = 1; i < stats.rows; i++) { // Начинаем с 1, так как 0 - это фон
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // Проверка, насколько компонента похожа на круг
        double aspect_ratio = static_cast<double>(width) / height;
        double circularity = 4 * M_PI * stats.at<int>(i, cv::CC_STAT_AREA) / ((height + width) * (height + width));
        double epsilon = 0.02; // Погрешность для круглой формы

        if (stats.at<int>(i, cv::CC_STAT_AREA) >= threshold_area_min && stats.at<int>(i, cv::CC_STAT_AREA) <= threshold_area_max) {
            centers.push_back(cv::Point(cv::Point(x + width / 2, y + height / 2)));
            radii.push_back(width / 2);
        }
    }  
}

// Функция для обнаружения блобов с использованием алгоритма LoG
void detectBlobsLoG(std::vector<cv::Point>& centers, std::vector<float>& radii, cv::Mat& image, double min_sigma, double max_sigma, double step_sigma, int pyr_depth, int count, float thresh_iou, std::string folder_name, int level_show = 0) {

    std::vector<cv::Mat> pyramid;
    cv::Mat image32;
    //преобразуем тип так как нам нужна точность,  а в GaussianBlur нельзя указать тип 
    image.convertTo(image32, CV_32F);
    pyramid.push_back(image32);

    for (int i = 1; i < pyr_depth; i++) {
        cv::Mat pyr_prev = pyramid[i - 1];
        cv::Mat pyr_cur;
        cv::pyrDown(pyr_prev, pyr_cur, pyr_prev.size() / 2);
        pyramid.push_back(pyr_cur);
    }

    //нахождение подозрительных окружностей
    struct Blob {
        cv::Point center;
        float radius;
        float response;
    };

    std::vector<Blob> blobList;

    for (int i = 0; i < pyr_depth; i++) {

        //для отображения блобов на уровне пирамиды
        cv::Mat res;
        cv::cvtColor(image, res, cv::COLOR_GRAY2BGR);
        for (float sigma = min_sigma; sigma <= max_sigma; sigma += step_sigma) {
            float radius = sigma * sqrt(2.0);
            int kernel_size = static_cast<int>(6*sigma);

            if (kernel_size % 2 == 0) {
                kernel_size += 1;
            }

            if (i == 0) {
                std::cout << "kernel_size: " << kernel_size << " sigma: " << sigma << " radius: " << sigma * sqrt(2.0) << std::endl;
            }

            cv::Mat gaus;
            cv::GaussianBlur(pyramid[i], gaus, cv::Size(kernel_size, kernel_size), sigma);

            cv::Mat image_clone;
            cv::Laplacian(gaus, image_clone, CV_32F, kernel_size);
            
            if (i == level_show - 1) {
                outputImage("LoG", image_clone, folder_name, true);
            }

            cv::Mat erode;
            cv::Mat image_compare;
            int erode_size = static_cast<int>(2*radius + 1);
            cv::erode(image_clone, erode, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erode_size, erode_size)));

            if (i == level_show - 1) {
                outputImage("erode", erode, folder_name, true);
            }

            cv::compare(image_clone, erode, image_compare, cv::CMP_EQ);
            
            //находим count количество минимальных откликов
            cv::Mat bitwise;
            image_compare.convertTo(image_compare, CV_32F);
            if (i == level_show - 1) {
                outputImage("min", image_compare);
            }

            bitwise = image_clone.mul(image_compare / 255.0);
            if (i == level_show - 1) {
                outputImage("min_response", bitwise);
            }

            cv::Mat bitwise_copy = bitwise.clone();
            bitwise = bitwise.reshape(1,1);
            cv::sort(bitwise, bitwise, cv::SORT_ASCENDING);

            //матрица соответсвий
            cv::Mat responses = (bitwise_copy <= bitwise.at<float>(0, count - 1));
            responses.convertTo(responses, CV_32F);
            if (i == level_show - 1) {
                outputImage("responses" + std::to_string(count), responses, folder_name, true);
            }
            image_compare = bitwise_copy.mul(responses / 255.0);

            if (i == level_show - 1) {
                outputImage("count" + std::to_string(count), image_compare, folder_name, true);
            }

            //находим отсавщиеся минимумы
            std::vector<cv::Point> idxpoints;

            cv::findNonZero(image_compare, idxpoints);

            for (int j = 0; j < idxpoints.size(); j++) {
                Blob blob;
                blob.center = (pow(2, i) * idxpoints[j]);
                blob.radius = pow(2, i) * radius;
                blob.response = sigma * sigma * image_compare.at<float>(idxpoints[j]);

                if (i == level_show - 1) {
                    cv::circle(res, blob.center, 3, cv::Scalar(0, 0, 255), -1);
                    cv::circle(res, blob.center, blob.radius, cv::Scalar(0, 0, 255), 2);
                    std::cout << std::fixed <<std::setprecision(10) <<"x " << blob.center.x << " y " << blob.center.y << " r " << blob.radius << " res " << blob.response << std::endl;
                }  

                blobList.push_back(blob);
            }
        }

        if (i == level_show - 1) {
                cv::imshow("blobs" + std::to_string(i + 1), res);
                cv::imwrite(folder_name + "blobs" + std::to_string(i + 1) + ".jpg", res);
            }
    }

    cv::Mat res;
    cv::cvtColor(image, res, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < blobList.size(); i++) {
        cv::circle(res, blobList[i].center, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(res, blobList[i].center, blobList[i].radius, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("all blobs", res);

    for (int f = 0; f < blobList.size(); f++) {
        for (int s = f + 1; s < blobList.size();) {
            float iou = IoU(blobList[f].center, blobList[f].radius, blobList[s].center, blobList[s].radius);

            if (iou >= thresh_iou) {
                if (blobList[f].response < blobList[s].response) {
                    blobList.erase(blobList.begin() + s);
                } else {
                    blobList.erase(blobList.begin() + f);
                    s = f + 1;
                } 
            } else {
                s++;
            }
        }
    }

    for (int i = 0; i < blobList.size(); i++) {
        centers.push_back(blobList[i].center);
        radii.push_back(blobList[i].radius);
    }
}

//DoG детектор
void detectBlobsDoG(std::vector<cv::Point>& centers, std::vector<float>& radii, cv::Mat& image, double min_sigma, double max_sigma, double step_sigma, int pyr_depth, int count, float thresh_iou, std::string folder_name, int level_show = 0) {

    std::vector<cv::Mat> pyramid;
    cv::Mat image32;
    //преобразуем тип так как нам нужна точность,  а в GaussianBlur нельзя указать тип 
    image.convertTo(image32, CV_32F);
    pyramid.push_back(image32);

    for (int i = 1; i < pyr_depth; i++) {
        cv::Mat pyr_prev = pyramid[i - 1];
        cv::Mat pyr_cur;
        cv::pyrDown(pyr_prev, pyr_cur, pyr_prev.size() / 2);
        pyramid.push_back(pyr_cur);
    }

    //нахождение подозрительных окружностей
    struct Blob {
        cv::Point center;
        float radius;
        float response;
    };

    std::vector<Blob> blobList;

    for (int i = 0; i < pyr_depth; i++) {

        int kernel_size = static_cast<int>(6*min_sigma);

        if (kernel_size % 2 == 0) {
            kernel_size += 1;
        }

        //для отображения блобов на уровне пирамиды
        cv::Mat res;
        cv::cvtColor(image, res, cv::COLOR_GRAY2BGR);

        cv::Mat prev_gaus;
        cv::Mat pyramid32;
        pyramid[i].convertTo(pyramid32, CV_32F);
        cv::GaussianBlur(pyramid32, prev_gaus, cv::Size(kernel_size, kernel_size), min_sigma);
        for (float sigma = min_sigma; sigma <= max_sigma; sigma += step_sigma) {
            float radius = sigma * sqrt(2.0);
            kernel_size = static_cast<int>(6*sigma);

            if (kernel_size % 2 == 0) {
                kernel_size += 1;
            }

            if (i == 0) {
                std::cout << "kernel_size: " << kernel_size << " sigma: " << sigma << " radius: " << sigma * sqrt(2.0) << std::endl;
            }

            cv::Mat cur_gaus;
            pyramid[i].convertTo(pyramid32, CV_32F);
            cv::GaussianBlur(pyramid32, cur_gaus, cv::Size(kernel_size, kernel_size), sigma + step_sigma);

            cv::Mat image_clone = cur_gaus - prev_gaus;
            
            if (i == level_show - 1) {
                outputImage("DoG", image_clone, folder_name, true);
            }

            cv::Mat erode;
            cv::Mat image_compare;
            int erode_size = static_cast<int>(2*radius + 1);
            cv::erode(image_clone, erode, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erode_size, erode_size)));

            if (i == level_show - 1) {
                outputImage("erode", erode, folder_name, true);
            }

            cv::compare(image_clone, erode, image_compare, cv::CMP_EQ);
            
            //находим count количество минимальных откликов
            cv::Mat bitwise;
            image_compare.convertTo(image_compare, CV_32F);
            if (i == level_show - 1) {
                outputImage("min", image_compare);
            }

            bitwise = image_clone.mul(image_compare / 255.0);
            if (i == level_show - 1) {
                outputImage("min_response", bitwise);
            }

            cv::Mat bitwise_copy = bitwise.clone();
            bitwise = bitwise.reshape(1,1);
            cv::sort(bitwise, bitwise, cv::SORT_ASCENDING);

            //матрица соответсвий
            cv::Mat responses = (bitwise_copy <= bitwise.at<float>(0, count - 1));
            responses.convertTo(responses, CV_32F);
            if (i == level_show - 1) {
                outputImage("responses" + std::to_string(count), responses, folder_name, true);
            }
            
            image_compare = bitwise_copy.mul(responses / 255.0);

            if (i == level_show - 1) {
                outputImage("count" + std::to_string(count), image_compare);
            }

            //находим отсавщиеся минимумы
            std::vector<cv::Point> idxpoints;

            cv::findNonZero(image_compare, idxpoints);

            for (int j = 0; j < idxpoints.size(); j++) {
                Blob blob;
                blob.center = (pow(2, i) * idxpoints[j]);
                blob.radius = pow(2, i) * radius;
                blob.response = sigma * sigma * image_compare.at<float>(idxpoints[j]);

                if (i == level_show - 1) {
                    cv::circle(res, blob.center, 3, cv::Scalar(0, 0, 255), -1);
                    cv::circle(res, blob.center, blob.radius, cv::Scalar(0, 0, 255), 2);
                    std::cout << std::fixed <<std::setprecision(10) <<"x " << blob.center.x << " y " << blob.center.y << " r " << blob.radius << " res " << blob.response << std::endl;
                }  

                blobList.push_back(blob);
            }

            prev_gaus = cur_gaus;
        }

        if (i == level_show - 1) {
            cv::imshow("blobs" + std::to_string(i + 1), res);
            cv::imwrite(folder_name + "blobs" + std::to_string(i + 1) + ".jpg", res);
        }
    }

    cv::Mat res;
    cv::cvtColor(image, res, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < blobList.size(); i++) {
        cv::circle(res, blobList[i].center, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(res, blobList[i].center, blobList[i].radius, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("all blobs", res);

    for (int f = 0; f < blobList.size(); f++) {
        for (int s = f + 1; s < blobList.size();) {
            float iou = IoU(blobList[f].center, blobList[f].radius, blobList[s].center, blobList[s].radius);

            if (iou >= thresh_iou) {
                if (blobList[f].response < blobList[s].response) {
                    blobList.erase(blobList.begin() + s);
                } else {
                    blobList.erase(blobList.begin() + f);
                    s = f + 1;
                } 
            } else {
                s++;
            }
        }
    }

    for (int i = 0; i < blobList.size(); i++) {
        centers.push_back(blobList[i].center);
        radii.push_back(blobList[i].radius);
    }
}

//оценка качества
std::vector<float> qualityAssessment (std::vector<int>& froc, std::vector<cv::Point>& centers0, std::vector<float>& radii0, std::vector<cv::Point>& centers, std::vector<float>& radii, float tresh, bool print = true) {
    std::vector<float>iou(centers0.size(), -4);
    std::vector<float>flags(centers.size(), 0);
    int tp = 0;
    int fn = 0;
    int fp = 0;
    for (int i = 0; i < centers.size(); i++) {
        for (int j = 0; j < centers0.size(); j++) {
            float iou_metric = IoU(centers0[j], radii0[j], centers[i], radii[i]);

            if (iou_metric > tresh && flags[i] == 0) {
                tp += 1;
                flags[i] = 1;
            }

            if (iou_metric > iou[j]) {
                iou[j] = iou_metric;
            }
        }

        if (flags[i] == 0) {
            fp += 1;
        }
    }

    for (int j = 0; j < centers0.size(); j++) {
        if (iou[j] <= tresh || iou[j] == 0) {
            fn += 1;
        }
    }

    // fp = centers.size() - tp;
    // if (fp < 0) {
    //     fp = 0;
    // }
    // fn = centers0.size() - tp;
    // if (fn < 0) {
    //     fn = 0;
    // }

    froc.push_back(tp);
    froc.push_back(fp);
    froc.push_back(fn);

    if (print) {
        std::cout << "tp: " << tp << " fn: " << fn << " fp: " << fp << std::endl;
    }

    return iou;
}

//получаем точки для кривой Froc
std::vector<cv::Point3d> Froc(std::vector<std::vector<int>>& qa, std::vector<cv::Point>& centers0, std::vector<float>& radii0, std::vector<cv::Point>& centers, std::vector<float>& radii, float step) {
    std::vector<cv::Point3d> froc_dots;

    //при iou_threshold = 0
    std::vector<int> froc0;
    std::cout << " threshold: " << 0 << std::endl;
    qualityAssessment(froc0, centers0, radii0, centers, radii, -1);
    int tp0 = froc0[0];
    int fp0 = froc0[1];
    int fn0 = froc0[2];

    qa.push_back(froc0);
    double tpr0;
    if (tp0 == 0) {
        tpr0 = 0;
    } else {
        tpr0 = tp0 / static_cast<double>(tp0 + fn0);
    }
    double fpN0 = fp0 / static_cast<double>(radii0.size());
    std::cout << std::fixed << "tpr: " << tpr0 << " fpN: " << fpN0 << std::endl;
    cv::Point3d point0(tpr0, fpN0, 0);
    froc_dots.push_back(point0);

    for (float iou_threshold = step; iou_threshold <= 1.0; iou_threshold += step) {
        std::vector<int> froc;
        qualityAssessment(froc, centers0, radii0, centers, radii, iou_threshold);
        std::cout << std::fixed << " threshold: " << iou_threshold << std::endl;

        int tp = froc[0];
        int fp = froc[1];
        int fn = froc[2];
        qa.push_back(froc);


        double tpr;
        if (tp == 0) {
            tpr = 0;
        } else {
            tpr = tp / static_cast<double>(tp + fn);
        }

        double fpN = fp / static_cast<double>(radii0.size());

        std::cout << std::fixed << "tpr: " << tpr << " fpN: " << fpN  << std::endl;

        cv::Point3d point(tpr, fpN, iou_threshold);
        froc_dots.push_back(point);
    }

    float iou_threshold = 1.0;
    std::cout << std::fixed << " threshold: " << iou_threshold << std::endl;
    std::vector<int> froc;
    qualityAssessment(froc, centers0, radii0, centers, radii, iou_threshold);

    int tp = froc[0];
    int fp = froc[1];
    int fn = froc[2];
    qa.push_back(froc);


    double tpr;
    if (tp == 0) {
        tpr = 0;
    } else {
        tpr = tp / static_cast<double>(tp + fn);
    }
    double fpN = fp / static_cast<double>(radii0.size());

    std::cout << std::fixed << "tpr: " << tpr << " fpN: " << fpN << std::endl;

    cv::Point3d point(tpr, fpN, iou_threshold);
    froc_dots.push_back(point);

    return froc_dots;
}

//создаём json с значениями tp, fp, fn, threshhold для оценки качества
void makeQaJson(std::string file_name,std::vector<std::vector<int>>& qa, float step) {
    //открываем запись в json
    cv::FileStorage js(file_name + ".json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

    std::ostringstream ss;
    std::string floatString;

    float thresh = 0;
    js << "qualityAssesment" << "[";
    for (int i = 0; i < qa.size() - 1; i++) {
        ss.precision(4);
        ss << std::fixed << thresh; // Установка формата вывода
        floatString = ss.str();
        if (floatString == "1.0000") {
            break;
        }
        js << "{";

        int tp = qa[i][0];
        js << "tp" << tp;

        int fp = qa[i][1];
        js << "fp" << fp;

        int fn = qa[i][2];
        js << "fn" << fn;

        
        js << "threshold" << floatString;
        thresh += step;
        ss.str("");

        js << "}";
    }

    js << "{";

    int tp = qa[qa.size() - 1][0];
    js << "tp" << tp;
    int fp = qa[qa.size() - 1][1];
    js << "fp" << fp;
    int fn = qa[qa.size() - 1][2];
    js << "fn" << fn;
    js << "threshold" << 1;

    js << "}";

    js << "]";

    js.release();

    std::cout << file_name << " successfully generate" << std::endl;
}

//создаём json с точками для Froc прямой
void makeFrocJson(std::string file_name,std::vector<cv::Point3d>& froc_dots) {
    //открываем запись в json
    cv::FileStorage js(file_name + ".json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

    std::ostringstream ss;
    std::string floatString;
    js << "dots" << "[";
    for (int i = 0; i < froc_dots.size(); i++) {
        cv::Point3d point = froc_dots[i];
        js << "{";

        ss << std::fixed << point.x; // Установка формата вывода
        floatString = ss.str();
        js << "tpr" << floatString;
        ss.str("");

        ss << std::fixed << point.y; // Установка формата вывода
        floatString = ss.str();
        js << "fpN" << floatString;
        ss.str("");

        ss << std::fixed << point.z; // Установка формата вывода
        floatString = ss.str();
        js << "threshold" << floatString;
        js << "}";
        ss.str("");
    }
    js << "]";

    js.release();

    std::cout << file_name << " successfully generate" << std::endl;
}

std::vector<cv::Point3d> readFrocJson(std::string file_name) {
    cv::FileStorage js(file_name + ".json", cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    std::vector<cv::Point3d> dots;

    cv::FileNode js_dots;
    if (js["dots"].empty()) {
        std::cerr << "Field 'dots' is empty" << std::endl;
        js.release();
    } else if (js["dots"].type() != cv::FileNode::SEQ) {
        std::cerr << "Invalid file format! Not array" << std::endl;
        js.release();
    } else {
        js_dots = js["dots"];
    }

    for (cv::FileNodeIterator i = js_dots.begin(); i != js_dots.end(); i++) {
        cv::FileNode objectNode = *i;
        double tpr = 0;
        double fpN = 0;
        double threshold = 0;

        if (objectNode["tpr"].empty()) {
            std::cerr << "Field 'tpr' is empty" << std::endl;
            js.release();
        } else {
            tpr = std::stod(objectNode["tpr"]);
        }

        if (objectNode["fpN"].empty()) {
            std::cerr << "Field 'fpN' is empty" << std::endl;
            js.release();
        } else {
            fpN = std::stod(objectNode["fpN"]);
        }

        if (objectNode["threshold"].empty()) {
            std::cerr << "Field 'threshold' is empty" << std::endl;
            js.release();
        } else {
            threshold = std::stod(objectNode["threshold"]);
        }

        cv::Point3d point(tpr, fpN, threshold);

        dots.push_back(point);
    }

    js.release();

    std::cout << file_name << " successfully read" << std::endl;

    return dots;
}

std::vector<double> findMaxMinDotX(std::vector<cv::Point3d>& dots) {
    std::vector<double> maxmin;
    double max_x = -1;
    double min_x = std::numeric_limits<double>::infinity();
    for (int i = 0; i < dots.size(); i++) {
        //в y записанно fpN
        if (dots[i].y > max_x) {
            max_x = dots[i].y;
        }

        if (dots[i].y < min_x) {
            min_x = dots[i].y;
        }
    }

    maxmin.push_back(max_x);
    maxmin.push_back(min_x);
    return maxmin;
}

//площадь под графиком
double AuC(std::vector<cv::Point3d> dots) {
    double area = 0;

    double x_prev = dots[0].y;
    double y_prev = dots[0].x;

    std::cout << "area: " << area << std::endl;
    for (int i = 1; i < dots.size(); i++) {
        double x_cur = dots[i].y;
        double y_cur = dots[i].x;

        if (x_prev != x_cur || y_prev != y_cur) {
            area += (y_prev + y_cur) / 2 * abs(x_cur - x_prev);
            std::cout << "y_prev: " << y_prev << " y_cur: " << y_cur << " x_prev: " << x_prev << " x_cur: " << x_cur <<std::endl;
            std::cout << "+ " << (y_prev + y_cur) / 2 * abs(x_cur - x_prev) <<std::endl;

            x_prev = x_cur;
            y_prev = y_cur;
        }
    }

    std::cout << "area: " << area << std::endl;

    return area;
}

//рисуем кривые Froc получая данные из json
cv::Mat drawFrocCruveFromFiles(std::vector<std::string>& file_names, std::vector<cv::Scalar>& colors, std::vector<std::string>& names, std::vector<double>& auc) {
    int size_coor_sys = 455; //размер окна с графиками
    std::vector<std::vector<cv::Point3d>> all_dots;

    for (int i = 0; i < file_names.size(); i++) {
        std::vector<cv::Point3d> dots = readFrocJson(file_names[i]);

        std::vector<double> minMaxX = findMaxMinDotX(dots);
        double max_x = minMaxX[0];
        double min_x = minMaxX[1];

        //нормируем от 0.0 до 1.0
        double a = 1.0 / (max_x - min_x);
        double b = (-min_x * 1.0 / (max_x - min_x));

        std::cout << names[i] << std::endl;
        for (int j = 0; j < dots.size(); j++) {
            dots[j].y = dots[j].y * a + b;
            std::cout << "x: " << dots[j].y << " y: " << dots[j].x << std::endl;
        }

        //получаем значения площадей под кривыми
        double area_under_cruve = AuC(dots);
        
        auc.push_back(area_under_cruve);

        all_dots.push_back(dots);
    }

    for (int i = 0; i < names.size(); i++) {
        std::cout << std::fixed << names[i] << " auc: " << auc[i] << std::endl;
    }


    cv::Mat cruve(size_coor_sys + 1 + 50*(colors.size() + 2), size_coor_sys + 1, CV_8UC3, cv::Scalar(255, 255, 255));

    //линия 0.5
    for (int i = 0; i <= size_coor_sys; i += size_coor_sys / 10 + 10) {
        cv::line(cruve, cv::Point2d(i, i), cv::Point2d(i + 20, i + 20), cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    }
    cv::line(cruve, cv::Point2d(1, size_coor_sys + 50), cv::Point2d(3, size_coor_sys + 50), cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::line(cruve, cv::Point2d(8, size_coor_sys + 50), cv::Point2d(11, size_coor_sys + 50), cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::line(cruve, cv::Point2d(16, size_coor_sys + 50), cv::Point2d(21, size_coor_sys + 50), cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(cruve, "Base line", cv::Point2d(25, size_coor_sys + 50 + 5), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 0.5);

    cv::line(cruve, cv::Point2d(0, 0), cv::Point2d(0, size_coor_sys), cv::Scalar(0, 0, 0), 2);
    cv::putText(cruve, "0", cv::Point2d(0, size_coor_sys + 30), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 0.5);
    cv::line(cruve, cv::Point2d(0, size_coor_sys), cv::Point2d(size_coor_sys, size_coor_sys), cv::Scalar(0, 0, 0), 2);
    cv::putText(cruve, "1", cv::Point2d(size_coor_sys - 25, size_coor_sys + 30), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 0.5);

    for (int i = 0; i < all_dots.size(); i++) {
        std::vector<cv::Point3d> dots = all_dots[i];
        cv::Scalar color = colors[i];

        //легенда
        cv::line(cruve, cv::Point2d(1, size_coor_sys + 50 * (i + 2)), cv::Point2d(21, size_coor_sys + 50 * (i + 2)), color, 4);
        cv::putText(cruve, names[i], cv::Point2d(25, size_coor_sys + 50 * (i + 2) + 5), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 0.5);

        //отображение в декартовой прямоугольной плоскости размером size_coor_sys + 1 на size_coor_sys + 1
        int new_x_prev = static_cast<int>(dots[0].y * size_coor_sys + 0);
        int new_y_prev = static_cast<int>((dots[0].x) * (-size_coor_sys) + size_coor_sys);

        std::cout << names[i] << std::endl;
        std::cout << "x: " << new_x_prev << " y: " << new_y_prev << std::endl;

        for (int j = 1; j < dots.size(); j++) {
            int new_x_cur = static_cast<int>(dots[j].y * size_coor_sys + 0);
            int new_y_cur = static_cast<int>((dots[j].x) * (-size_coor_sys) + size_coor_sys);

            if (new_x_prev != new_x_cur || new_y_prev != new_y_cur) {
                cv::line(cruve, cv::Point2d(new_x_prev, new_y_prev), cv::Point2d(new_x_cur, new_y_cur), color, 2);  
                new_x_prev = new_x_cur;
                new_y_prev = new_y_cur;
                std::cout << "x: " << new_x_prev << " y: " << new_y_prev << std::endl;
            }
        }
    }

    cv::imshow("Cruve", cruve);

    return cruve;
}

//создаём json с результатами auc
void makeAuCJson(std::string file_name, std::vector<std::string>& names, std::vector<double>& auc) {
    //открываем запись в json
    cv::FileStorage js(file_name + ".json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

    std::ostringstream ss;
    std::string floatString;
    
    for (int i = 0; i < names.size(); i++) {
        ss << std::fixed << auc[i]; // Установка формата вывода
        floatString = ss.str();
        js << names[i] << floatString;
        ss.str("");
    }

    js.release();

    std::cout << file_name << " successfully generate" << std::endl;
}

int main (int argc, char** argv) {
        std::string prefix = "../prj.lab/result_lab04_lab06/";
        std::string file_name_gen = "generate";
        std::string file_name_res = "result";
        std::string file_name_config = "config";
        std::string file_name_gen_json = "../prj.lab/lab04/results/generate";
        std::string folder_name;
        std::string subfolder_name;
        std::string file_name_froc;
        std::string file_name_qa;

        int config_flag = 0;
        int read_flag = 0;
        int gui_flag = 0;
        std::string detection_name = "log";
        std::string binarization_name = "";
        if (argc > 1) {
            for (int i = 1; i < argc; i++) {
                std::string arg = argv[i];

                if (arg == "-r") {
                    file_name_gen_json = argv[i + 1];
                    read_flag = 1;
                    i++;
                } else if (arg == "-config") {
                    file_name_config = argv[i + 1];
                    config_flag = 1;
                    i++;
                } else if (arg == "-d") {
                    detection_name = argv[i + 1];
                    i++;
                } else if (arg == "-b") {
                    binarization_name = argv[i + 1];
                    i++;
                } else if (arg == "gui") {
                    gui_flag = 1;
                }
            }
        }

        int color_background;
        int nobj;
        float radius[2];
        std::vector<float> contrast;
        float blur[2];
        float noise[2];
        float binary[2];
        float iou_threshold;
        float step;

        color_background = 30;
        nobj = 3;
        radius[0] = 5.0;
        radius[1] = 52.0;
        contrast = { 0.5, 0.55, 0.67};
        blur[0] = 1.0;
        blur[1] = 0.8;
        noise[0] = 8; 
        noise[1] = 4;
        binary[0] = 30;
        binary[1]  = 0.0;
        iou_threshold = 0.8;
        step = 0.01;

        if (binarization_name != "") {
            binarization_name += "_";
        }

        //конфиг
        if (config_flag == 0) {
            //создание папки
            folder_name = std::to_string(nobj) + "+" + std::to_string(contrast.size());
            createFolder(prefix + folder_name);
            folder_name += "/";

            //подпапка
            subfolder_name = detection_name + "_" + binarization_name;
            createFolder(prefix + folder_name + subfolder_name);

            subfolder_name += "/";

            file_name_config = prefix + folder_name +  subfolder_name + detection_name + "_" + binarization_name + file_name_config + "_" + std::to_string(nobj) + "+" + std::to_string(contrast.size());

            createConfig(color_background, nobj, radius, contrast, blur, noise, binary, iou_threshold, step, file_name_config);
        } else if (config_flag == 1) {
            contrast.clear();

            readFromConfig(color_background, nobj, radius, contrast, blur, noise, binary, iou_threshold, step, file_name_config);
            //создание папки
            folder_name = std::to_string(nobj) + "+" + std::to_string(contrast.size());
            createFolder(prefix + folder_name);
            folder_name += "/";

            //подпапка
            subfolder_name = detection_name + "_" + binarization_name;
            createFolder(prefix + folder_name + subfolder_name);
            subfolder_name += "/";

            file_name_config = prefix + folder_name + subfolder_name + detection_name + "_" + binarization_name + "config_" + std::to_string(nobj) + "+" + std::to_string(contrast.size());
            createConfig(color_background, nobj, radius, contrast, blur, noise, binary, iou_threshold, step, file_name_config);
        }

        //генерация изображения
        std::vector<cv::Point> centers0;
        std::vector<float> radii0;
        cv::Mat image;
        if (read_flag == 0) {
            image = testImage(color_background, nobj, radius, contrast, blur, centers0, radii0);
        } else {
            image = createImageFromFile(file_name_gen_json, color_background, blur, centers0, radii0);
        }

        addNoise(image, noise);

        std::vector<cv::Point> centers;
        std::vector<float> radii;

        //анализ компонент связности
        if (detection_name == "con") {
            cv::Mat binary_image = image.clone();
            if (binarization_name == "brad_") {
                if (gui_flag == 1) {
                    cv::namedWindow("Bradley", cv::WINDOW_AUTOSIZE); // Create Window
                    cv::createTrackbar("ksize", "Bradley", &binary1, 50, BradleyThreshHoldGui, &binary_image);
                    cv::createTrackbar("percent", "Bradley", &binary2, 100, BradleyThreshHoldGui, &binary_image);
                    cv::waitKey(0);
                    cv::destroyAllWindows();
                    binary_image = image.clone();
                    binary[0] = static_cast<float>(binary1);
                    binary[1] = static_cast<float>(binary2) / 100;
                }    

                BradleyThreshHold(binary_image, binary);
            } else if (binarization_name == "bern_") {
                if (gui_flag == 1) {
                    cv::namedWindow("Bernsen", cv::WINDOW_AUTOSIZE); // Create Window
                    cv::createTrackbar("ksize", "Bernsen", &binary1, 50, BernsenThreshHoldGui, &binary_image);
                    cv::createTrackbar("percent", "Bernsen", &binary2, 100, BernsenThreshHoldGui, &binary_image);
                    cv::waitKey(0);
                    cv::destroyAllWindows();
                    binary_image = image.clone();
                    binary[0] = static_cast<float>(binary1);
                    binary[1] = static_cast<float>(binary2) / 100;
                }

                BernsenThreshHold(binary_image, binary);
            } else if (binarization_name == "ni_") {
                if (gui_flag == 1) {
                    cv::namedWindow("Niblack", cv::WINDOW_AUTOSIZE); // Create Window
                    cv::createTrackbar("ksize", "Niblack", &binary1, 50, NiblackThreshHoldGui, &binary_image);
                    cv::createTrackbar("percent", "Niblack", &binary2, 100, NiblackThreshHoldGui, &binary_image);
                    cv::waitKey(0);
                    cv::destroyAllWindows();
                    binary_image = image.clone();
                    binary[0] = static_cast<float>(binary1);
                    binary[1] = static_cast<float>(binary2) / 100;
                }

                NiblackThreshHold(binary_image, binary);
            }
            
            outputImage("binary", binary_image);
            cv::imwrite(prefix + folder_name + subfolder_name + detection_name + "_" + binarization_name + "binary_" + std::to_string(nobj) + "+" + std::to_string(contrast.size()) + ".jpg", binary_image);

            file_name_res = binarization_name + file_name_res;

            detectBlobsConnected(centers, radii, binary_image, M_PI * radius[0] * radius[0] - 10, M_PI * radius[1] * radius[1] + 20);
            createConfig(color_background, nobj, radius, contrast, blur, noise, binary, iou_threshold, step, file_name_config);
        }

        double min_sigma = 1.5;
        double max_sigma = 5.3; 
        double step_sigma = 0.1;
        int pyr_depth = static_cast<int>(ceil(log2f(radius[1] / (max_sigma * sqrt(2))))) + 1; 
        int count = contrast.size() + 5;
        float thresh_iou = 0.01;
        
        //LoG
        if (detection_name == "log") {
            // Обнаружение блобов с использованием алгоритма LoG
            detectBlobsLoG(centers, radii, image, min_sigma, max_sigma, step_sigma, pyr_depth, count, thresh_iou, prefix + folder_name + subfolder_name, 2);
        }

        //DoG
        if (detection_name == "dog") {
            // Обнаружение блобов с использованием алгоритма DoG
            detectBlobsDoG(centers, radii, image, min_sigma, max_sigma + 1, step_sigma + 0.1, pyr_depth, count, thresh_iou,  prefix + folder_name + subfolder_name, 3);
        }

        cv::Mat res;
        cv::Size image_size = image.size();
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
        res = image.clone();

        for (int i = 0; i < centers.size(); i++) {
            cv::circle(res, centers[i], 3, cv::Scalar(0, 0, 255), -1);
            cv::circle(res, centers[i], radii[i], cv::Scalar(0, 0, 255), 2);
        }

        std::vector<int> froc;
        std::vector<float> iou;
        iou = qualityAssessment(froc, centers0, radii0, centers, radii, iou_threshold);

        std::vector<cv::Point3d> froc_dots;
        std::vector<std::vector<int>> qa;
        froc_dots = Froc(qa, centers0, radii0, centers, radii, step);

        //сохранение резултатов в файлы
        file_name_qa = detection_name + "_" + binarization_name + "qa_" + std::to_string(nobj) + "+" + std::to_string(contrast.size());
        makeQaJson(prefix + folder_name + subfolder_name + file_name_qa, qa, step);

        file_name_froc = detection_name + "_" + binarization_name + "froc_" + std::to_string(nobj) + "+" + std::to_string(contrast.size());
        makeFrocJson(prefix + folder_name + subfolder_name + file_name_froc, froc_dots);
        
        file_name_res = detection_name + "_" + file_name_res + "_" + std::to_string(nobj) + "+" + std::to_string(contrast.size());
        makeResJson(prefix + folder_name + subfolder_name + "generate_" + file_name_res, image.size(), centers0, radii0, contrast, iou, iou_threshold, froc);

        file_name_gen = file_name_gen + "_" + std::to_string(nobj) + "+" + std::to_string(contrast.size()) + ".jpg";
        cv::imshow("image", image);
        cv::imwrite(prefix + folder_name + file_name_gen, image);

        file_name_res = file_name_res + ".jpg";
        cv::imshow("resut", res);
        cv::imwrite(prefix + folder_name + subfolder_name + file_name_res, res);

        //cv::waitKey(0);
        return 0;
}