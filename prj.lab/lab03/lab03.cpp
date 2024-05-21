#include <opencv2/opencv.hpp>
#include <iostream>

//рисование гистограммы
cv::Mat drawHistogram(cv::Mat image) {
    cv::Mat background(256, 260, CV_8UC1, cv::Scalar(230));

    cv::Mat hist;

    const int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    //указатель на изображение, количество изображений, индекс канала, маска, выходной массив, количество измерений, указател) на количество бинов, указатель на диапазон значений
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    //нормализация 
    cv::normalize(hist, hist, 0, 230, cv::NORM_MINMAX);

    //отрисовка гистограмы
    for (int i = 0; i < histSize; i++) {
        cv::rectangle(background, cv::Rect(i + 2, 256 - hist.at<float>(i, 0), 1, hist.at<float>(i, 0)), cv::Scalar(0), -1);
    }

    return background;
}

cv::Mat drawHistogram3Channels(cv::Mat image) {
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    cv::Mat background(256, 3 * 260, CV_8UC1, cv::Scalar(230));

    for (int i = 0; i < 3; i++) {
        cv::Mat hist;
        cv::Mat background0(256, 260, CV_8UC1, cv::Scalar(230));

        const int histSize = 256;
        float range[] = { 0, 256 };
        const float* histRange = { range };

        cv::calcHist(&channels[i], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

        cv::normalize(hist, hist, 0, 230, cv::NORM_MINMAX);

        for (int j = 0; j < histSize; j++) {
            cv::rectangle(background0, cv::Rect(j + 2, 256 - hist.at<float>(j, 0), 1, hist.at<float>(j, 0)), cv::Scalar(0), -1);
        }

        if (i == 0) {
            cv::putText(background0, "Blue", cv::Point(124, 253), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
        }
        else if (i == 1) {
            cv::putText(background0, "Green", cv::Point(124, 253), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
        }
        else if (i == 2) {
            cv::putText(background0, "Red", cv::Point(124, 253), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
        }

        background0.copyTo(background(cv::Rect(i * 256, 0, 260, 256)));
    }

    return background;
}

void quanteles(cv::Mat& image, float blackQuantile, float whiteQuantile, int& blackTreshold, int& whiteTreshold) {
    blackTreshold = 0;
    whiteTreshold = 255;

    float countPixels = image.rows * image.cols;

    float countBlack = countPixels * blackQuantile;
    float countWhite = countPixels * whiteQuantile;

    cv::Mat hist;
    const int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    float sumBlack = 0;
    int flagBlack = 0;
    float sumWhite = 0;
    int flagWhite = 0;
    for (int i = 0; i < 256; i++) {
        if ((sumBlack + hist.at<float>(i, 0)) <= countBlack) {
            sumBlack += hist.at<float>(i, 0);
        }
        else if (flagBlack == 0) {
            blackTreshold = i;
            flagBlack = 1;
        }

        if ((sumWhite + hist.at<float>(255 - i, 0)) <= countWhite) {
            sumWhite += hist.at<float>(255 - i, 0);
        }
        else if (flagWhite == 0) {
            whiteTreshold = 255 - i;
            flagWhite = 1;
        }

        if (flagBlack == 1 && flagWhite == 1) {
            break;
        }
    }
}

void autoContrastOneChannel(cv::Mat& image, float blackQuantile, float whiteQuantile) {
    int clow, chigh;

    quanteles(image, blackQuantile, whiteQuantile, clow, chigh);

    int cmin = 0;
    int cmax = 255;

    cv::Mat lut(1, 256, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < 256; i++) {
        if (i <= clow) {
            lut.at<uchar>(0, i) = static_cast<uchar>(cmin);
        }
        else if (i >= chigh) {
            lut.at<uchar>(0, i) = static_cast<uchar>(cmax);
        }
        else {
            lut.at<uchar>(0, i) = static_cast<uchar>(cmin + (i - clow) * (cmax - cmin) / (chigh - clow));
        }
    }
    cv::LUT(image, lut, image);
    std::cout << clow << " " << chigh << std::endl;
}

void autoContrastChannelWise(cv::Mat& image, float blackQuantile, float whiteQuantile) {
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    int cmin = 0;
    int cmax = 255;

    for (int i = 0; i < 3; i++) {
        int clow, chigh;
        quanteles(channels[i], blackQuantile, whiteQuantile, clow, chigh);
        cv::Mat lut(1, 256, CV_8UC1, cv::Scalar(0));

        for (int j = 0; j < 256; j++) {
            if (j <= clow) {
                lut.at<uchar>(0, j) = static_cast<uchar>(cmin);
            }
            else if (j >= chigh) {
                lut.at<uchar>(0, j) = static_cast<uchar>(cmax);
            }
            else {
                lut.at<uchar>(0, j) = static_cast<uchar>(cmin + (j - clow) * (cmax - cmin) / (chigh - clow));
            }
        }
        cv::LUT(channels[i], lut, channels[i]);
        std::cout << clow << " " << chigh << std::endl;
    }

    cv::merge(channels, image);
}

void autoContrastJoint(cv::Mat& image, float blackQuantile, float whiteQuantile) {
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    int cmin = 0;
    int cmax = 255;

    int clow, chigh;
    quanteles(channels[0], blackQuantile, whiteQuantile, clow, chigh);

    for (int i = 1; i < 3; i++) {
        int clow_temp, chigh_temp;
        quanteles(channels[i], blackQuantile, whiteQuantile, clow_temp, chigh_temp);

        clow = (clow > clow_temp) ? clow_temp : clow;
        chigh = (chigh < chigh_temp) ? chigh_temp : chigh;
    }

    cv::Mat lut(1, 256, CV_8UC3, cv::Scalar(0));
    for (int i = 0; i < 256; i++) {
        if (i <= clow) {
            uchar val = static_cast<uchar>(cmin);
            lut.at<cv::Vec3b>(0, i) = cv::Vec3b(val, val, val);
        }
        else if (i >= chigh) {
            uchar val = static_cast<uchar>(cmax);
            lut.at<cv::Vec3b>(0, i) = cv::Vec3b(val, val, val);
        }
        else {
            uchar val = static_cast<uchar>(cmin + (i - clow) * (cmax - cmin) / (chigh - clow));
            lut.at<cv::Vec3b>(0, i) = cv::Vec3b(val, val, val);
        }
    }
    cv::LUT(image, lut, image);
    std::cout << clow << " " << chigh << std::endl;
}

int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "image not specified" << std::endl;
        return -1;
    }

    float blackQuantile = 0.3;
    float whiteQuantile = 0.7;

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);

    cv::imshow("Start", image);

    cv::Mat baseMat(2 * 256, image.channels() * 260, CV_8UC1);

    if (image.channels() == 1) {
        std::cout << "gray" << std::endl;


        cv::Mat hist = drawHistogram(image);
        hist.copyTo(baseMat(cv::Rect(0 * 256, 0, 260, 256)));
        autoContrastOneChannel(image, blackQuantile, whiteQuantile);
        cv::imshow("End", image);

        cv::Mat hist1 = drawHistogram(image);
        hist1.copyTo(baseMat(cv::Rect(0 * 256, 1 * 256, 260, 256)));
        cv::imshow("hist", baseMat);
    }
    else if (image.channels() == 3) {
        std::cout << "color" << std::endl;

        cv::Mat imageClone = image.clone();


        cv::Mat hist0 = drawHistogram3Channels(imageClone);
        hist0.copyTo(baseMat(cv::Rect(0 * 256, 0, imageClone.channels() * 260, 256)));
        autoContrastChannelWise(imageClone, blackQuantile, whiteQuantile);
        cv::imshow("End3channel", imageClone);

        cv::Mat hist1 = drawHistogram3Channels(imageClone);
        hist1.copyTo(baseMat(cv::Rect(0 * 256, 256, imageClone.channels() * 260, 256)));
        cv::imshow("hist3channel", baseMat);

        baseMat = cv::Mat::zeros(2 * 256, image.channels() * 260, CV_8UC1);
        hist0 = drawHistogram3Channels(image);
        hist0.copyTo(baseMat(cv::Rect(0 * 256, 0, image.channels() * 260, 256)));
        autoContrastJoint(image, blackQuantile, whiteQuantile);
        cv::imshow("EndJoint", image);

        hist1 = drawHistogram3Channels(image);
        hist1.copyTo(baseMat(cv::Rect(0 * 256, 256, image.channels() * 260, 256)));
        cv::imshow("histJoint", baseMat);
    }



    cv::waitKey(0);
    return 0;
}