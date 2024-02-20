#include <iostream>
#include <opencv2/opencv.hpp>

void gammaCorrection(cv::Mat& img, float gamma) {
    unsigned char lut[256];
    for (int i = 0; i < 256; i++) {
        lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
    }
    cv::LUT(img, cv::Mat(1, 256, CV_8UC1, lut), img);
}

int main(int argc, char** argv) {

    int s = 3;
    int h = 30;
    float gamma = 2.4;
    std::string filename = "";

    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            std::string str = std::string(argv[i]);
            if (str == "-s") {
                s = std::stoi(argv[i + 1]);
                i++;
            }
            else if (str == "-h") {
                h = std::stoi(argv[i + 1]);
                ++i;
            }
            else if (str == "-gamma") {
                gamma = std::stoi(argv[i + 1]);
                ++i;
            }
            else {
                filename = str;
            }
        }
    }

    int image_w = s * 256;
    int image_h = h * 2;

    cv::Mat image = cv::Mat::zeros(image_h, image_w, CV_8UC1);

    for (int i = 0; i < 256; i++) {
        cv::Rect rect(i * s, 0, s, 2 * h);
        image(rect) = i;
    }

    cv::Mat roi = image(cv::Rect(0, h, image_w, h));

    gammaCorrection(roi, gamma);

    if (filename == "") {
        cv::imshow("Image2", image);
        cv::waitKey(0);
    }
    else {
        cv::imwrite("../" + filename, image);
    }

    return 0;
}