#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>

class DocumentSegmentation {
public:
    cv::Mat segmentImage(const cv::Mat& inputImage) {
        cv::Mat hsvImage;
        cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

        std::vector<float> hueHistogram(360, 0);
        calculateHueHistogram(hsvImage, hueHistogram);

        std::vector<float> smoothedHistogram = smoothHistogram(hueHistogram);

        std::vector<int> peaks = findPeaks(smoothedHistogram);
        std::vector<int> clusters = clusterHues(smoothedHistogram, peaks);

        return applySegmentation(inputImage, hsvImage, clusters);
    }

    cv::Mat visualizeHistogram(const std::vector<float>& histogram) {
        int hist_w = 512, hist_h = 400;
        int bin_w = cvRound((double)hist_w / histogram.size());
        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

        float max = *std::max_element(histogram.begin(), histogram.end());
        for (int i = 0; i < histogram.size(); i++) {
            int height = cvRound((float)histogram[i] / max * hist_h);
            cv::line(histImage, cv::Point(bin_w * i, hist_h),
                cv::Point(bin_w * i, hist_h - height),
                cv::Scalar(i / 2, 255, 255), bin_w);
        }

        cv::cvtColor(histImage, histImage, cv::COLOR_HSV2BGR);
        return histImage;
    }

    std::vector<cv::Mat> getSegmentedLayers(const cv::Mat& inputImage, const cv::Mat& hsvImage, const std::vector<int>& clusters) {
        std::vector<cv::Mat> layers;
        int numClusters = *std::max_element(clusters.begin(), clusters.end()) + 1;

        for (int i = 0; i < numClusters; ++i) {
            cv::Mat layer = cv::Mat::zeros(inputImage.size(), CV_8UC3);
            for (int y = 0; y < inputImage.rows; ++y) {
                for (int x = 0; x < inputImage.cols; ++x) {
                    cv::Vec3b hsvPixel = hsvImage.at<cv::Vec3b>(y, x);
                    int hue = static_cast<int>(hsvPixel[0] * 2);
                    if (clusters[hue] == i) {
                        layer.at<cv::Vec3b>(y, x) = inputImage.at<cv::Vec3b>(y, x);
                    }
                }
            }
            layers.push_back(layer);
        }

        return layers;
    }

    void calculateHueHistogram(const cv::Mat& hsvImage, std::vector<float>& histogram) {
        for (int y = 0; y < hsvImage.rows; ++y) {
            for (int x = 0; x < hsvImage.cols; ++x) {
                cv::Vec3b pixel = hsvImage.at<cv::Vec3b>(y, x);
                float hue = pixel[0] * 2.0f;
                histogram[static_cast<int>(hue)]++;
            }
        }
    }

    std::vector<float> smoothHistogram(const std::vector<float>& histogram, int kernelSize = 5) {
        std::vector<float> smoothed(histogram.size());
        cv::Mat histMat(1, histogram.size(), CV_32F);
        std::memcpy(histMat.data, histogram.data(), histogram.size() * sizeof(float));

        cv::GaussianBlur(histMat, histMat, cv::Size(kernelSize, 1), 0);
        std::memcpy(smoothed.data(), histMat.data, histogram.size() * sizeof(float));

        return smoothed;
    }

    std::vector<int> findPeaks(const std::vector<float>& histogram) {
        std::vector<int> peaks;
        for (int i = 1; i < histogram.size() - 1; ++i) {
            if (histogram[i] > histogram[i - 1] && histogram[i] > histogram[i + 1]) {
                // Check if the peak is significant
                if (histogram[i] > 0.1 * (*std::max_element(histogram.begin(), histogram.end()))) {
                    peaks.push_back(i);
                }
            }
        }
        return peaks;
    }

    std::vector<int> clusterHues(const std::vector<float>& histogram, const std::vector<int>& peaks) {
        std::vector<int> clusters(360, -1);
        for (int i = 0; i < peaks.size(); ++i) {
            int left = i > 0 ? (peaks[i] + peaks[i - 1]) / 2 : 0;
            int right = i < peaks.size() - 1 ? (peaks[i] + peaks[i + 1]) / 2 : 359;

            for (int j = left; j <= right; ++j) {
                clusters[j] = i;
            }
        }
        return clusters;
    }

private:
    cv::Mat applySegmentation(const cv::Mat& inputImage, const cv::Mat& hsvImage, const std::vector<int>& clusters) {
        cv::Mat segmented = cv::Mat::zeros(inputImage.size(), CV_8UC3);
        std::vector<cv::Vec3b> clusterColors;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);

        for (int i = 0; i < *std::max_element(clusters.begin(), clusters.end()) + 1; ++i) {
            clusterColors.push_back(cv::Vec3b(dis(gen), dis(gen), dis(gen)));
        }

        for (int y = 0; y < segmented.rows; ++y) {
            for (int x = 0; x < segmented.cols; ++x) {
                cv::Vec3b hsvPixel = hsvImage.at<cv::Vec3b>(y, x);
                int hue = static_cast<int>(hsvPixel[0] * 2); // Convert back to [0, 360] range
                int cluster = clusters[hue];
                if (cluster != -1) {
                    segmented.at<cv::Vec3b>(y, x) = clusterColors[cluster];
                }
            }
        }

        return segmented;
    }
};

int main() {
    cv::Mat inputImage = cv::imread("C:/cv/misis2024s-21-02-suchoruchenkov-m-e/prj.cw/image9.png");
    if (inputImage.empty()) {
        std::cout << "Failed to load image" << std::endl;
        return -1;
    }

    DocumentSegmentation segmenter;
    cv::Mat segmentedImage = segmenter.segmentImage(inputImage);

    // Visualize histogram
    cv::Mat hsvImage;
    cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);
    std::vector<float> hueHistogram(360, 0);
    segmenter.calculateHueHistogram(hsvImage, hueHistogram);
    std::vector<float> smoothedHistogram = segmenter.smoothHistogram(hueHistogram);
    cv::Mat histogramImage = segmenter.visualizeHistogram(smoothedHistogram);

    // Get individual layers
    std::vector<int> peaks = segmenter.findPeaks(smoothedHistogram);
    std::vector<int> clusters = segmenter.clusterHues(smoothedHistogram, peaks);
    std::vector<cv::Mat> layers = segmenter.getSegmentedLayers(inputImage, hsvImage, clusters);

    // Display results
    cv::imshow("Original Image", inputImage);
    cv::imshow("Segmented Image", segmentedImage);
    cv::imshow("Hue Histogram", histogramImage);

    // Display each layer
    for (size_t i = 0; i < layers.size(); ++i) {
        cv::imshow("Layer " + std::to_string(i), layers[i]);
    }

    cv::waitKey(0);
    return 0;
}