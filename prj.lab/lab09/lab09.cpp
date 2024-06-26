#include <opencv2/opencv.hpp>

void greyWorldCorrection(cv::Mat& image) {
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    cv::Scalar meanValue = cv::mean(image);
    double meanGray = (meanValue[0] + meanValue[1] + meanValue[2]) / 3.0;

    for (int i = 0; i < 3; ++i) {
        double scaleFactor = meanGray / meanValue[i];
        channels[i] = channels[i] * scaleFactor;
    }
    cv::merge(channels, image);
}

int main() {
    cv::Mat originalImage = cv::imread("../prj.lab/lab09/1.jpg");

    if (originalImage.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }

    cv::Mat correctedImage = originalImage.clone();
    cv::imshow("Input image", originalImage);

    greyWorldCorrection(correctedImage);

    cv::imshow("Grey World", correctedImage);
    cv::imwrite("../prj.lab/lab01/images/lab09_output.png", correctedImage);

    // Evaluate output image quality
    double meanAbsoluteError = 0.0;
    double peakSignalToNoiseRatio = 0.0;
    double structuralSimilarity = 0.0;

    // Mean Absolute Error (MAE)
    cv::Mat difference;
    cv::absdiff(originalImage, correctedImage, difference);
    meanAbsoluteError = cv::mean(difference)[0];
    std::cout << "Mean Absolute Error (MAE): " << meanAbsoluteError << std::endl;

    // Peak Signal-to-Noise Ratio (PSNR)
    cv::Scalar originalMean, originalStdDev;
    cv::Scalar correctedMean, correctedStdDev;
    cv::meanStdDev(originalImage, originalMean, originalStdDev);
    cv::meanStdDev(correctedImage, correctedMean, correctedStdDev);
    double maxValue = 255.0;
    peakSignalToNoiseRatio = 10 * log10(maxValue * maxValue / (cv::sum(difference.mul(difference))[0] / (double)(originalImage.total())));
    std::cout << "Peak Signal-to-Noise Ratio (PSNR): " << peakSignalToNoiseRatio << " dB" << std::endl;

    // Structural Similarity Index (SSIM)
    std::vector<cv::Mat> originalChannels, correctedChannels;
    cv::split(originalImage, originalChannels);
    cv::split(correctedImage, correctedChannels);
    cv::Mat originalHist, correctedHist;
    for (int i = 0; i < 3; i++) {
        cv::Mat originalHistChannel, correctedHistChannel;
        cv::calcHist(&originalChannels[i], 1, 0, cv::Mat(), originalHistChannel, 1, &originalChannels[i].cols, 0);
        cv::calcHist(&correctedChannels[i], 1, 0, cv::Mat(), correctedHistChannel, 1, &correctedChannels[i].cols, 0);
        originalHistChannel.convertTo(originalHistChannel, CV_32F);
        correctedHistChannel.convertTo(correctedHistChannel, CV_32F);
        originalHist.push_back(originalHistChannel);
        correctedHist.push_back(correctedHistChannel);
    }
    double ssimR = 1.0 - cv::compareHist(originalHist.row(2), correctedHist.row(2), cv::HISTCMP_BHATTACHARYYA);
    double ssimG = 1.0 - cv::compareHist(originalHist.row(1), correctedHist.row(1), cv::HISTCMP_BHATTACHARYYA);
    double ssimB = 1.0 - cv::compareHist(originalHist.row(0), correctedHist.row(0), cv::HISTCMP_BHATTACHARYYA);
    structuralSimilarity = (ssimR + ssimG + ssimB) / 3.0;
    std::cout << "Structural Similarity Index (SSIM): " << structuralSimilarity << std::endl;

    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
    return 0;
}