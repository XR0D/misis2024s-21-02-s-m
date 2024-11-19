#include <iostream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

std::vector<cv::Vec3f> groundTruths;

void evaluateDetections(const std::vector<cv::Vec3f>& detections, const std::vector<cv::Vec3f>& groundTruths, double iouThreshold) {
	int TP = 0, FP = 0, FN = 0;

	for (const auto& truth : groundTruths) {
		bool detected = false;
		cv::Point truthCenter(cvRound(truth[0]), cvRound(truth[1]));
		int truthRadius = cvRound(truth[2]);

		for (const auto& detection : detections) {
			cv::Point detectCenter(cvRound(detection[0]), cvRound(detection[1]));
			int detectRadius = cvRound(detection[2]);

			double distance = cv::norm(truthCenter - detectCenter);
			double radiusSum = truthRadius + detectRadius;
			double iou = (distance <= radiusSum) ? 1.0 : 0.0; // Упростим вычисление IoU

			if (iou >= iouThreshold) {
				TP++;
				detected = true;
				break;
			}
		}
		if (!detected) FN++;
	}

	FP = detections.size() - TP;

	std::cout << "TP: " << TP << std::endl;
	std::cout << "FP: " << FP << std::endl;
	std::cout << "FN: " << FN << std::endl;
}

cv::Mat niblackBinary(const cv::Mat& image, const int radius, const int k, const int d) {
	cv::Mat binary;

	constexpr int max = static_cast<int>(std::numeric_limits<uchar>::max());
	cv::Mat kernel{ cv::Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_32FC1) };
	cv::circle(kernel, cv::Point{ radius, radius }, radius, cv::Scalar{ 1.0f }, -1);
	kernel /= cv::sum(kernel)[0];
	cv::Mat tmp{ image.clone() };
	tmp.convertTo(tmp, CV_32FC1, 1.0 / max);
	cv::Mat mean{ tmp.clone() };
	cv::filter2D(tmp, mean, -1, kernel);
	cv::multiply(tmp, tmp, tmp);
	cv::Mat meanSq{ tmp.clone() };
	cv::filter2D(tmp, meanSq, -1, kernel);
	cv::multiply(mean, mean, tmp);
	cv::sqrt(meanSq - tmp, tmp);
	image.convertTo(binary, CV_32FC1, 1.0 / max);
	binary = binary > (mean + k * tmp + static_cast<double>(d) / max);
	binary.convertTo(binary, CV_8UC1, max);

	return binary;
}

cv::Mat bernsenBinary(const cv::Mat& image, const int radius, const int cmin) {
	cv::Mat binary;

	constexpr int max = static_cast<int>(std::numeric_limits<uchar>::max());
	cv::Mat kernel{ cv::Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_8UC1) };
	cv::circle(kernel, cv::Point{ radius, radius }, radius, cv::Scalar{ max }, -1);
	cv::Mat iMin{ cv::Mat::zeros(image.rows, image.cols, CV_8UC1) };
	cv::erode(image, iMin, kernel);
	cv::Mat iMax{ cv::Mat::zeros(image.rows, image.cols, CV_8UC1) };
	cv::dilate(image, iMax, kernel);
	cv::Mat diff{ iMax - iMin };
	cv::threshold(diff, diff, cmin, max, cv::THRESH_BINARY_INV);
	binary = image > (diff | ((iMin + iMax) / 2));

	return binary;
}

cv::Mat add_gaussian_noise(const cv::Mat& image, const double std_dev) {
	cv::Mat noisy_image;
	cv::Mat noise(image.size(), CV_32FC1);
	cv::randn(noise, 0, std_dev);

	image.convertTo(noisy_image, CV_32FC1);
	noisy_image += noise;
	noisy_image.convertTo(noisy_image, CV_8UC1);

	return noisy_image;
}

cv::Mat generateImage(const int& countCircles, int minRadius, int maxRadius, int minContrast, int maxContrast, int blur, int noise) {
	int side_length = 4 * maxRadius * (countCircles - 1);
	cv::Mat image(side_length, side_length, CV_8UC1, cv::Scalar(0));

	int radius = minRadius;
	int contrast = minContrast;
	for (int i = side_length / 10; i <= (side_length - side_length / 10); i += (side_length - side_length / 5) / (countCircles - 1)) {
		for (int j = side_length / 10; j <= (side_length - side_length / 10); j += (side_length - side_length / 5) / (countCircles - 1)){
			cv::Point center(i, j);
			cv::circle(image, center, radius, cv::Scalar(contrast), -1);
			groundTruths.push_back(cv::Vec3f(i, j, radius));
			contrast += (maxContrast - minContrast) / countCircles;
		}
		contrast = minContrast;
		radius += (maxRadius - minRadius) / countCircles;
	}
	
	cv::GaussianBlur(image, image, cv::Size(blur * 2 + 1, blur * 2 + 1), 0);

	cv::Mat result;
	result = add_gaussian_noise(image, noise);

	return result;
}

void onTrackbar(int, void*) {}

int main(int argc, char* argv[]) {
	//Default parametrs
	int countCircles = 7;
	int minRadius = 5;
	int maxRadius = 20;
	int minContrast = 40;
	int maxContrast = 180;
	int blur = 3;
	int noise = 7;
	std::string method = "detection";

	if (argc >= 7) {
		countCircles = std::stoi(argv[1]);
		minRadius = std::stoi(argv[2]);
		maxRadius = std::stoi(argv[3]);
		minContrast = std::stoi(argv[4]);
		maxContrast = std::stoi(argv[5]);
		blur = std::stoi(argv[6]);
		noise = std::stoi(argv[7]);
		method = argv[8];
	}

	cv::Mat image = generateImage(countCircles, minRadius, maxRadius, minContrast, maxContrast, blur, noise);
	//cv::imwrite("example3.png", image);
	cv::Mat binaryImage;

	if (method == "bernsen") {
		int radius = 1;
		int cmin = 0;

		cv::namedWindow("Binary Image");
		cv::createTrackbar("Radius", "Binary Image", &radius, 30, onTrackbar);
		cv::setTrackbarMin("Radius", "Binary Image", radius);
		cv::createTrackbar("CMin", "Binary Image", &cmin, 255, onTrackbar);
		cv::setTrackbarMin("CMin", "Binary Image", cmin);
		onTrackbar(0, 0);

		while (true) {
			binaryImage = bernsenBinary(image, radius, cmin);
			imshow("Binary Image", binaryImage);

			char key = cv::waitKey(10);
			if (key == 27) // ESC key to exit
				break;
		}
	
	}
	else if (method == "niblack") {
		int radius = 1;
		int k = 0;
		int d = 0;

		cv::namedWindow("Binary Image");
		cv::createTrackbar("Radius", "Binary Image", &radius, 30, onTrackbar);
		cv::setTrackbarMin("Radius", "Binary Image", radius);
		cv::createTrackbar("k Value", "Binary Image", &k, 255, onTrackbar);
		cv::setTrackbarMin("k Value", "Binary Image", k);
		cv::createTrackbar("d Value", "Binary Image", &d, 255, onTrackbar);
		cv::setTrackbarMin("d Value", "Binary Image", d);
		onTrackbar(0, 0);

		while (true) {
			k = static_cast<double>(cv::getTrackbarPos("k Value", "Binary Image")) / 10.0;
			binaryImage = niblackBinary(image, radius, k, d);
			imshow("Binary Image", binaryImage);

			char key = cv::waitKey(10);
			if (key == 27) // ESC key to exit
				break;
		}
	}
	else if (method == "detection") {
		//binaryImage = bernsenBinary(image, 10, 30);
		binaryImage = niblackBinary(image, 20, 0.90, 23);
		
		// Search for circles in the image
		std::vector<cv::Vec3f> circles;
		cv::HoughCircles(binaryImage, circles, cv::HOUGH_GRADIENT, 1,
			(binaryImage.rows - binaryImage.rows / 5) / (countCircles - 1),
			100, 10,       
			minRadius, maxRadius);  

		// Drawing detected circles
		cv::Mat detected_circles_image;
		cv::cvtColor(image, detected_circles_image, cv::COLOR_GRAY2BGR);
		for (size_t i = 0; i < circles.size(); i++)
		{
			cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);

			cv::circle(detected_circles_image, center, 3, cv::Scalar(0, 255, 0), -1, 1);

			cv::circle(detected_circles_image, center, radius, cv::Scalar(0, 0, 255), 2, 1);
		}

		double iouThreshold = 0.5; // Порог IoU для считать детекцию TP
		evaluateDetections(circles, groundTruths, iouThreshold);

		cv::imshow("Original Image", image);
		cv::imshow("Binary Image", binaryImage);
		cv::imshow("Detected Circles", detected_circles_image);
		cv::waitKey(0);
	}
}