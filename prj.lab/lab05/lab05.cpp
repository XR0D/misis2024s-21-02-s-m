#include <opencv2/opencv.hpp>
#include <cmath>

cv::Mat testImage(int width, int height, int radius) {
          cv::Mat background(height*2, width*3, CV_8UC1, cv::Scalar(0));
          
          cv::Mat image(height, width, CV_8UC1, cv::Scalar(0));
          cv::circle(image, cv::Point(width / 2, height / 2), radius, cv::Scalar(127), -1);
          image.copyTo(background(cv::Rect(0, 0, width, height)));

          image = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
          cv::circle(image, cv::Point(width / 2, height / 2), radius, cv::Scalar(255), -1);
          image.copyTo(background(cv::Rect(width, 0, width, height)));

          image = cv::Mat(height, width, CV_8UC1, cv::Scalar(127));
          cv::circle(image, cv::Point(width / 2, height / 2), radius, cv::Scalar(0), -1);
          image.copyTo(background(cv::Rect(2 * width, 0 * height, width, height)));

          image = cv::Mat(height, width, CV_8UC1, cv::Scalar(127));
          cv::circle(image, cv::Point(width / 2, height / 2), radius, cv::Scalar(255), -1);
          image.copyTo(background(cv::Rect(0 * width, height, width, height)));

          image = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));
          cv::circle(image, cv::Point(width / 2, height / 2), radius, cv::Scalar(0), -1);
          image.copyTo(background(cv::Rect(1 * width, height, width, height)));

          image = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));
          cv::circle(image, cv::Point(width / 2, height / 2), radius, cv::Scalar(127), -1);
          image.copyTo(background(cv::Rect(2 * width, height, width, height)));

          return background;
}

int main() {

          cv::Mat image;

          image = testImage(99, 99, 30);

          cv::imshow("img", image);

          //начально ядро
          cv::Mat kernel = (cv::Mat_<float>(2, 2) << -1, 1, -1, 1);
          cv::Mat filteredImage;
          cv::filter2D(image, filteredImage, CV_32F, kernel);

          cv::Mat filteredImageFinal  = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);

          for (int i = 0; i < filteredImage.cols; i++) {
                    for (int j = 0; j < filteredImage.rows; j++) {
                              filteredImageFinal.at<uchar>(j, i) = static_cast<uchar>(127.5 + 0.5 * filteredImage.at<float>(j, i)); 
                    }
          }

          cv::imshow("img_filter", filteredImageFinal);

          //транспонированное ядро
          cv::Mat kernelTran = (cv::Mat_<float>(2, 2) << 1, 1, -1, -1);
          cv::Mat filteredImageTran;
          cv::filter2D(image, filteredImageTran, CV_32F, kernelTran);

          
          cv::Mat filteredImageTranFinal  = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
          for (int i = 0; i < filteredImage.cols; i++) {
                    for (int j = 0; j < filteredImage.rows; j++) {
                              filteredImageTranFinal.at<uchar>(j, i) = static_cast<uchar>(127.5 + 0.5 * filteredImageTran.at<float>(j, i)); 
                    }
          }

          cv::imshow("img_filter_tran", filteredImageTranFinal);

          //производная по двум направлениям
          cv::Mat filteredImageJoint  = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
          for (int i = 0; i < filteredImage.cols; i++) {
                    for (int j = 0; j < filteredImage.rows; j++) {
                              filteredImageJoint.at<uchar>(j, i) = static_cast<uchar>(sqrt(filteredImageFinal.at<uchar>(j, i) * filteredImageFinal.at<uchar>(j, i) + filteredImageTranFinal.at<uchar>(j, i) * filteredImageTranFinal.at<uchar>(j, i))); 
                    }
          }

          cv::imshow("img_filter_joint", filteredImageJoint);

          //соединение в 3 канальное изображение
          cv::Mat image3ChannelJoint  = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
          for (int i = 0; i < filteredImage.cols; i++) {
                    for (int j = 0; j < filteredImage.rows; j++) {
                              image3ChannelJoint.at<cv::Vec3b>(j, i)[0] = filteredImageFinal.at<uchar>(j, i); 
                              image3ChannelJoint.at<cv::Vec3b>(j, i)[1] = filteredImageTranFinal.at<uchar>(j, i); 
                              image3ChannelJoint.at<cv::Vec3b>(j, i)[2] = filteredImageJoint.at<uchar>(j, i); 

                    }
          }

          cv::imshow("final", image3ChannelJoint);

          cv::imwrite("../prj.lab/lab05/results/start.jpg", image);
          cv::imwrite("../prj.lab/lab05/results/end.jpg", image3ChannelJoint);

          cv::waitKey(0);
          return 0;
}