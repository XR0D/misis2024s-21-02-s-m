#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


int main() {
  cv::Mat1b m(200, 200, 127);
  cv::imshow("example", m);
  cv::waitKey(0);
}