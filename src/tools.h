#ifndef FD_TOOLS
#define FD_TOOLS


#include "anchor_generator.h"
#include<net.h>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes);
pair<Mat, float> square_crop(ncnn::Mat im, int S);
pair<cv::Mat, float> square_crop(cv::Mat im, int S);
void ncnnMat_print(const ncnn::Mat& m);
vector<Anchor> choose_one(std::vector<Anchor> faces);
pair <cv::Rect, vector<Point2f>> recover_point(Anchor face, float det_scale);
pair<cv::Mat, cv::Mat> pic_transform(cv::Mat img, cv::Point center, float scale, int image_size = 192, float rotate = 0.0);
vector<cv::Point> get_point_2d(ncnn::Mat out);
vector<cv::Point> trans_point(std::vector<Point> points, cv::Mat IM);
void draw_points(cv::Mat& img, vector<vector<Point>> outpoints);
#endif // FD_TOOLS

