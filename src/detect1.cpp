#include "anchor_generator.h"
#include "opencv2/opencv.hpp"
#include "config.h"
#include "tools.h"
#include"retinaface.hpp"
#include "landmark.hpp"
//#include<cmath>

using namespace std;

int main(int args, char** argv) {
	//人脸检测模型
	std::string param_path = "D:/AI/Face/Face_detect/RetinaFace-Cpp/Demo/ncnn/models/retina.param";
	std::string model_path = "D:/AI/Face/Face_detect/RetinaFace-Cpp/Demo/ncnn/models/retina.bin";
	// 特征点检测模型
	string det_param_path = "D:/AI/Face/Face_detect/RetinaFace-Cpp/Demo/ncnn/models/2d106det.param";
	string det_model_path = "D:/AI/Face/Face_detect/RetinaFace-Cpp/Demo/ncnn/models/2d106det.bin";

	//加载人脸加测模型
	Retinaface retina(param_path, model_path);
	//加载特征点检测模型
	Landmark det_net(det_param_path, det_model_path);


	int tr = retina.Init();
	int td = det_net.Init();
	//cv::Mat img = cv::imread("../../../images/test.jpg");
	cv::Mat img = cv::imread("D:\\AI\\Face\\face_identify\\Insightface\\Facemark\\coordinateReg\\data\\test_f.jpeg");
	//cv::Mat img = cv::imread("C:/Users/jerry/Pictures/pic_test/black3.jpg");
	//cv::Mat img = cv::imread("C:\\Users\\jerry\\Pictures\\oumei1.jpg");
	if (!img.data) {
		printf("load error");
		return 0;
	}

	cv::Mat img_c = img.clone();
	std::vector < pair<cv::Rect, vector<Point2f>>> faces= retina.detect(img_c, 640);

	vector<vector<Point>> all_points;

	for (int i = 0; i < faces.size(); i++)  //遍历人脸
	{
		pair<cv::Rect, vector<Point2f>> face = faces[i];   //这里的人脸还是在det_im(640x640)图中的坐标
		
		Rect box = face.first;   //img下人脸框
		vector<Point2f> pts = face.second;  //img下的5个特征点

		//cv::rectangle(img_c, cv::Point((int)box.x, (int)box.y), cv::Point((int)box.width, (int)box.height), cv::Scalar(0, 255, 255), 2, 8, 0);
		//for (int j = 0; j < pts.size(); ++j) {
		//	cv::circle(img_c, cv::Point((int)pts[j].x, (int)pts[j].y), 1, cv::Scalar(225, 0, 225), 2, 8);
		//}
		vector<cv::Point> outpoint = det_net.get_landmark(img, box);
		all_points.push_back(outpoint);
	}

	draw_points(img_c, all_points);  //绘制 关键点

	cv::namedWindow("img", 3);
	cv::imshow("img", img_c);
	cv::waitKey(0);
	return 0;
}

