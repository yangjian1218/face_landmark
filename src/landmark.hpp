#include "anchor_generator.h"
#include "tools.h"
#include<net.h>
#include<opencv2/opencv.hpp>

//using namespace cv;

float pixel_mean2[3] = { 0, 0, 0 };
float pixel_std2[3] = { 1, 1, 1 };


class Landmark
{
public:
	Landmark(string param_path, string model_path) {
		m_param_path = param_path;
		m_model_path = model_path;
	}
	int	Init() {
		int param_load = net.load_param(m_param_path.data());
		int model_load = net.load_model(m_model_path.data());
		if (param_load == 0 && model_load == 0) {
			cout << "人脸关键点检测模型加载成功" << endl;

			return 0;
		}
		else
		{
			cout << "人脸点关键点检测模型加载失败" << endl;
			return -1;
		}
	}
	vector<cv::Point> get_landmark(cv::Mat img, cv::Rect box) {
		ncnn::Extractor ex = net.create_extractor();
		ex.set_light_mode(true);
		ex.set_num_threads(4);
		float x1 = box.x, y1 = box.y;
		float x2 = box.width, y2 = box.height;
		float w = x2 - x1, h = y2 - y1;
		cv::Point center((x1 + x2) / 2, (y1 + y2) / 2);  //得到人脸框的中心
		//cout << "center=" << center;
		float rotate = 0;
		float _scale = 192 * 2 / 3.0 / max(w, h);
		pair<cv::Mat, cv::Mat> rimg_M = pic_transform(img, center, _scale, 192, rotate);
		cv::Mat rimg = rimg_M.first;   //仿射变换后192x192的图
		cv::Mat M = rimg_M.second;    //仿射变换矩阵
		cv::cvtColor(rimg, rimg, cv::COLOR_BGR2RGB);   //可有可无
		cv::imshow("rimg", rimg);
		cv::waitKey(0);
		ncnn::Mat indet = ncnn::Mat::from_pixels(rimg.clone().data, ncnn::Mat::PIXEL_BGR, rimg.cols, rimg.rows);  //rimg已经是192x192x3
		indet.substract_mean_normalize(pixel_mean2, pixel_std2);   //数据标准化,可有可无
		ex.input("data", indet);

		ncnn::Mat outdet;
		ex.extract("fc1", outdet);
		ncnnMat_print(outdet);
		cv::Mat IM;
		cv::invertAffineTransform(M, IM);  //得到仿射变换逆矩阵
		vector<cv::Point> pred = get_point_2d(outdet);
		vector<cv::Point> outpoint = trans_point(pred, IM);
		return outpoint;
	}
	~Landmark() {
		cout << "landmark 释放" << endl;
	};
protected:
	
private:
	ncnn::Net net;
	int moele_load;
	string m_param_path;
	string m_model_path;
};