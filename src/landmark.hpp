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
			cout << "�����ؼ�����ģ�ͼ��سɹ�" << endl;

			return 0;
		}
		else
		{
			cout << "������ؼ�����ģ�ͼ���ʧ��" << endl;
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
		cv::Point center((x1 + x2) / 2, (y1 + y2) / 2);  //�õ������������
		//cout << "center=" << center;
		float rotate = 0;
		float _scale = 192 * 2 / 3.0 / max(w, h);
		pair<cv::Mat, cv::Mat> rimg_M = pic_transform(img, center, _scale, 192, rotate);
		cv::Mat rimg = rimg_M.first;   //����任��192x192��ͼ
		cv::Mat M = rimg_M.second;    //����任����
		cv::cvtColor(rimg, rimg, cv::COLOR_BGR2RGB);   //���п���
		cv::imshow("rimg", rimg);
		cv::waitKey(0);
		ncnn::Mat indet = ncnn::Mat::from_pixels(rimg.clone().data, ncnn::Mat::PIXEL_BGR, rimg.cols, rimg.rows);  //rimg�Ѿ���192x192x3
		indet.substract_mean_normalize(pixel_mean2, pixel_std2);   //���ݱ�׼��,���п���
		ex.input("data", indet);

		ncnn::Mat outdet;
		ex.extract("fc1", outdet);
		ncnnMat_print(outdet);
		cv::Mat IM;
		cv::invertAffineTransform(M, IM);  //�õ�����任�����
		vector<cv::Point> pred = get_point_2d(outdet);
		vector<cv::Point> outpoint = trans_point(pred, IM);
		return outpoint;
	}
	~Landmark() {
		cout << "landmark �ͷ�" << endl;
	};
protected:
	
private:
	ncnn::Net net;
	int moele_load;
	string m_param_path;
	string m_model_path;
};