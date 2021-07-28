#include "anchor_generator.h"
#include "opencv2/opencv.hpp"
#include "config.h"
#include "tools.h"
//#include<cmath>

using namespace std;

int main(int args, char** argv) {
	extern float pixel_mean[3];   //均值,在config规定了,这里无法改变值
	extern float pixel_std[3];    //方差,在config规定了,这里无法改变值
	//人脸检测模型
	std::string param_path = "D:/AI/Face/Face_detect/RetinaFace-Cpp/Demo/ncnn/models/retina.param";
	std::string model_path = "D:/AI/Face/Face_detect/RetinaFace-Cpp/Demo/ncnn/models/retina.bin";
	// 特征点检测模型
	string det_param_path = "D:/AI/Face/Face_detect/RetinaFace-Cpp/Demo/ncnn/models/v3.param";
	string det_model_path = "D:/AI/Face/Face_detect/RetinaFace-Cpp/Demo/ncnn/models/v3.bin";

	//加载人脸加测模型
	ncnn::Net _net;
	int ret_param = _net.load_param(param_path.data());
	int ret_model = _net.load_model(model_path.data());
	if (ret_param == 0 && ret_model == 0) {
		cout << "人脸检测模型Retinaface加载成功" << endl;
	}
	else
	{
		cout << "人脸检测模型Retinaface加载失败" << endl;
		return 0;
	}
	//加载特征点检测模型
	ncnn::Net det_net;
	// 2d106det
	int det_param = det_net.load_param("D:/AI/Face/Face_detect/RetinaFace-Cpp/Demo/ncnn/models/2d106det.param");
	int det_model = det_net.load_model("D:/AI/Face/Face_detect/RetinaFace-Cpp/Demo/ncnn/models/2d106det.bin");
	if (det_param == 0 && det_model == 0) {
		cout << "人脸关键点检测模型2d106det加载成功" << endl;
	}
	else
	{
		cout << "人脸关键点检测模型2d106det加载失败" << endl;
		return 0;
	}
	//cv::Mat img = cv::imread("../../../images/test.jpg");
	cv::Mat img = cv::imread("D:\\AI\\Face\\face_identify\\Insightface\\Facemark\\coordinateReg\\data\\test_f.jpeg");
	//cv::Mat img = cv::imread("C:/Users/jerry/Pictures/pic_test/black3.jpg");
	//cv::Mat img = cv::imread("C:\\Users\\jerry\\Pictures\\oumei1.jpg");
	if (!img.data) {
		printf("load error");
		return 0;
	}
	cv::Mat img_c = img.clone();
	int detim_size = 640; //原始照片要缩放后的尺寸
	pair<cv::Mat, float> detIm_scala = square_crop(img, detim_size); // 进过square_crop 保证图片保持衡纵比
	cv::Mat det_im = detIm_scala.first;   //缩放后的图 640x640
	float det_scale = detIm_scala.second;   //缩放比例

	ncnn::Mat input = ncnn::Mat::from_pixels(det_im.data, ncnn::Mat::PIXEL_BGR2RGB, det_im.cols, det_im.rows);

	input.substract_mean_normalize(pixel_mean, pixel_std);
	ncnn::Extractor _extractor = _net.create_extractor();
	_extractor.set_light_mode(true);
	_extractor.set_num_threads(4);
	_extractor.input("data", input);   //可以从.param查看网络结构,查看输入名称

	std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());  //_feat_stride_fpn = {32, 16, 8}  .size=3
	for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
		int stride = _feat_stride_fpn[i];   //32, 16,8
		ac[i].Init(stride, anchor_cfg[stride], false);
	}

	std::vector<Anchor> proposals;
	proposals.clear();

	for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
		ncnn::Mat cls;
		ncnn::Mat reg;
		ncnn::Mat pts;

		// get blob output
		char clsname[100]; sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);  //sprintf打印到字符串中,即赋值
		char regname[100]; sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
		char ptsname[100]; sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
		_extractor.extract(clsname, cls);  //得到分类
		_extractor.extract(regname, reg);  //得到目标框
		_extractor.extract(ptsname, pts);  //得到特征点

		//printf("cls %d %d %d\n", cls.c, cls.h, cls.w);  //4 10 10
		//printf("reg %d %d %d\n", reg.c, reg.h, reg.w);  //8 10 10
		//printf("pts %d %d %d\n", pts.c, pts.h, pts.w);

		ac[i].FilterAnchor(cls, reg, pts, proposals);  //筛选

		//printf("stride %d, res size %lld\n", _feat_stride_fpn[i], proposals.size());
		//for (int r = 0; r < proposals.size(); ++r) {
		//	proposals[r].print();
		//}
	}

	// nms
	std::vector<Anchor> result;   //
	nms_cpu(proposals, nms_threshold, result);  //得到最终的人脸,  原函数result传入的是地址,
	printf("final result %lld\n", result.size());  //人脸个数
	//result = choose_one(result);

	vector<vector<Point>> all_points;
	for (int i = 0; i < result.size(); i++)  //遍历人脸
	{
		result[i].print();
		Anchor face = result[i];   //这里的人脸还是在det_im(640x640)图中的坐标
		pair<cv::Rect, vector<Point2f>> new_face = recover_point(face,det_scale);  //得到原始img下的人脸框信息
		Rect new_box = new_face.first;   //img下人脸框
		vector<Point2f> new_pts = new_face.second;  //img下的5个特征点
		float x1 = new_box.x, y1 = new_box.y;
		float x2 = new_box.width, y2 = new_box.height;
		//cv::rectangle(img_c, cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2), cv::Scalar(0, 255, 255), 2, 8, 0);
		//for (int j = 0; j < new_pts.size(); ++j) {
		//	cv::circle(img_c, cv::Point((int)new_pts[j].x, (int)new_pts[j].y), 1, cv::Scalar(225, 0, 225), 2, 8);
		//}
		float w = x2 - x1, h = y2 - y1;
		cv::Point center((x1+x2)/2,(y1+y2)/2);  //得到人脸框的中心
		//cout << "center=" << center;
		float rotate = 0;
		float _scale = 192 * 2 / 3.0 / max(w, h);
		pair<cv::Mat,cv::Mat> rimg_M= pic_transform(img, center, _scale, 192, rotate);
		cv::Mat rimg = rimg_M.first;   //仿射变换后192x192的图
		cv::Mat M = rimg_M.second;    //仿射变换矩阵
		cv::cvtColor(rimg, rimg, cv::COLOR_BGR2RGB);   //可有可无
		ncnn::Mat indet = ncnn::Mat::from_pixels(rimg.data, ncnn::Mat::PIXEL_BGR, rimg.cols, rimg.rows);  //rimg已经是192x192x3
		ncnn::Extractor exdet = det_net.create_extractor();
		exdet.set_light_mode(true);
		exdet.set_num_threads(4);
		//indet.substract_mean_normalize(pixel_mean, pixel_std);   //数据标准化,可有可无
		exdet.input("data", indet);
		ncnn::Mat outdet;
		exdet.extract("fc1", outdet);
		cv::Mat IM;
		cv::invertAffineTransform(M, IM);  //得到仿射变换逆矩阵
		vector<cv::Point> pred = get_point_2d(outdet);   
		vector<cv::Point> outpoint = trans_point(pred, IM);
		all_points.push_back(outpoint);
	}

	draw_points(img_c, all_points);  //绘制 关键点

	cv::namedWindow("img", 3);
	cv::imshow("img", img_c);
	cv::waitKey(0);
	return 0;

}

