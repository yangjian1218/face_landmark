#include "tools.h"
#include<opencv2/opencv.hpp>
using namespace cv;
#define M_PI 3.1415926      //规定π的值.

void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes) {
	filterOutBoxes.clear();
	if (boxes.size() == 0)
		return;
	std::vector<size_t> idx(boxes.size());

	for (unsigned i = 0; i < idx.size(); i++)
	{
		idx[i] = i;
	}

	//descending sort
	sort(boxes.begin(), boxes.end(), std::greater<Anchor>());

	while (idx.size() > 0)
	{
		int good_idx = idx[0];
		filterOutBoxes.push_back(boxes[good_idx]);

		std::vector<size_t> tmp = idx;
		idx.clear();
		for (unsigned i = 1; i < tmp.size(); i++)
		{
			int tmp_i = tmp[i];
			float inter_x1 = std::max(boxes[good_idx][0], boxes[tmp_i][0]);
			float inter_y1 = std::max(boxes[good_idx][1], boxes[tmp_i][1]);
			float inter_x2 = std::min(boxes[good_idx][2], boxes[tmp_i][2]);
			float inter_y2 = std::min(boxes[good_idx][3], boxes[tmp_i][3]);

			float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
			float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

			float inter_area = w * h;
			float area_1 = (boxes[good_idx][2] - boxes[good_idx][0] + 1) * (boxes[good_idx][3] - boxes[good_idx][1] + 1);
			float area_2 = (boxes[tmp_i][2] - boxes[tmp_i][0] + 1) * (boxes[tmp_i][3] - boxes[tmp_i][1] + 1);
			float o = inter_area / (area_1 + area_2 - inter_area);
			if (o <= threshold)
				idx.push_back(tmp_i);
		}
	}
}

pair<cv::Mat, float> square_crop(cv::Mat im, int S) {
	// 把图短边进行填充为正方形,并进行缩放.保证缩放后的图横纵比不变
	//im: 传入原始图片
	//S: 要缩放到的尺寸
	//return: <修正后的图片,缩放比例>
	int height, width;
	float scale;  //缩放比例
	if (im.rows > im.cols) {
		height = S;
		width = int(float(im.cols) / im.rows * S);
		scale = float(S) / im.rows;
	}
	else
	{
		width = S;
		height = int(float(im.rows) / im.cols * S);
		scale = float(S) / im.cols;
	}
	cv::Mat resized_im;
	cv::resize(im, resized_im, Size(width, height), 0, 0, INTER_AREA);
	cv::Mat det_im = cv::Mat::zeros(Size(S, S), im.type());
	cv::Rect roi(0, 0, resized_im.cols, resized_im.rows);
	resized_im.clone().copyTo(det_im(roi));
	pair<cv::Mat, float> pp(det_im, scale);
	return pp;
}

void ncnnMat_print(const ncnn::Mat& m)
//打印mat
{
	cout << "打印ncnnMat:" << endl;
	for (int q = 0; q < m.c; q++)
	{
		const float* ptr = m.channel(q); //每个通道的首地址
		for (int y = 0; y < m.h; y++)
		{
			for (int x = 0; x < m.w; x++)
			{
				printf("%f ", ptr[x]);
			}
			ptr += m.w;
			printf("\n");
		}
	}
}

std::vector<Anchor> choose_one(std::vector<Anchor> faces) {
	// Summary: 选择面积最大的人脸框
	// Param rests: 所有人脸
	//return: 只返回最大人脸
	std::vector<Anchor> res;
	float max = 0.0;
	int max_index;
	for (int i = 0;i< faces.size(); i++) {
		Rect box = faces[i].finalbox;
		float area = (box.width - box.x) * (box.height - box.y);
		if (area > max) {
			max = area;
			max_index = i;
		}
	}
	res.push_back(faces[max_index]);
	return res;
}

pair <cv::Rect, vector<Point2f>> recover_point(Anchor face, float det_scale) {
	//把
	Rect box = face.finalbox;   //人脸框x1 y1 x2 y2 score
	vector<Point2f> pts = face.pts;  //5个 x y
	float x1 = box.x / det_scale;
	float y1 = box.y / det_scale;
	float x2 = box.width / det_scale;
	float y2 = box.height / det_scale;
	cv::Rect new_box(x1, y1, x2, y2);
	vector<Point2f> new_landmark;
	for (int i = 0; i < pts.size(); i++) {
		cv::Point2f point = pts[i];
		Point2f p(point.x / det_scale, point.y / det_scale);
		new_landmark.push_back(p);
	}
	return std::make_pair(new_box, new_landmark);
}


pair<cv::Mat, cv::Mat> pic_transform(cv::Mat img, cv::Point center, float scale, int image_size, float rotate) {
	float rot = (rotate * M_PI) / 180.0;
	float cx = center.x * scale;
	float cy = center.y * scale;
	cv::Mat t1 = (cv::Mat_<float>(3, 3) << scale, -0., 0., 0., scale, 0., 0., 0., 1.0);
	cv::Mat t2 = (cv::Mat_<float>(3, 3) << 0, -0., -1 * cx, 0., 0., -1 * cy, 0., 0., 1.0);
	cv::Mat t3 = (cv::Mat_<float>(3, 3) << 0, -0., 0., 0, 0, 0., 0., 0., 1.0);
	cv::Mat t4 = (cv::Mat_<float>(3, 3) << 0, -0., float(image_size) / 2, 0, 0, float(image_size) / 2, 0., 0., 1.0);
	cv::Mat t = t1 + t2 + t3 + t4;
	cv::Mat M = t.rowRange(0, 2);    //只要前2行
	//cout << "M=" << M << endl;
	cv::Mat cropped;
	cv::warpAffine(img, cropped, M, cv::Size(image_size, image_size));
	//cv::imshow("cropped", cropped);
	//cv::waitKey(0);
	return std::make_pair(cropped, M);

}


std::vector<cv::Point> get_point_2d(ncnn::Mat out) {
	// Summary:把输出存入容器中,元素为点
	//Parameters:
	//  out: 特征点网络2d06det输出,相对于仿射变化后的rimg下的相对坐标(-1,1)
	//Return: rimg图下的特征点放入容器中

	vector<cv::Point> outpoint;   //特征点容器,元素为点
	int xx, yy;
	const float* ptr = out.channel(0);
	for (int y = 0; y < out.h; y++)
	{
		for (int x = 0; x < out.w; x++)
		{
			if (x % 2 == 0) {
				xx = int((ptr[x] + 1) * (192.0 / 2));
			}
			else {
				yy = int((ptr[x] + 1) * (192.0 / 2));
				outpoint.push_back(Point(xx, yy));
			}
		}
		ptr += out.w;
	}
	return outpoint;
}

std::vector<cv::Point> trans_point(std::vector<Point> points, cv::Mat IM) {
	//把输出点通过仿射变换的矩阵逆矩阵,得到原图的坐标点
	//points: 仿射图rimg下特征点的
	//IM: 仿射变换矩阵的逆矩阵 
	//return: 原图下的特征点坐标点
	vector<cv::Point> new_pts;
	for (int i = 0; i < points.size(); i++) {
		cv::Point pt = points[i];
		cv::Mat new_pt = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1.0);
		new_pt = IM * new_pt;
		cv::Point p(new_pt.at<float>(0, 0), new_pt.at<float>(1, 0));
		new_pts.push_back(p);
	}
	return new_pts;
}


void draw_points(cv::Mat& img, vector<vector<Point>> outpoints) {
	//绘制特征点
	//img: 要绘制的图板
	//outpoints:所有人脸的特征点
	for (int i = 0; i < outpoints.size(); i++) {
		vector<Point> outpoint = outpoints[i];
		for (int j = 0; j < outpoint.size(); j++) {
			cv::circle(img, outpoint[j], 2, Scalar(0, 0, 255), -1, 8, 0);
		}
	}
}