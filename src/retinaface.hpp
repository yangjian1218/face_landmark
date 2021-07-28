#include "anchor_generator.h"
#include "tools.h"
#include<net.h>
#include<opencv2/opencv.hpp>

using namespace cv;

float pixel_mean1[3] = { 0, 0, 0 };
float pixel_std1[3] = { 1, 1, 1 };

class Retinaface
{
public:
	Retinaface(string param_path, string model_path) {
		m_param_path = param_path;
		m_model_path = model_path;
	}
	int Init() {
		int param_load = retinet.load_param(m_param_path.data());
		int model_load = retinet.load_model(m_model_path.data());
		if (param_load == 0 && model_load == 0) {
			cout << "�������ģ�ͼ��سɹ�" << endl;
			return 0;
		}
		else
		{
			cout << "��������ģ�ͼ���ʧ��" << endl;
			return -1;
		}
	}
	std::vector < pair<cv::Rect, vector<Point2f>>> detect(cv::Mat img, int detim_size = 640) {

		std::pair<cv::Mat, float> detIm_scala = square_crop(img, detim_size); // ����square_crop ��֤ͼƬ���ֺ��ݱ�
		cv::Mat det_im = detIm_scala.first;   //���ź��ͼ 640x640
		float det_scale = detIm_scala.second;   //���ű���

		ncnn::Mat input = ncnn::Mat::from_pixels(det_im.data, ncnn::Mat::PIXEL_BGR2RGB, det_im.cols, det_im.rows);
		input.substract_mean_normalize(pixel_mean1, pixel_std1);
		ncnn::Extractor ex = retinet.create_extractor();
		ex.set_light_mode(true);
		ex.set_num_threads(4);
		std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());  //_feat_stride_fpn = {32, 16, 8}  .size=3
		for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
			int stride = _feat_stride_fpn[i];   //32, 16,8
			ac[i].Init(stride, anchor_cfg[stride], false);
		}

		std::vector<Anchor> proposals;
		proposals.clear();

		for (int j = 0; j < _feat_stride_fpn.size(); j++) {
			
			ncnn::Mat cls;
			ncnn::Mat reg;
			ncnn::Mat pts;

			// get blob output
			char clsname[100]; sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[j]);  //sprintf��ӡ���ַ�����,����ֵ
			char regname[100]; sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[j]);
			char ptsname[100]; sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[j]);
			ex.extract(clsname, cls);  //�õ�����
			ex.extract(regname, reg);  //�õ�Ŀ���
			ex.extract(ptsname, pts);  //�õ�������
			ac[j].FilterAnchor(cls, reg, pts, proposals);  //ɸѡ

			printf("stride %d, res size %lld\n", _feat_stride_fpn[j], proposals.size());
			for (int r = 0; r < proposals.size(); ++r) {
				proposals[r].print();
			}
		}

		// nms
		std::vector<Anchor> result;   //
		nms_cpu(proposals, nms_threshold, result);  //�õ����յ�����,  ԭ����result������ǵ�ַ,
		printf("final result %lld\n", result.size());  //��������
		result = choose_one(result); //��ֻ��det_im�µ�
		std::vector<pair<cv::Rect, vector<Point2f>>> img_faces;
		for (int i = 0; i < result.size(); i++)  //��������
		{
			result[i].print();
			Anchor face = result[i];   //���������������det_im(640x640)ͼ�е�����
			pair<cv::Rect, vector<Point2f>> new_face = recover_point(face, det_scale);  //�õ�ԭʼimg�µ���������Ϣ

			img_faces.push_back(new_face);
		}
		return img_faces;
	}
	~Retinaface() {
		cout << "�����ͷ�" << endl;
	}
protected:

private:
	ncnn::Net retinet;
	int param_load;
	int moele_load;
	string m_param_path;
	string m_model_path;

};