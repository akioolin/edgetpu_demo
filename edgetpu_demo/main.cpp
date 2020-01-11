/*
 Creator : Akio Lin
 Date    : 2020-01-08
 Email   : akioolin@gmail.com
*/

#include <opencv2/opencv.hpp>

#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/classification/engine.h"
#include "src/cpp/detection/engine.h"
#include "src/cpp/utils.h"
#include "src/cpp/bbox_utils.h"

#define MODEL_FILENAME	"./etc/model/mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite"
#define LABEL_FILENAME	"./etc/label/mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.label"

using namespace coral;

static void readLabel(const char* filename, std::vector<std::string> &labels);

static void readLabel(const char* filename, std::vector<std::string> &labels)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		std::cout << "failed to read " << filename << "\r\n";
		return;
	}

	std::string str;
	while(getline(ifs, str)) {
		labels.push_back(str);
	}
}

int main(int argc, char *argv[]) {
	int top_k = 1;
	int dim_size = 0;
	int image_width = 0;
	int image_height = 0;
	float score_threshold = 0.35f;
	float iou_threshold = 0.75f;
	cv::Mat video_frame;
	cv::Mat resize_frame;
	std::vector<std::string> labels;
	DetectionEngine *engine;

	// Load model's label file
	readLabel(LABEL_FILENAME, labels);	

	edgetpu::EdgeTpuManager::GetSingleton()->SetVerbosity(10);
	engine = new DetectionEngine(MODEL_FILENAME);
	std::vector<int> input_tensor_shape = engine->get_input_tensor_shape();
	std::vector<int> output_tensor_size = engine->get_all_output_tensors_sizes();

	dim_size = input_tensor_shape.size();
	std::cout << "input tensor dimension:" << dim_size << "\r\n";
	if(dim_size == 4) {
		std::cout <<"input tensor batch:"   << input_tensor_shape[0] << "\r\n";
		std::cout <<"input tensor width:"   << input_tensor_shape[1] << "\r\n";
		std::cout <<"input tensor height:"  << input_tensor_shape[2] << "\r\n";
		std::cout <<"input tensor channel:" << input_tensor_shape[3] << "\r\n";
		image_width  = input_tensor_shape[1];
		image_height = input_tensor_shape[2];
	} else {
		std::cout << "model input tensor dimension wrong, please check model\r\n";
	}

	dim_size = output_tensor_size.size();
	std::cout << "output tensor dimension:" << dim_size << "\r\n";
	if(dim_size == 4) {
		std::cout <<"output tensor batch:"   << output_tensor_size[0] << "\r\n";
		std::cout <<"output tensor width:"   << output_tensor_size[1] << "\r\n";
		std::cout <<"output tensor height:"  << output_tensor_size[2] << "\r\n";
		std::cout <<"output tensor channel:" << output_tensor_size[3] << "\r\n";
	} else {
		std::cout << "model output tensor dimension wrong, please check model\r\n";
	}

	cv::VideoCapture uvc_camera(1);

	if(!uvc_camera.isOpened()){
		std::cout << "Open Camera Failed!!\r\n";
		return -1;
	} else {
		// Camera Setting
		uvc_camera.set(cv::CAP_PROP_FRAME_WIDTH,  640);
		uvc_camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
		uvc_camera.set(cv::CAP_PROP_FPS, 30.0f);
		std::cout << "Open Camera Done.\r\n";
		std::cout << "Camera frame width: "  << uvc_camera.get(cv::CAP_PROP_FRAME_WIDTH)  << "\r\n";
		std::cout << "Camera frame height: " << uvc_camera.get(cv::CAP_PROP_FRAME_HEIGHT) << "\r\n";
		std::cout << "Camera frame fps: "    << uvc_camera.get(cv::CAP_PROP_FPS) << "\r\n";
	}

	while(1) {
		if(!uvc_camera.read(video_frame)) {
			std::cout << "video frame captured failed!!\r\n";
			break;
		}
		// video frame color format conversion, resize.
		cv::cvtColor(video_frame, resize_frame, cv::COLOR_BGR2RGB);
		cv::resize(resize_frame, resize_frame, cv::Size(image_width, image_height));
		cv::imshow("UVC Camera raw frame", video_frame);
		cv::imshow("UVC Camera resize frame", resize_frame);
		
		std::vector<uint8_t> input_tensor(resize_frame.data, resize_frame.data + (resize_frame.cols * resize_frame.rows * resize_frame.elemSize()));
		auto candiates = engine->DetectWithInputTensor(input_tensor, score_threshold, top_k);

		dim_size = candiates.size();
		if(dim_size == 1) {
			DetectionCandidate result = candiates[0];
			std::cout << "object=> " << labels[result.label] << ", probability=> " << result.score << "\r\n";
			std::cout << "bottom=> " << result.corners.ymin << "\r\n";
			std::cout << "left=> "   << result.corners.xmin << "\r\n";
			std::cout << "top=> "    << result.corners.ymax << "\r\n";
			std::cout << "right=> "  << result.corners.xmax << "\r\n";
		} else {
			std::cout << "no object detect\r\n";
		}

		cv::waitKey(1);
	}

	delete engine;

	return 0;
}
