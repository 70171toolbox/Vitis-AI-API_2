#include <vitis/ai/proto/dpu_model_param.pb.h>

#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/dpu_task.hpp>
using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  string model_name = argv[1];
  string input_file = argv[2];

  auto model = vitis::ai::DpuTask::create(model_name);
  cv::Mat img = cv::imread(input_file);
  auto input_tensor = model->getInputTensor(0u)[0];
  vector<cv::Mat> mats = {img};
  model->setImageRGB(mats);
  model->run(0u);
  auto output_tensor = model->getOutputTensor(0u)[0];
  int8_t* output_ptr = (int8_t*)output_tensor.get_data(0);
  
  cv::Mat res(output_tensor.height, output_tensor.width, CV_8UC3, cv::Scalar(0,0,0));
  for(size_t h = 0; h < output_tensor.height; ++h) {
    for(size_t w = 0; w < output_tensor.width; ++w) {
      int pos_1 = (h * output_tensor.width + w) * output_tensor.channel + 0;
      int pos_2 = (h * output_tensor.width + w) * output_tensor.channel + 1;
      float temp_1 = output_ptr[pos_1] >= output_ptr[pos_2];
      float temp_2 = output_ptr[pos_2] > output_ptr[pos_1];
      res.at<Vec3b>(h, w)[0] = (uint8_t)(temp_1*255);
      res.at<Vec3b>(h, w)[1] = (uint8_t)(temp_2*127.5);
    }
  }
  cv::imwrite(input_file + "_result.jpg", res);
  LOG(INFO) << "done!";
}
  
