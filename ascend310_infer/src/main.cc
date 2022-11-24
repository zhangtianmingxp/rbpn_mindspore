
#include <sys/time.h>
#include <gflags/gflags.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>

#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "include/dataset/execute.h"
#include "include/dataset/vision.h"
#include "inc/utils.h"

using mindspore::Context;
using mindspore::Serialization;
using mindspore::Model;
using mindspore::Status;
using mindspore::MSTensor;
using mindspore::dataset::Execute;
using mindspore::ModelType;
using mindspore::GraphCell;
using mindspore::kSuccess;


// 从命令行获取参数
DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(dataset_path, "", "dataset path");
DEFINE_string(x_path, "", "x path");
DEFINE_string(neighbor_path, "", "neighbor path");
DEFINE_string(flow_path, "", " flow path");
DEFINE_int32(device_id, 0, "device id");

int main(int argc, char **argv) {
  // 从命令行获取参数  mindir的路径
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (RealPath(FLAGS_mindir_path).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }

  // 配置环境变量
  auto context = std::make_shared<Context>();
  auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310->SetDeviceID(FLAGS_device_id);
  // ascend310->SetPrecisionMode("allow_fp32_to_fp16");
  context->MutableDeviceInfo().push_back(ascend310);

  // 导入模型
  mindspore::Graph graph;
  Serialization::Load(FLAGS_mindir_path, ModelType::kMindIR, &graph);

  std::cout << "start build model." << std::endl;
  Model model;
  Status ret = model.Build(GraphCell(graph), context);
  if (ret != kSuccess) {
    std::cout << "ERROR: Build failed." << std::endl;
    return 1;
  }
  std::cout << "finish build model." << std::endl;

  // 获取模型输入格式
  std::vector<MSTensor> model_inputs = model.GetInputs();
  if (model_inputs.empty()) {
    std::cout << "Invalid model, inputs is empty." << std::endl;
    return 1;
  }

  auto x_files = GetAllFiles(FLAGS_x_path);
  auto neighbor_files = GetAllFiles(FLAGS_neighbor_path);
  auto flow_files = GetAllFiles(FLAGS_flow_path);

  if (x_files.empty()) {
    std::cout << "ERROR: x data empty." << std::endl;
    return 1;
  }
  if (neighbor_files.empty()) {
    std::cout << "ERROR: neighbor data empty." << std::endl;
    return 1;
  }
  if (flow_files.empty()) {
    std::cout << "ERROR: flow data empty." << std::endl;
    return 1;
  }

  std::map<double, double> costTime_map;
  // 逐个二进制文件导入
  size_t size = x_files.size();
  for (size_t i = 0; i < size; ++i) {
    struct timeval start = {0};
    struct timeval end = {0};
    double startTimeMs;
    double endTimeMs;
    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;


    // 导入二进制流文件
    auto x = ReadFileToTensor(x_files[i]);
    auto neighbor = ReadFileToTensor(neighbor_files[i]);
    auto flow = ReadFileToTensor(flow_files[i]);
    inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                        x.Data().get(), x.DataSize());
    inputs.emplace_back(model_inputs[1].Name(), model_inputs[1].DataType(), model_inputs[1].Shape(),
                        neighbor.Data().get(), neighbor.DataSize());
    inputs.emplace_back(model_inputs[2].Name(), model_inputs[2].DataType(), model_inputs[2].Shape(),
                        flow.Data().get(), flow.DataSize());
    gettimeofday(&start, nullptr);
    // 推理得到预测结果
    ret = model.Predict(inputs, &outputs);
    gettimeofday(&end, nullptr);
    if (ret != kSuccess) {
      std::cout << "Predict " << x_files[i] << " failed." << std::endl;
      return 1;
    }
    std::cout << "predict finished" << std::endl;
    // 将二进制结果保存为二进制流文件
    WriteResult(x_files[i], outputs);
    std::cout << "writeresult finished" << std::endl;

    startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
  }
  // 记录信息日志
  double average = 0.0;
  int inferCount = 0;

  for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
    double diff = 0.0;
    diff = iter->second - iter->first;
    average += diff;
    inferCount++;
  }
  average = average / inferCount;
  std::stringstream timeCost;
  timeCost << "NN inference cost average time: "<< average << " ms of infer_count " << inferCount << std::endl;
  std::cout << "NN inference cost average time: "<< average << "ms of infer_count " << inferCount << std::endl;
  std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
  std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
  fileStream << timeCost.str();
  fileStream.close();
  costTime_map.clear();
  return 0;
}
