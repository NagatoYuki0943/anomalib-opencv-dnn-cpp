#include "inference.hpp"
#include <opencv2/opencv.hpp>


int main() {
    // 不支持patchcore,无法载入模型
    string model_path = "D:/ml/code/anomalib/results/efficient_ad/mvtec/bottle/run/weights/openvino/model.onnx";
    string meta_path  = "D:/ml/code/anomalib/results/efficient_ad/mvtec/bottle/run/weights/openvino/metadata.json";
    string image_path = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
    string image_dir  = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large";
    string save_dir   = "D:/ml/code/anomalib-opencv-dnn-cpp/result"; // 注意目录不会自动创建,要手动创建才会保存
    bool efficient_ad = true; // 是否使用efficient_ad模型

    // 创建推理器
    auto inference = Inference(model_path, meta_path, efficient_ad);

     // 单张图片推理
     cv::Mat image = readImage(image_path);
     Result result = inference.single(image);
     saveScoreAndImages(result.score, result.anomaly_map, image_path, save_dir);
     cv::resize(result.anomaly_map, result.anomaly_map, { 1500, 500 });
     cv::imshow("result", result.anomaly_map);
     cv::waitKey(0);

    // 多张图片推理
    inference.multi(image_dir, save_dir);
    return 0;
}
