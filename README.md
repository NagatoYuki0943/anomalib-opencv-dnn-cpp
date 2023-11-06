# 说明

> 适用于anomalib导出的onnx格式的模型
>
> opencv版本 4.7.0
>
> 测试了fastflow模型
>
> 不支持patchcore,efficient_ad模型,载入失败(opencv4.8.0版本载入成功,不过所有模型的结果都错误)

```yaml
# 模型配置文件中设置为onnx,导出openvino会导出onnx
optimization:
  export_mode: onnx # options: torch, onnx, openvino
```

# 其他推理方式

> [anomalib-onnxruntime-cpp](https://github.com/NagatoYuki0943/anomalib-onnxruntime-cpp)
>
> [anomalib-openvino-cpp](https://github.com/NagatoYuki0943/anomalib-openvino-cpp)
>
> [anomalib-tensorrt-cpp](https://github.com/NagatoYuki0943/anomalib-tensorrt-cpp)

# example

```C++
#include "inference.hpp"
#include <opencv2/opencv.hpp>


int main() {
    // 不支持patchcore,无法载入模型
    string model_path = "D:/ml/code/anomalib/results/fastflow/mvtec/bottle/run/weights/openvino/model.onnx";
    string meta_path  = "D:/ml/code/anomalib/results/fastflow/mvtec/bottle/run/weights/openvino/metadata.json";
    string image_path = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
    string image_dir  = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large";
    string save_dir   = "D:/ml/code/anomalib-opencv-dnn-cpp/result"; // 注意目录不会自动创建,要手动创建才会保存
    bool efficient_ad = false; // 是否使用efficient_ad模型

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
```

# 下载topencv

> https://opencv.org

## 配置环境变量

```yaml
# opencv
$opencv_path\build\x64\vc16\bin
```

# 关于include文件夹

> include文件夹是rapidjson的文件，用来解析json

# Cmake

> cmake版本要设置 `CMakeLists.txt` 中 opencv 路径为自己的路径

# 查看是否缺失dll

> https://github.com/lucasg/Dependencies 这个工具可以查看exe工具是否缺失dll

# 第三方库

处理json使用了rapidjson https://rapidjson.org/zh-cn/

opencv方式参考了mmdeploy https://github.com/open-mmlab/mmdeploy/tree/master/csrc/mmdeploy/utils/opencv
