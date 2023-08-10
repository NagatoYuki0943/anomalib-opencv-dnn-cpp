# 说明

> 适用于anomalib导出的onnx格式的模型
>
> 测试了fastflow,efficient_ad模型
>
> 不支持patchcore模型,载入失败

```yaml
# 模型配置文件中设置为onnx,导出openvino会导出onnx
optimization:
  export_mode: onnx # options: torch, onnx, openvino
```

> 使用 `trtexec` 转换模型

```shell
trtexec --onnx=model.onnx --saveEngine=model.engine
```

# 其他推理方式

> [anomalib-onnxruntime-cpp](https://github.com/NagatoYuki0943/anomalib-onnxruntime-cpp)
>
> [anomalib-openvino-cpp](https://github.com/NagatoYuki0943/anomalib-openvino-cpp)
>
> [anomalib-tensorrt-cpp](https://github.com/NagatoYuki0943/anomalib-tensorrt-cpp)

# example

```C++

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
