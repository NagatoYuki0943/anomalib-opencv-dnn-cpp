#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <math.h>
#include <numeric>
#include "utils.h"

using namespace std;



class Inference {
private:
    MetaData meta{};    // 超参数
    cv::dnn::Net model; // model

public:
    /**
     * @param model_path    模型路径
     * @param meta_path     超参数路径
     */
    Inference(string& model_path, string& meta_path) {
        // 1.读取meta
        this->meta = getJson(meta_path);
        // 2.创建模型
        this->model = cv::dnn::readNetFromONNX(model_path);
        // 4.模型预热
        this->warm_up();
    }

    /**
     * 模型预热
     */
    void warm_up() {
        // 输入数据
        cv::Size size = cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]);
        cv::Scalar color = cv::Scalar(0, 0, 0);
        cv::Mat input = cv::Mat(size, CV_8UC3, color);
        this->infer(input);
    }

    ///**
    // * 推理单张图片
    // * @param image 原始图片
    // * @return      标准化的并所放到原图热力图和得分
    // */
    Result infer(cv::Mat & image) {
        // 1.保存图片原始高宽
        this->meta.image_size[0] = image.size().height;
        this->meta.image_size[1] = image.size().width;

        // 2.图片预处理
        cv::Mat resized_image = pre_process(image, this->meta);
        cv::Mat blob = cv::dnn::blobFromImage(resized_image);

        // 4.推理
        this->model.setInput(blob);
        cv::Mat anomaly_map = this->model.forward();
        double _, maxValue;    // 最大值，最小值
        cv::minMaxLoc(anomaly_map, &_, &maxValue);
        cv::Mat pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, maxValue);

        // 5.后处理:标准化,缩放到原图
        vector<cv::Mat> post_mat = post_process(anomaly_map, pred_score, this->meta);
        anomaly_map = post_mat[0];
        float score = post_mat[1].at<float>(0, 0);

        // 6.返回结果
        return Result{ anomaly_map, score };
    }

    /**
     * 单张图片推理
     * @param image    RGB图片
     * @return      标准化的并所放到原图热力图和得分
     */
    Result single(cv::Mat& image) {
        // time
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        // 1.推理单张图片
        Result result = this->infer(image);
        cout << "score: " << result.score << endl;

        // 2.生成其他图片(mask,mask边缘,热力图和原图的叠加)
        vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
        // time
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        cout << "infer time: " << end - start << " ms" << endl;

        // 3.保存显示图片
        // 将mask转化为3通道,不然没法拼接图片
        cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);
        // 拼接图片
        cv::Mat res;
        cv::hconcat(images, res);

        return Result{ res, result.score };
    }

    /**
     * 多张图片推理
     * @param image_dir 图片文件夹路径
     * @param save_dir  保存路径
     */
    void multi(string& image_dir, string& save_dir) {
        // 1.读取全部图片路径
        vector<cv::String> paths = getImagePaths(image_dir);

        vector<float> times;
        for (auto& image_path : paths) {
            // 2.读取单张图片
            cv::Mat image = readImage(image_path);

            // time
            auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            // 3.推理单张图片
            Result result = this->infer(image);
            cout << "score: " << result.score << endl;

            // 4.图片生成其他图片(mask,mask边缘,热力图和原图的叠加)
            vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
            // time
            auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            cout << "infer time: " << end - start << " ms" << endl;
            times.push_back(end - start);

            // 5.保存图片
            // 将mask转化为3通道,不然没法拼接图片
            cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);
            // 拼接图片
            cv::Mat res;
            cv::hconcat(images, res);
            saveScoreAndImages(result.score, res, image_path, save_dir);
        }

        // 6.统计数据
        double sumValue = accumulate(begin(times), end(times), 0.0); // accumulate函数就是求vector和的函数；
        double avgValue = sumValue / times.size();                   // 求均值
        cout << "avg infer time: " << avgValue << " ms" << endl;
    }
};