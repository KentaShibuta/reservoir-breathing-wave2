#ifndef SIMAGE_H_
#define SIMAGE_H_

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>

// 画像の中心座標を取得する（Pythonのdetect_cat_centerに相当）
inline cv::Point detect_image_center(const cv::Mat& image) {
    // 仮実装: 画像の中心を猫の中心と仮定
    int h = image.rows;
    int w = image.cols;
    return cv::Point(w / 2, h / 2);
}

// 画像を正方形に切り出す（Pythonのcrop_squareに相当）
inline cv::Mat crop_square(const cv::Mat& image, cv::Point center, int size, bool save = false, const std::string& image_name = "") {
    int h = image.rows;
    int w = image.cols;
    int cx = center.x;
    int cy = center.y;
    int half = size / 2;

    int left = std::max(0, cx - half);
    int top = std::max(0, cy - half);
    int right = std::min(w, cx + half);
    int bottom = std::min(h, cy + half);

    cv::Rect roi(left, top, right - left, bottom - top);
    cv::Mat cropped = image(roi);

    // 足りない部分があれば黒でパディング
    cv::Mat padded = cv::Mat::zeros(size, size, CV_8UC3);
    cropped.copyTo(padded(cv::Rect(0, 0, cropped.cols, cropped.rows)));

    if (save) {
        std::filesystem::path output_path = "output_images"; // Pythonコードのoutput_image_dirに相当
        if (!std::filesystem::exists(output_path)) {
            std::filesystem::create_directories(output_path);
        }
        
        std::filesystem::path filename = std::filesystem::path(image_name).stem().string() + "_square.png";
        cv::imwrite((output_path / filename).string(), padded);
    }

    return padded;
}

// 指定したパスの画像を正方形に切り出す（Pythonのget_crop_squareに相当）
inline cv::Mat get_crop_square(const std::string& input_img_path, int crop_size) {
    cv::Mat image = cv::imread(input_img_path);
    if (image.empty()) {
        std::cerr << "Could not read the image: " << input_img_path << std::endl;
        return cv::Mat();
    }
    
    std::filesystem::path image_path(input_img_path);
    std::string image_name = image_path.stem().string();

    // 中心の座標を定義
    cv::Point center = detect_image_center(image);
    
    // 上記座標を中心に、正方形で切り出す
    cv::Mat cropped = crop_square(image, center, crop_size, true, image_name); // `save`引数を`true`に変更
    
    return cropped;
}

// 画像の前処理を行う
inline std::vector<float> preprocess_image(const cv::Mat& image, const std::vector<int64_t>& dims) {
    cv::Mat resized_image;

    // 正しくリサイズするために、dimsから高さを dims[1]、幅を dims[2] として取得
    int target_height = dims[1];
    int target_width = dims[2];

    cv::resize(image, resized_image, cv::Size(target_width, target_height)); 

    // BGRからRGBに変換
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    // float型に変換し、正規化 (0-1)
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);
    
    // HWCからCHW形式に変換
    std::vector<float> input_data(resized_image.total() * resized_image.channels());
    size_t channels = resized_image.channels();
    size_t rows = resized_image.rows;
    size_t cols = resized_image.cols;
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t col = 0; col < cols; ++col) {
                input_data[c * rows * cols + r * cols + col] = resized_image.at<cv::Vec3f>(r, col)[c];
            }
        }
    }
    
    return input_data;
}

// 画像の前処理を行う（バッチ対応版）
// TensorFlow/Keras由来のONNXモデル用 (NHWC入力)
inline std::vector<float> preprocess_batch_images(
    const std::vector<cv::Mat>& images, const std::vector<int64_t>& dims) {
    
    std::vector<float> batch_data;
    batch_data.reserve(images.size() * dims[1] * dims[2] * dims[3]);

    // ONNXモデルが要求するサイズ (N, H, W, C)
    int target_height = dims[1]; // 224
    int target_width  = dims[2]; // 224
    int channels      = dims[3]; // 3

    // VGG16の平均値 (BGR順)
    const std::vector<float> mean = {103.939f, 116.779f, 123.68f};

    for (const auto& image : images) {
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(target_width, target_height));

        // float型に変換 (255で割らない)
        resized_image.convertTo(resized_image, CV_32F);

        // HWCのまま格納
        for (int r = 0; r < target_height; ++r) {
            for (int c = 0; c < target_width; ++c) {
                cv::Vec3f pixel = resized_image.at<cv::Vec3f>(r, c);
                for (int ch = 0; ch < channels; ++ch) {
                    batch_data.push_back(pixel[ch] - mean[ch]);
                }
            }
        }
    }

    return batch_data;
}



#endif // SIMAGE_H_