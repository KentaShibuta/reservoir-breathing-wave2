#include "SOnnxRuntime.hpp"
#include "SMovie.hpp"
#include "SImage.hpp"
#include "SIOBinary.hpp"
#include "ESN.hpp"

// --------------------------------------------------------------
// プログラムのエントリーポイント
// --------------------------------------------------------------
int main() {
    // 1. 画像をクロップ
    std::string input_image_path = "/root/app/data/object_detection_test_data/LINE_ALBUM_ネコ写真資料２_250518_21.jpg";
    //std::string input_image_path = "/root/app/data/object_detection_test_data/LINE_ALBUM_ネコ写真資料２_250518_52.jpg";
    std::string model_path = "/root/app/model/vgg16_block5_conv3.onnx";
    int crop_size = 800;

    cv::Mat cropped_image = get_crop_square(input_image_path, crop_size);

    if (!cropped_image.empty()) {
        std::cout << "Cropped image saved successfully to output_images directory." << std::endl;
    }

    int h = cropped_image.rows;
    int w = cropped_image.cols;
    int grid_size = 3;
    int patch_h = h / grid_size;
    int patch_w = w / grid_size;

    int division_num = 6;
    int step_size = static_cast<int>(static_cast<double>(crop_size) / division_num);

    // スライディングウィンドウで画像を分割
    std::vector<cv::Mat> batch_images;

    // デバッグ用の保存ディレクトリを作成
    /*
    std::string output_dir = "./output_patches/";
    std::filesystem::create_directories(output_dir);
    int patch_count = 0;
    */

    // `step_size`の代わりに`patch_h`と`patch_w`をステップとして使用
    std::vector<cv::Rect> rect_list;
    for (int y = 0; y <= h - patch_h; y += step_size) {
        for (int x = 0; x <= w - patch_w; x += step_size) {
            cv::Rect roi(x, y, patch_w, patch_h);
            cv::Mat patch = cropped_image(roi);
            batch_images.push_back(patch.clone()); // ROIは参照なので、クローンを作成してプッシュ

            rect_list.push_back(roi);

            // デバッグ用に画像を保存
            /*
            std::string output_path = output_dir + "patch_" + std::to_string(patch_count) + ".jpg";
            cv::imwrite(output_path, patch);
            patch_count++;
            */
        }
    }

    std::cout << "Image divided into " << batch_images.size() << " patches." << std::endl;


    // 2. モデルの入力要件を定義
    // VGG16は {バッチサイズ, 高さ, 幅, チャンネル数} の形状を要求
    int batch_size = batch_images.size();
    int input_channel = 3;
    int input_height = 224;
    int input_width = 224;
    std::vector<int64_t> input_dims = {
        static_cast<int64_t>(batch_size),
        static_cast<int64_t>(input_height),
        static_cast<int64_t>(input_width),
        static_cast<int64_t>(input_channel)
    };

    // 3. クロップした画像を前処理
    std::vector<float> input_data = preprocess_batch_images(batch_images, input_dims);

    // 4. ONNX Runtimeで推論を実行
    SOnnxRuntime onnx_runtime;
    std::pair<std::vector<float>, std::vector<int64_t>> result = onnx_runtime.runInference(model_path, input_data, input_dims);

    // 5. 結果を標準化
    std::vector<double> mean;
    std::vector<double> scale;
    std::string scale_file_path = "/root/app/model/20250824_070228_scale.dat"; // 適切なファイルパスに修正してください
    std::string esn_w_file_path = "/root/app/model/20250824_070327_700_wout.dat";

    try {
        Read_scaler_bin(mean, scale, scale_file_path);
        std::vector<float> standardized_features = Standardize_features(result.first, mean, scale);

        // 標準化された特徴量のサイズと形状を表示
        std::cout << "Inference completed. Standardized feature vector size: " << standardized_features.size() << std::endl;
        std::cout << "Output shape: [";
        for (size_t i = 0; i < result.second.size(); ++i) {
            std::cout << result.second[i] << (i == result.second.size() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

        int batch = static_cast<int>(result.second[0]);
        int h = static_cast<int>(result.second[1]);
        int w = static_cast<int>(result.second[2]);
        int c = static_cast<int>(result.second[3]);

        // reshape(-1, 512) の行数は batch * h * w
        int new_rows = batch * h * w;
        int new_cols = c;

        // 2次元配列に変換（C++ではvector<vector<float>>で扱うのが便利）
        std::vector<std::vector<float>> reshaped(new_rows, std::vector<float>(new_cols));

        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < h; ++i) {
                for (int j = 0; j < w; ++j) {
                    int row_index = b * (h * w) + i * w + j;
                    for (int k = 0; k < c; ++k) {
                        // 元のインデックス = ((((b * h) + i) * w) + j) * c + k
                        int idx = (((b * h) + i) * w + j) * c + k;
                        reshaped[row_index][k] = standardized_features[idx];
                    }
                }
            }
        }

        float density = 0.1f;
        float input_scale = 1.0f;
        float rho = 0.9f;
        float fb_scale = 0.0f;
        float leaking_rate = 1.0f;
        float y_scale = 0.5f;
        float y_shift = 0.5f;
        bool reset_reservoir_state = true;

        float threshold = 0.55f;
        size_t data_length = 196;

        ESN esn = ESN(512, 2, 700, density, input_scale, rho, leaking_rate, fb_scale,
                        false, 0, y_scale, y_shift, reset_reservoir_state,
                        false, 1.0f, 1.0f, false, 0,
                        false);
        esn.SetWoutFromWeightFile(esn_w_file_path);
        auto y = esn.Predict_cpp(reshaped);

        std::vector<int> pred_test;
        int start = 0;
        const int len_batch_list = rect_list.size();

        cv::Mat img = cv::Mat::zeros(800, 800, CV_8UC3); // 800x800の黒い画像

        for (int i = 0; i < len_batch_list; ++i) {
            // Y[start:start+data_length, :] に相当
            std::vector<float> tmp_positive;
            for (size_t j = 0; j < data_length; ++j) {
                tmp_positive.push_back((*y)[start + j][1]);
            }

            // condition と max_index の処理
            std::vector<int> max_index;
            for (double val : tmp_positive) {
                max_index.push_back(val >= threshold ? 1 : 0);
            }

            // histogram の処理
            std::vector<int> histogram(2, 0);
            for (int val : max_index) {
                if (val == 0 || val == 1) {
                    histogram[val]++;
                }
            }

            // argmax の処理
            int argmax_val = 0;
            if (histogram[1] > histogram[0]) {
                argmax_val = 1;
            }

            pred_test.push_back(argmax_val);

            if (pred_test[i] == 1) {
                cv::Rect rect = rect_list[i];
                std::cout << i + 1 << " / " << len_batch_list << ": " << pred_test[i] << " 【Positive】" << std::endl;
                std::cout << "x1:" << rect.x << ", y1:" << rect.y << ", x2:" << rect.x + rect.width << ", y2:" << rect.y + rect.height << std::endl;
                cv::rectangle(cropped_image, rect, cv::Scalar(0, 0, 255), 2);
            } else {
                std::cout << i + 1 << " / " << len_batch_list << ": " << pred_test[i] << " 【Negative】" << std::endl;
            }

            start = start + data_length;
        }

        // 結果の画像をファイルに保存
        std::string output_filename = "output_image.jpg";
        cv::imwrite(output_filename, cropped_image);
        std::cout << "結果の画像を " << output_filename << " に保存しました。" << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "エラー: " << e.what() << std::endl;
        return 1;
    }

    return 0; // 正常終了
}
