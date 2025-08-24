// --------------------------------------------------------------
// main.hppをインクルードして、関数や型定義を利用可能にする
// --------------------------------------------------------------
#include "SOnnxRuntime.hpp"

// --------------------------------------------------------------
// main.hppで宣言した関数の定義
// --------------------------------------------------------------
// void setup() {
//     std::cout << "Setup completed." << std::endl;
// }

// void run_logic() {
//     std::cout << "Running program logic." << std::endl;
// }

// void cleanup() {
//     std::cout << "Cleanup completed." << std::endl;
// }

// int calculate_sum(int a, int b) {
//     return a + b;
// }

// --------------------------------------------------------------
// プログラムのエントリーポイント
// --------------------------------------------------------------
int main() {
    try {
        std::cout << "Program started." << std::endl;

        try {
            std::string model_path = "/root/app/model/vgg16_block5_conv3.onnx";
            const int64_t batch_size = 25;

            // 入力データの準備
            std::vector<int64_t> input_dims = {batch_size, 224, 224, 3};
            size_t input_tensor_size = batch_size * 224 * 224 * 3;
            std::vector<float> input_tensor_values(input_tensor_size, 1.0f);

            std::cout << "Starting inference with batch size: " << batch_size << std::endl;
            std::cout << "Model path: " << model_path << std::endl;

            // 入力データの形状を出力
            std::cout << "Input shape (NHWC): ";
            for (const auto& dim : input_dims) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;

            // 推論関数を呼び出し、結果と形状を一般的な型で取得
            auto result_pair = SOnnxRuntime::runInference(model_path, input_tensor_values, input_dims);
            std::vector<float> result_data = result_pair.first;
            std::vector<int64_t> result_shape = result_pair.second;

            std::cout << "Inference completed." << std::endl;
            
            // 推論結果の形状を出力
            std::cout << "Output shape (NHWC): ";
            for (const auto& dim : result_shape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            
            // 推論結果の表示
            std::cout << "First 10 output values: ";
            for (size_t i = 0; i < 10 && i < result_data.size(); i++) {
                std::cout << result_data[i] << " ";
            }
            std::cout << std::endl;

        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        }

        std::cout << "Program finished." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1; // エラー終了
    }

    return 0; // 正常終了
}
