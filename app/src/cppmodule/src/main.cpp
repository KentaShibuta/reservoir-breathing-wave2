#include "SOnnxRuntime.hpp"
#include "SMovie.hpp"

// --------------------------------------------------------------
// プログラムのエントリーポイント
// --------------------------------------------------------------
int main() {
    try {
        std::cout << "Program started." << std::endl;

        std::cout << "Start onnx runtime sample." << std::endl;
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

        std::cout << "End onnx runtime sample." << std::endl;

        std::cout << "Start opencv sample." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        std::string fName = "/root/app/data/LINE_ALBUM_25621つくば🐶ランドよ_250707_1.mp4";

        SMovie movie;
        movie.Read(fName);
        movie.CreateFrames();
        movie.Stabilize();
        movie.CreateOutputVideo();
        movie.Release();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_sec = end - start;
        std::cout << "実行時間: " << elapsed_sec.count() << " 秒" << std::endl;

        std::cout << "End opencv sample." << std::endl;

        std::cout << "Program finished." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1; // エラー終了
    }

    return 0; // 正常終了
}
