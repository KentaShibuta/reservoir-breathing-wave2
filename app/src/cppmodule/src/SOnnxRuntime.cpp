#include "SOnnxRuntime.hpp"

std::pair<std::vector<float>, std::vector<int64_t>> SOnnxRuntime::runInference(
    const std::string& model_path,
    const std::vector<float>& input_data,
    const std::vector<int64_t>& input_dims)
{
    // ORT 環境とセッションを作成
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, model_path.c_str(), session_options);

    // モデルの入出力名を取得
    std::vector<std::string> input_names = session.GetInputNames();
    std::vector<std::string> output_names = session.GetOutputNames();

    // 入力テンソルを作成
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(input_data.data()),
        input_data.size(),
        input_dims.data(),
        input_dims.size()
    );

    // 推論実行
    const char* input_names_c[] = {input_names[0].c_str()};
    const char* output_names_c[] = {output_names[0].c_str()};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names_c, 
        &input_tensor, 
        1, 
        output_names_c, 1
    );

    // 出力テンソルを取得し、std::vector<float>にコピー
    const Ort::Value& output_tensor = output_tensors.front();
    const float* output_data_ptr = output_tensor.GetTensorData<float>();
    size_t output_tensor_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    
    // 出力形状を取得
    std::vector<int64_t> output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();

    // データと形状をペアで返す
    return {std::vector<float>(output_data_ptr, output_data_ptr + output_tensor_size), output_shape};

    // 💡 推論結果のfloatデータをdoubleに変換
    //std::vector<double> output_data_double(output_tensor_size);
    //for (size_t i = 0; i < output_tensor_size; ++i) {
    //    output_data_double[i] = static_cast<double>(output_data_ptr[i]);
    //}

    // doubleに変換したデータと形状をペアで返す
    //return {output_data_double, output_shape};
}
