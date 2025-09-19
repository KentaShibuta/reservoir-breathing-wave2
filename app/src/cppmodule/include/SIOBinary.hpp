#ifndef SIOBINARY_H_
#define SIOBINARY_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint> // int32_t, int64_tを使用するために必要

// テンプレート関数を使用して、任意の型のバイナリデータをファイルに書き込む
template<typename T>
inline void Write_bin(const std::vector<std::vector<T>>& data_to_save, const std::string& output_file_name) {
    std::ofstream file(output_file_name, std::ios::out | std::ios::binary);
    if (!file) {
        throw std::runtime_error("ファイルを開けませんでした: " + output_file_name);
    }

    // 形状情報（ヘッダー）を書き込む
    // Pythonの'ii'フォーマットに対応するint32_tを使用
    int32_t rows = static_cast<int32_t>(data_to_save.size());
    int32_t cols = static_cast<int32_t>(data_to_save[0].size());
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    // 配列データを書き込む
    for (const auto& row : data_to_save) {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(T));
    }

    std::cout << "保存しました: " << output_file_name << std::endl;
}

// テンプレート関数を使用して、任意の型のバイナリデータをファイルから読み込む
template<typename T>
inline void Read_bin(std::vector<std::vector<T>>& loaded_data, const std::string& input_file_name) {
    std::ifstream file(input_file_name, std::ios::in | std::ios::binary);
    if (!file) {
        throw std::runtime_error("ファイルが見つかりません: " + input_file_name);
    }

    // 形状情報（ヘッダー）を読み込む
    int32_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    std::cout << "読み込んだ形状: " << rows << " x " << cols << std::endl;

    // データの本体を読み込む
    size_t row_size = static_cast<size_t>(rows);
    size_t col_size = static_cast<size_t>(cols);
    loaded_data.resize(row_size, std::vector<T>(col_size));
    for (size_t i = 0; i < row_size; ++i) {
        file.read(reinterpret_cast<char*>(loaded_data[i].data()), cols * sizeof(T));
    }

    std::cout << "読み込みました: " << input_file_name << std::endl;
}

// StandardScalerに相当するパラメータを保存する関数
inline void Write_scaler_bin(const std::vector<double>& mean, const std::vector<double>& scale, const std::string& output_file_name) {
    std::ofstream file(output_file_name, std::ios::out | std::ios::binary);
    if (!file) {
        throw std::runtime_error("ファイルを開けませんでした: " + output_file_name);
    }
    
    // 特徴量数を書き込む
    int32_t n_features = static_cast<int32_t>(mean.size());
    file.write(reinterpret_cast<const char*>(&n_features), sizeof(n_features));
    
    // mean_とscale_のデータを書き込む
    file.write(reinterpret_cast<const char*>(mean.data()), mean.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(scale.data()), scale.size() * sizeof(double));

    std::cout << "保存しました: " << output_file_name << std::endl;
}

// StandardScalerに相当するパラメータを読み込む関数
inline void Read_scaler_bin(std::vector<double>& mean, std::vector<double>& scale, const std::string& input_file_name) {
    std::ifstream file(input_file_name, std::ios::in | std::ios::binary);
    if (!file) {
        throw std::runtime_error("ファイルが見つかりません: " + input_file_name);
    }

    // 特徴量数を読み込む
    int32_t n_features;
    file.read(reinterpret_cast<char*>(&n_features), sizeof(n_features));

    // mean_とscale_を読み込む
    mean.resize(n_features);
    scale.resize(n_features);
    file.read(reinterpret_cast<char*>(mean.data()), mean.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(scale.data()), scale.size() * sizeof(double));

    std::cout << "読み込みました: " << input_file_name << std::endl;
}

// 推論結果を標準化する関数
inline std::vector<float> Standardize_features(const std::vector<float>& features, const std::vector<double>& mean, const std::vector<double>& scale) {
    if (mean.size() != scale.size() || features.size() % mean.size() != 0) {
        throw std::invalid_argument("features, mean, and scaleのサイズが一致しません。");
    }

    size_t num_features = mean.size();
    size_t num_samples = features.size() / num_features;
    std::vector<float> standardized_features(features.size());

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            size_t index = i * num_features + j;
            if (scale[j] != 0.0f) {
                standardized_features[index] = (features[index] - mean[j]) / scale[j];
            } else {
                // scaleが0の場合は元の値をそのまま使用
                standardized_features[index] = features[index];
            }
        }
    }
    return standardized_features;
}

#endif // SIOBINARY_H_