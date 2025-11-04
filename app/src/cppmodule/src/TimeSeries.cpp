#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <Dense>
#include "matplotlibcpp.h"
#include "ESN.hpp"   // ← あなたの cppmodule/esn/ESN.h に相当するヘッダを指定
#include "SUtil.hpp"

namespace plt = matplotlibcpp;
using namespace Eigen;
using namespace std;

const double FS = 30.0;

// CSVファイルの読み込み
MatrixXf LoadCSV(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        exit(1);
    }

    vector<vector<float>> data;
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<float> row;
        while (getline(ss, cell, ',')) {
            row.push_back(stof(cell));
        }
        data.push_back(row);
    }

    int rows = data.size();
    int cols = data[0].size();
    MatrixXf mat(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            mat(i, j) = data[i][j];

    return mat;
}

// 結果を保存（グラフ描画）
// 結果を保存（グラフ描画）
void SaveResult(const std::vector<std::vector<float>>& inputX,
                const std::vector<std::vector<float>>& inputY,
                const std::vector<std::vector<float>>& predictY,
                const std::string& timeSeriesFileName,
                const std::string& errorFileName)
{
    if (inputX.empty() || inputY.empty() || predictY.empty()) {
        std::cerr << "Error: 入力データが空です。" << std::endl;
        return;
    }

    // データ数を取得
    size_t T = std::min({inputX.size(), inputY.size(), predictY.size()});

    std::vector<double> time(T), y_true(T), y_pred(T), y_err(T);

    for (size_t i = 0; i < T; ++i) {
        // 各ベクトルは1列想定：inputX[i][0], inputY[i][0], predictY[i][0]
        double x_val = (inputX[i].empty()) ? 0.0 : inputX[i][0];
        double y_val = (inputY[i].empty()) ? 0.0 : inputY[i][0];
        double p_val = (predictY[i].empty()) ? 0.0 : predictY[i][0];

        time[i] = static_cast<double>(i) / FS;
        y_true[i] = y_val;
        y_pred[i] = p_val;
        y_err[i]  = std::fabs(y_val - p_val);
    }

    // 予測 vs 元データ（時間変化）
    plt::figure_size(1200, 400);
    plt::plot(time, y_true, {{"color", "blue"}, {"label", "original"}});
    plt::plot(time, y_pred, {{"color", "red"}, {"label", "predict"}});
    plt::xlabel("time [s]");
    plt::ylabel("y");
    plt::legend();
    plt::grid(true);
    plt::save(timeSeriesFileName);
    plt::close();

    // 誤差のグラフ
    plt::figure_size(1200, 400);
    plt::plot(time, y_err, {{"color", "purple"}});
    plt::xlabel("time [s]");
    plt::ylabel("error");
    plt::grid(true);
    plt::save(errorFileName);
    plt::close();
}

// 時系列予測メイン処理
void RunTimeSeriesPrediction()
{
    // データ読み込み
    auto trainData = readCSV("/root/app/data/LINE_ALBUM_今日のソフィ☀️🍉ver.7_250922_1/ROI_series_filtered.csv");
    auto testData = readCSV("/root/app/data/IMG_2047/ROI_series_filtered.csv");

    auto trainX = extractBlock(*trainData, 0, 0, trainData->size() - 1, 1);
    auto trainY = extractBlock(*trainData, 1, 0, trainData->size() - 1, 1);
    auto testX  = extractBlock(*testData, 0, 0, testData->size() - 1, 1);
    auto testY  = extractBlock(*testData, 1, 0, testData->size() - 1, 1);

    std::cout << "trainX size: " << trainX->size() << ", " << (*trainX)[0].size() << std::endl;
    std::cout << "trainY size: " << trainY->size() << ", " << (*trainY)[0].size() << std::endl;
    std::cout << "testX size: " << testX->size() << ", " << (*testX)[0].size() << std::endl;
    std::cout << "testY size: " << testY->size() << ", " << (*testY)[0].size() << std::endl;

    Write2DVectorToFile("/root/app/data/timeseries/trainX.csv", *trainX);
    Write2DVectorToFile("/root/app/data/timeseries/trainY.csv", *trainY);
    Write2DVectorToFile("/root/app/data/timeseries/testX.csv", *testX);
    Write2DVectorToFile("/root/app/data/timeseries/testY.csv", *testY);

    // ESNパラメータ設定
    int N_x = 300;
    float density = 0.15f;     //
    float input_scale = 0.5f;  //
    float rho = 0.9f;          //
    float fb_scale = 0.0f;
    float leaking_rate = 0.1f; //
    float y_scale = 0.5f;      //
    float y_shift = 0.5f;      //
    bool reset_reservoir_state = false;

    // ESNモデル生成
    ESN esn = ESN(1, 1, N_x, density, input_scale, rho, leaking_rate, fb_scale,
                false, 0, y_scale, y_shift, reset_reservoir_state,
                false, 1.0f, 1.0f, false, 0,
                false);

    // 学習
    esn.Train_cpp(*trainX, *trainY, 1e-1);

    // 訓練データでの予測
    auto predict_train_Y = esn.Predict_cpp(*trainX);
    SaveResult(*trainX, *trainY, *predict_train_Y,
               "/root/app/data/timeseries/trainData_predict.png",
               "/root/app/data/timeseries/trainData_error.png");

    // テストデータでの予測
    auto predict_test_Y = esn.Predict_cpp(*testX);
    SaveResult(*testX, *testY, *predict_test_Y,
               "/root/app/data/timeseries/testData_predict.png",
               "/root/app/data/timeseries/testData_error.png");
}

int main()
{
    RunTimeSeriesPrediction();
    return 0;
}
