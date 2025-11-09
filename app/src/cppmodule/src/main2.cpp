#include <string>
#include "TimeSeries.hpp"
#include "SUtil.hpp"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

// ビルドコマンド
// clang++ -O3 -std=c++23 -stdlib=libc++ -I/root/app/lib/spdlog/include -I/root/app/lib/eigen3 -I/usr/local/include/python3.12 -I/usr/local/lib/python3.12/site-packages/numpy/_core/include -I/root/app/lib/matplotlibcpp -I/root/app/src/cppmodule/include /root/app/src/cppmodule/src/ESN.cpp /root/app/src/cppmodule/src/SMatrix2.cpp /root/app/src/cppmodule/src/TimeSeries.cpp /root/app/src/cppmodule/src/main2.cpp  -lpython3.12 -I/usr/local/include/python3.12 -I/usr/local/lib/python3.12/site-packages/pybind11/include -o timeseries_app.out

// 結果を保存（グラフ描画）
void SaveResult(
                const std::vector<std::vector<float>>& inputY,
                const std::vector<std::vector<float>>& predictY,
                const std::string& timeSeriesFileName,
                const std::string& errorFileName,
                double FS)
{
    if (inputY.empty() || predictY.empty()) {
        std::cerr << "Error: 入力データが空です。" << std::endl;
        return;
    }

    // データ数を取得
    size_t T = std::min({inputY.size(), predictY.size()});

    std::vector<double> time(T), y_true(T), y_pred(T), y_err(T);

    for (size_t i = 0; i < T; ++i) {
        // 各ベクトルは1列想定：inputY[i][0], predictY[i][0]
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

int main()
{
    TimeSeriesMain tsm = TimeSeriesMain();

    // ESNの訓練
    std::string trainFileName = "/root/app/data/LINE_ALBUM_今日のソフィ☀️🍉ver.7_250922_1/ROI_series_filtered.csv"; // 訓練データのパス
    std::string weightFileName = generateUniqueFilename("/root/app/data/timeseries/weights_", ".bin");           // 重みファイルを保存するパス
    std::string trainPredictFileName = generateUniqueFilename("/root/app/data/timeseries/output_", ".csv");      // 訓練データを入力とした推論結果を保存するパス
    tsm.Train(trainFileName, weightFileName, trainPredictFileName);
    // デバッグ用として、「4.」の結果と訓練データを画面上にグラフでプロットする
    double FS = 30.0;
    std::string trainYFileName = generateUniqueFilename("/root/app/data/timeseries/train_y_", ".png");
    std::string trainErrorFileName = generateUniqueFilename("/root/app/data/timeseries/train_error_", ".png");
    auto Y_train = readCSV(trainFileName);
    auto Y_train_predict = readCSV(trainPredictFileName);
    SaveResult(*Y_train, *Y_train_predict, trainYFileName, trainErrorFileName, FS);

    // ESNの推論
    std::string testFileName = "/root/app/data/IMG_2047/ROI_series_filtered.csv";                                // テストデータのパス
    std::string testPredictFileName = generateUniqueFilename("/root/app/data/timeseries/output_", ".csv");       // テストデータを入力とした推論結果を保存するパス
    tsm.TimeSeriesMain::Predict(testFileName, weightFileName, testPredictFileName);
    // デバッグ用として、「4.」の結果とテストデータと画面上にグラフでプロットする
    std::string testYFileName = generateUniqueFilename("/root/app/data/timeseries/test_y_", ".png");
    std::string testErrorFileName = generateUniqueFilename("/root/app/data/timeseries/test_error_", ".png");
    auto Y_test = readCSV(testFileName);
    auto Y_test_predict = readCSV(testPredictFileName);
    SaveResult(*Y_test, *Y_test_predict, testYFileName, testErrorFileName, FS);

    return 0;
}