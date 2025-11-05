#include "TimeSeries.hpp"
#include "SUtil.hpp"
#include <string>

int main()
{
    std::string trainFileName = "/root/app/data/LINE_ALBUM_今日のソフィ☀️🍉ver.7_250922_1/ROI_series_filtered.csv";
    std::string testFileName = "/root/app/data/IMG_2047/ROI_series_filtered.csv";
    std::string weightFileName = generateUniqueFilename("/root/app/data/timeseries/weights_", ".bin");
    double FS = 30.0;

    TimeSeries ts = TimeSeries();
    ts.Train(trainFileName, weightFileName);


    // 訓練データを用いて推論実行
    std::string predictTrainFileName = ts.Predict(trainFileName, weightFileName);

    std::string trainYFileName = generateUniqueFilename("/root/app/data/timeseries/train_y_", ".png");
    std::string trainErrorFileName = generateUniqueFilename("/root/app/data/timeseries/train_error_", ".png");
    auto Y_train = readCSV(trainFileName);
    auto Y_train_predict = readCSV(predictTrainFileName);
    SaveResult(*Y_train, *Y_train_predict, trainYFileName, trainErrorFileName, FS);


    // テストデータを用いて推論実行
    std::string predictTestFileName = ts.Predict(testFileName, weightFileName);

    std::string testYFileName = generateUniqueFilename("/root/app/data/timeseries/test_y_", ".png");
    std::string testErrorFileName = generateUniqueFilename("/root/app/data/timeseries/test_error_", ".png");
    auto Y_test = readCSV(testFileName);
    auto Y_test_predict = readCSV(predictTestFileName);
    SaveResult(*Y_test, *Y_test_predict, testYFileName, testErrorFileName, FS);

    return 0;
}