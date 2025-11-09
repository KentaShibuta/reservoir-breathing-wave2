#include "TimeSeries.hpp"

void TimeSeries::Train(const std::string &inputFileName, const std::string &weightFileName)
{
    auto data = readCSV(inputFileName);
    auto X = extractBlock(*data, 0, 0, data->size() - 1, 1);
    auto Y = extractBlock(*data, 1, 0, data->size() - 1, 1);

    std::cout << "X size: " << X->size() << ", " << (*X)[0].size() << std::endl;
    std::cout << "Y size: " << Y->size() << ", " << (*Y)[0].size() << std::endl;

    // ESNモデル生成
    ESN esn = ESN(1, 1, m_N_x, m_density, m_input_scale, m_rho, m_leaking_rate, m_fb_scale,
                false, 0, m_y_scale, m_y_shift, m_reset_reservoir_state,
                false, 1.0f, 1.0f, false, 0,
                false);

    // 学習
    esn.Train_cpp(*X, *Y, m_beta);

    // 書き込み
    Write_bin(esn.vec_w_out, weightFileName);
}

std::unique_ptr<std::vector<std::vector<float>>> TimeSeries::Predict(const std::string &inputFileName, const std::string &weightFileName)
{
    std::cout << "read input file name: " << inputFileName << std::endl;
    auto data = readCSV(inputFileName);
    auto X  = extractBlock(*data, 0, 0, data->size() - 1, 1);
    auto Y  = extractBlock(*data, 1, 0, data->size() - 1, 1);

    std::cout << "X size: " << X->size() << ", " << (*X)[0].size() << std::endl;
    std::cout << "Y size: " << Y->size() << ", " << (*Y)[0].size() << std::endl;

    // ESNモデル生成
    ESN esn = ESN(1, 1, m_N_x, m_density, m_input_scale, m_rho, m_leaking_rate, m_fb_scale,
                false, 0, m_y_scale, m_y_shift, m_reset_reservoir_state,
                false, 1.0f, 1.0f, false, 0,
                false);

    esn.SetWoutFromWeightFile(weightFileName);

    // 訓練データでの予測
    auto predict_Y = esn.Predict_cpp(*X);

    return predict_Y;
}

void TimeSeriesMain::Train(const std::string &trainFileName, const std::string &weightFileName, const std::string &predictOutputFileName)
{
    // 訓練を実行する
    TimeSeries ts = TimeSeries();
    ts.Train(trainFileName, weightFileName);

    // 推論結果をcsv形式で保存する
    auto predict_Y = ts.Predict(trainFileName, weightFileName);
    Write2DVectorToFile(predictOutputFileName, *predict_Y);
}

void TimeSeriesMain::Predict(const std::string &testFileName, const std::string &weightFileName, const std::string &predictOutputFileName)
{
    // 推論を実行する
    TimeSeries ts = TimeSeries();

    // 推論結果をcsv形式で保存する
    auto predict_Y = ts.Predict(testFileName, weightFileName);
    Write2DVectorToFile(predictOutputFileName, *predict_Y);
}
