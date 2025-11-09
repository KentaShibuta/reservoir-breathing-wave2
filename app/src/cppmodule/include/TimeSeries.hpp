#ifndef TIMESERIES_H_
#define TIMESERIES_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "ESN.hpp"
#include "SUtil.hpp"

class TimeSeries{
    private:
        // ESNパラメータ設定
        int m_N_x = 300;
        float m_density = 0.15f;     //
        float m_input_scale = 0.5f;  //
        float m_rho = 0.9f;          //
        float m_fb_scale = 0.0f;
        float m_leaking_rate = 0.1f; //
        float m_y_scale = 0.5f;      //
        float m_y_shift = 0.5f;      //
        bool m_reset_reservoir_state = false;
        float m_beta = 0.1f;

    public:
        void Train(const std::string &inputFileName, const std::string &weightFileName);
        std::unique_ptr<std::vector<std::vector<float>>> Predict(const std::string &inputFileName, const std::string &weightFileName);
};

class TimeSeriesMain{
    public:
        void Train(const std::string &trainFileName, const std::string &weightFileName, const std::string &predictOutputFileName);
        void Predict(const std::string &testFileName, const std::string &weightFileName, const std::string &predictOutputFileName);
};

#endif // TIMESERIES_H_
