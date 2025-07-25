#ifndef ESN_H_
#define ESN_H_

#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <memory>
#include <string>
#include <omp.h>
#include <random>
#include <Dense> // Eigen
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include "SMatrix.hpp"
#include "SMatrix2.hpp"
#include "SCirculationBuffer.hpp"

#ifdef USE_PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#endif

class Reservoir{
    public:
        inline Reservoir (){};

        inline void Init (size_t nx){
                m_nx = nx;
                auto x_ptr = std::make_unique<std::vector<float>>(nx, 0.0f);
                m_x = std::move(*x_ptr);
        }

        inline void SetX (const std::vector<float>& x){
                size_t i = 0;
                for (const auto &elem : x){
                        m_x[i] = elem;
                        i++;
                }
        }

        inline std::unique_ptr<std::vector<float>> GetX (){
                auto x_output = std::make_unique<std::vector<float>>(m_nx, 0.0f);
                for (size_t i = 0; i < (*x_output).size(); i++){
                        (*x_output)[i] = m_x[i];
                }
                return x_output;
        }
    private:
        std::vector<float> m_x;
        size_t m_nx;
};

class ESN{
    private:
        SMatrix2 m_matlib;
        //std::deque<std::vector<float>> vec_window;              // 分類時に時間平均をとるためのデータ格納場所
        bool m_classification;                                  // 扱う問題を分類タスクにするか？
        size_t m_average_window;                                // ウィンドウサイズ
        float m_y_scale;                                        // yのスケール
        float m_y_inv_scale;                                    // yのスケールの逆数
        float m_y_shift;                                        // yのシフト
        std::vector<float> m_y_prev;                            // フィードバック用に出力を保存する変数
        void set_Wout (const std::vector<std::vector<double>>& mat);
        Reservoir reservoir;
        SCirculationBuffer m_vec_window;
    
    public:
        std::vector<std::vector<float>> vec_u;
        std::vector<std::vector<float>> vec_w_in;
        std::vector<std::vector<float>> vec_w;
        std::vector<std::vector<float>> vec_w_out;
        std::vector<std::vector<float>> vec_w_fb;
        std::vector<float> vec_x;
        float a_alpha;
        size_t N;
        size_t N_window;
        size_t N_u;
        size_t N_x;
        size_t N_y;

        ESN();
#ifdef USE_PYBIND
        ESN(size_t n_u, size_t n_y, size_t n_x, float density, float input_scale, float rho, float leaking_rate=1.0f, float fb_scale=0.0f, bool classification=false, size_t average_window=0, float y_scale=1.0f, float y_shift=0.0f);
        ESN(py::array_t<float> u, py::array_t<float> w_in, py::array_t<float> w, py::array_t<float> w_out, py::array_t<float> x, float alpha);
#endif

        void Print();

#ifdef USE_PYBIND
        py::array_t<float> Predict(py::array_t<float> u);
        py::array_t<float> Train(py::array_t<float> u, py::array_t<float> d, float beta=0.0f);
        void SetWout(py::array_t<float> w_out);
        void SetWin(py::array_t<float> w_in);
        void SetW(py::array_t<float> w);
        void SetWfb(py::array_t<float> w_fb);
        py::array_t<float> GetWout();

        template <typename MatrixType, typename VectorType, typename T>
        std::unique_ptr<std::vector<std::vector<T>>> make_connection_mat(size_t N_x, T density, T rho);
#endif

#ifdef USE_PYBIND
        py::tuple GetInversePy2 (py::array_t<double> mat);
#endif
};

#endif // ESN_H_