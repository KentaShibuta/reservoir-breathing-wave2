#ifndef ESN_H_
#define ESN_H_

#include <iostream>
#include <vector>
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

#ifdef USE_PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#endif

class ESN{
    private:
        SMatrix2 m_matlib;
        void set_Wout (const std::vector<std::vector<double>>& mat);
    
    public:
        std::vector<std::vector<std::vector<float>>> vec_u;
        std::vector<std::vector<float>> vec_w_in;
        std::vector<std::vector<float>> vec_w;
        std::vector<std::vector<float>> vec_w_out;
        std::vector<float> vec_x;
        float a_alpha;
        size_t N;
        size_t N_window;
        size_t N_u;
        size_t N_x;
        size_t N_y;

        ESN();
#ifdef USE_PYBIND
        ESN(size_t n_u, size_t n_y, size_t n_x, float density, float input_scale, float rho, float leaking_rate);
        ESN(py::array_t<float> u, py::array_t<float> w_in, py::array_t<float> w, py::array_t<float> w_out, py::array_t<float> x, float alpha);
#endif

        void Print();

#ifdef USE_PYBIND
        py::array_t<float> Predict(py::array_t<float> u);
        py::array_t<float> Train(py::array_t<float> u, py::array_t<float> d);
        void SetWout(py::array_t<float> w_out);
        void SetWin(py::array_t<float> w_in);
        void SetW(py::array_t<float> w);
        py::array_t<float> GetWout();

        template <typename MatrixType, typename VectorType, typename T>
        std::unique_ptr<std::vector<std::vector<T>>> make_connection_mat(size_t N_x, T density, T rho);
#endif

#ifdef USE_PYBIND
        py::tuple GetInversePy2 (py::array_t<double> mat);
#endif
};

#endif // ESN_H_