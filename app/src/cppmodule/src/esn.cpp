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

#ifdef TEST
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN // doctestの実装部とmain関数を有効化する
#include "doctest.h"
#include <typeinfo>
#endif

class ESN{
    private:
        SMatrix m_matlib;

        void set_Wout (const std::vector<std::vector<double>>& mat){
            size_t row_size = mat.size();
            size_t col_size = mat[0].size();

            for (size_t i = 0; i < row_size; i++){
                for (size_t j = 0; j < col_size; j++){
                    vec_w_out[i][j] = static_cast<double>(mat[i][j]);
                }
            }
        }

    public:
        std::vector<std::vector<std::vector<double>>> vec_u;
        std::vector<std::vector<double>> vec_w_in;
        std::vector<std::vector<double>> vec_w;
        std::vector<std::vector<double>> vec_w_out;
        std::vector<double> vec_x;
        double a_alpha;
        size_t N;
        size_t N_window;
        size_t N_u;
        size_t N_x;
        size_t N_y;

        ESN(size_t n_u, size_t n_y, size_t n_x, float density, float input_scale, float rho, double leaking_rate){
            m_matlib = SMatrix();
            N_u = n_u;
            std::cout << "init N_u: " << N_u << std::endl;
            N_y = n_y;
            N_x = n_x;

            auto mat_w_in = m_matlib.generate_uniform_randomd(N_x, N_u, input_scale);

            auto mat_w = make_connection_matd(N_x, density, rho);
            auto mat_w_out = m_matlib.generate_normal_distributiond(N_y, N_x);

            auto x_ptr = std::make_unique<std::vector<double>>(N_x, 0.0);

            vec_w_in = *mat_w_in;
            vec_w = *mat_w;
            vec_w_out = *mat_w_out;
            vec_x = *x_ptr;
            a_alpha = leaking_rate;
        }

#ifdef USE_PYBIND
        ESN(py::array_t<double> u, py::array_t<double> w_in, py::array_t<double> w, py::array_t<double> w_out, py::array_t<double> x, double alpha){
            m_matlib = SMatrix();
            const auto &u_buf = u.request();
            const auto &u_shape = u_buf.shape;
            const auto &u_ndim = u_buf.ndim;
            N = u_shape[0];
            N_window = u_shape[1];
            N_u = u_shape[2];
            double *ptr_u = static_cast<double *>(u_buf.ptr);
            vec_u.resize(N, std::vector<std::vector<double>>(N_window, std::vector<double>(N_u)));
            if (u_ndim == 3) {
                for (size_t i = 0; i < N; i++){
                    for (size_t j = 0; j < N_window; j++){
                        for (size_t k = 0; k < N_u; k++){
                            vec_u[i][j][k] = ptr_u[i * N_window * N_u + j * N_u + k];
                        }
                    }
                }
            } else {
                std::cout << "u: shape error. ndim = " << u_ndim << std::endl;
            }
            
            const auto &w_in_buf = w_in.request();
            const auto &w_in_shape = w_in_buf.shape;
            const auto &w_in_ndim = w_in_buf.ndim;
            N_x = w_in_shape[0];
            double *ptr_w_in = static_cast<double *>(w_in_buf.ptr);
            vec_w_in.resize(N_x, std::vector<double>(N_u));
            if (w_in_ndim == 2 && (size_t)w_in_shape[1] == N_u) {
                for (size_t i = 0; i < N_x; i++){
                    for (size_t j = 0; j < N_u; j++){
                        vec_w_in[i][j] = ptr_w_in[i * N_u + j];
                    }
                }
            } else {
                std::cout << "w_in: shape error. ndim = " << w_in_ndim << ", shape[0]=" << w_in_shape[0] << ", shape[1]=" << w_in_shape[1] << std::endl;
            }

            const auto &w_buf = w.request();
            const auto &w_shape = w_buf.shape;
            const auto &w_ndim = w_buf.ndim;
            double *ptr_w = static_cast<double *>(w_buf.ptr);
            vec_w.resize(N_x, std::vector<double>(N_x));
            if (w_ndim == 2 && (size_t)w_shape[0] == N_x && (size_t)w_shape[1] == N_x) {
                for (size_t i = 0; i < N_x; i++){
                    for (size_t j = 0; j < N_x; j++){
                        vec_w[i][j] = ptr_w[i * N_x + j];
                    }
                }
            } else {
                std::cout << "w: shape error. ndim = " << w_ndim << ", shape[0]=" << w_shape[0] << ", shape[1]=" << w_shape[1] << std::endl;
            }

            const auto &w_out_buf = w_out.request();
            const auto &w_out_shape = w_out_buf.shape;
            const auto &w_out_ndim = w_out_buf.ndim;
            N_y = w_out_shape[0];
            double *ptr_w_out = static_cast<double *>(w_out_buf.ptr);
            vec_w_out.resize(N_y, std::vector<double>(N_x));
            if (w_out_ndim == 2 && (size_t)w_out_shape[1] == N_x) {
                for (size_t i = 0; i < N_y; i++){
                    for (size_t j = 0; j < N_x; j++){
                        vec_w_out[i][j] = ptr_w_out[i * N_x + j];
                    }
                }
            } else {
                std::cout << "w_out: shape error. ndim = " << w_out_ndim << ", shape[0]=" << w_out_shape[0] << ", shape[1]=" << w_out_shape[1] << std::endl;
            }

            const auto &x_buf = x.request();
            const auto &x_shape = x_buf.shape;
            const auto &x_ndim = x_buf.ndim;
            double *ptr_x = static_cast<double *>(x_buf.ptr);
            vec_x.resize(N_x);
            if (x_ndim == 1 && (size_t)x_shape[0] == N_x) {
                for (size_t i = 0; i < N_x; i++){
                    vec_x[i] = ptr_x[i];
                }
            } else {
                std::cout << "x: shape error. ndim = " << x_ndim << ", shape[0]=" << x_shape[0] << std::endl;
            }

            a_alpha = alpha;
        }
#endif

        ESN(){
            m_matlib = SMatrix();
        }

        void Print(){
            // print
            // vec_u
            /*
            std::cout << "vec_u" << std::endl;
            for (size_t i = 0; i < N; i++){
                std::cout << "i = " << i << std::endl;
                for (size_t j = 0; j < N_window; j++)
                {
                    for (size_t k = 0; k < N_u; k++)
                    {
                        std::cout << vec_u[i][j][k] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            // vec_w_in
            std::cout << "vec_w_in" << std::endl;
            for (size_t i = 0; i < N_x; i++){
                for (size_t j = 0; j < N_u; j++)
                {
                    std::cout << vec_w_in[i][j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            // vec_w
            std::cout << "vec_w" << std::endl;
            for (size_t i = 0; i < N_x; i++){
                for (size_t j = 0; j < N_x; j++)
                {
                    std::cout << vec_w[i][j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            // vec_w_out
            std::cout << "vec_w_out" << std::endl;
            for (size_t i = 0; i < N_y; i++){
                //std::cout << "i = " << i << std::endl;
                for (size_t j = 0; j < N_x; j++)
                {
                    //std::cout << "j = " << j << std::endl;
                    std::cout << vec_w_out[i][j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            // vec_x
            std::cout << "vec_x" << std::endl;
            for (size_t i = 0; i < N_x; i++){
                std::cout << vec_x[i] << " ";
            }
            std::cout << std::endl;
            */
            // a_alpha
            std::cout << "a_alpha = " << a_alpha << std::endl;

            std::cout << "N = " << N << std::endl;
            std::cout << "N_u = " << N_u << std::endl;
            std::cout << "N_x = " << N_x << std::endl;
            std::cout << "N_y = " << N_y << std::endl;
        }

#ifdef USE_PYBIND
        py::array_t<double> Predict(py::array_t<double> u){
            const auto &u_buf = u.request();
            const auto &u_shape = u_buf.shape;
            const auto &u_ndim = u_buf.ndim;
            N = u_shape[0];
            N_window = u_shape[1];
            double *ptr_u = static_cast<double *>(u_buf.ptr);
            vec_u.resize(N, std::vector<std::vector<double>>(N_window, std::vector<double>(N_u)));
            if (u_ndim == 3) {
                for (size_t i = 0; i < N; i++){
                    for (size_t j = 0; j < N_window; j++){
                        for (size_t k = 0; k < N_u; k++){
                            vec_u[i][j][k] = ptr_u[i * N_window * N_u + j * N_u + k];
                        }
                    }
                }
            } else {
                std::cout << "u: shape error. ndim = " << u_ndim << std::endl;
            }

            std::cout << "N_x: " << N_x << std::endl;
            std::cout << "N_u: " << N_u << std::endl;
            std::cout << "Win size: " << vec_w_in.size() << ", " << vec_w_in[0].size() << std::endl;

            std::cout << "Init y" << std::endl;
            py::array_t<double> y({N, N_y});
            
            std::cout << "Running Predict" << std::endl;
            size_t n = 0;
            for (const auto& input : vec_u){
                size_t step = 0;
                for (const auto& input_step : input){
                    auto x_in = m_matlib.dotd(vec_w_in, input_step);
                    auto w_dot_x = m_matlib.dotd(vec_w, vec_x);

                    // リザバー状態ベクトルの更新
                    for (size_t i = 0; i < N_x; i++){
                        vec_x[i] = (1.0 - a_alpha) * vec_x[i] + a_alpha * std::tanh((*w_dot_x)[i] + (*x_in)[i]);
                    }

                    step++;
                }

                auto y_pred = m_matlib.dotd(vec_w_out, vec_x);
                for (size_t j = 0; j < N_y; j++){
                    *y.mutable_data(n, j) = (*y_pred)[j];
                }

                n++;
            }

            std::cout << "Finish Predict" << std::endl;

            return y;
        }
#endif

#ifdef USE_PYBIND
        py::array_t<double> Train(py::array_t<double> u, py::array_t<double> d){
            auto file_logger = spdlog::basic_logger_mt("basic_logger", "./log/log_train.txt");
            spdlog::set_level(spdlog::level::debug);

            py::module_ np = py::module_::import("numpy");
            py::object np_tanh = np.attr("tanh");

            // read u
            std::cout << "start reading U" << std::endl;
            const auto &u_buf = u.request();
            const auto &u_shape = u_buf.shape;
            const auto &u_ndim = u_buf.ndim;
            N = u_shape[0];
            N_window = u_shape[1];
            //N_u = u_shape[2];
            double *ptr_u = static_cast<double *>(u_buf.ptr);
            vec_u.resize(N, std::vector<std::vector<double>>(N_window, std::vector<double>(N_u)));
            if (u_ndim == 3) {
                for (size_t i = 0; i < N; i++){
                    for (size_t j = 0; j < N_window; j++){
                        for (size_t k = 0; k < N_u; k++){
                            vec_u[i][j][k] = ptr_u[i * N_window * N_u + j * N_u + k];
                        }
                    }
                }
            } else {
                std::cout << "u: shape error. ndim = " << u_ndim << std::endl;
            }
            std::cout << "end reading U" << std::endl;

            // read d
            std::cout << "start reading D" << std::endl;
            const auto &d_buf = d.request();
            const auto &d_shape = d_buf.shape;
            const auto &d_ndim = d_buf.ndim;
            //size_t N_d = d_shape[0];
            double *ptr_d = static_cast<double *>(d_buf.ptr);
            auto vec_d = std::make_unique<std::vector<double>>(N, 0.0);
            if (d_ndim == 1 && (size_t)d_shape[0] == N) {
                for (size_t i = 0; i < N; i++){
                    (*vec_d)[i] = ptr_d[i];
                }
            } else {
                std::cout << "d: shape error. ndim = " << d_ndim << ", shape[0]=" << d_shape[0] << std::endl;
            }
            std::cout << "end reading D" << std::endl;

            /*
            std::cout << "vec_d" << std::endl;
            for (size_t i = 0; i < N; i++){
                    std::cout << (*vec_d)[i] << " ";
            }
            std::cout << std::endl;
            */

            /*
            std::cout << "vec_w_in" << std::endl;
            for (size_t i = 0; i < N_x; i++){
                for (size_t j = 0; j < N_u; j++){
                    std::cout << vec_w_in[i][j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            */

            //auto y = std::make_unique<std::vector<float>>(N, 0.0f);
            py::array_t<double> y(N);

            // 時間発展
            std::cout << "Running Train" << std::endl;
            size_t n = 0;

            // N_x行、N_x列
            auto X_XT = std::make_unique<std::vector<std::vector<double>>>(N_x, std::vector<double>(N_x, 0.0));
            auto D_XT = std::make_unique<std::vector<std::vector<double>>>(N_y, std::vector<double>(N_x, 0.0));

            for (const auto& input : vec_u){
                //std::cout << "n:" << n << std::endl;
                size_t step = 0;

                /*
                if (n == 0 || n == N-1){
                    std::cout << "input n: " << n << std::endl;
                    std::cout << "N_window: " << N_window << std::endl;
                    std::cout << "N_u: " << N_u << std::endl;
                    for (size_t i = 0; i < N_window; i++){
                        for (size_t j = 0; j < N_u; j++){
                            std::cout << input[i][j] << " ";
                        }
                    }
                    std::cout << std::endl;
                }
                */

                //std::cout << "start updating vec_x" << std::endl;
                for (const auto& input_step : input){;
                    auto x_in = m_matlib.dotd(vec_w_in, input_step);
                    auto w_dot_x = m_matlib.dotd(vec_w, vec_x);

                    // リザバー状態ベクトルの更新
                    for (size_t i = 0; i < N_x; i++){
                        vec_x[i] = (1.0 - a_alpha) * vec_x[i] + a_alpha * std::tanh((*w_dot_x)[i] + (*x_in)[i]);
                        //vec_x[i] = (1.0 - a_alpha) * vec_x[i] + a_alpha * np_tanh((*w_dot_x)[i] + (*x_in)[i]).cast<double>();
                    }

                    step++;
                }
                //std::cout << "end updating vec_x" << std::endl;

                // 目標値
                //auto d = (*vec_d)[n];
                //auto x = vec_x;

                // 学習器
                //std::cout << "start updating X_XT and D_XT" << std::endl;
                if (n > 0){
                    // optimizerの更新
                    // dとvec_x[i]を使って計算する
                    for (size_t i = 0; i < N_x; i++){
                        for (size_t j = 0; j < N_x; j++){
                            (*X_XT)[i][j] += vec_x[i] * vec_x[j];
                            if(n == 4 || n == 1497){
                                file_logger->debug("X_XT[{}][{}] = {}", i, j, (*X_XT)[i][j]);
                            }
                        }
                    }

                    //file_logger->debug("vec_d[{}] = {}", n, (*vec_d)[n]);
                    for (size_t i = 0; i < N_y; i++){
                        for (size_t j = 0; j < N_x; j++){
                            (*D_XT)[i][j] += (*vec_d)[n] * vec_x[j];
                            if(n == 4 || n == 1497){
                                file_logger->debug("D_XT[{}][{}] = {}", i, j, (*D_XT)[i][j]);
                            }
                        }
                    }
                }
                //std::cout << "end updating X_XT and D_XT" << std::endl;

                //std::cout << "start calculation y" << std::endl;
                //auto y_pred = dot(vec_w_out, vec_x);
                //std::cout << "end calculation y" << std::endl;
                /*
                *y.mutable_data(n) = (*y_pred)[n];
                */
                n++;

                //std::cout << "end n:" << n-1 << std::endl;
            }

            std::cout << "start updating Wout" << std::endl;
            // 学習済みの出力結合重み行列を設定
            // X_XTの疑似逆行列を求める
            //auto inv_X_XT = m_matlib.GetInverse(*X_XT);
            auto inv_X_XT = m_matlib.GetInversePy(*X_XT);

            size_t inv_i = 0;
            for (auto& row : *inv_X_XT){
                size_t inv_j = 0;
                for (auto& elem : row){
                    file_logger->debug("inv_X_XT[{}][{}] = {}", inv_i, inv_j, elem);
                    inv_j++;
                }
                inv_i++;
            }

            /*
            std::cout << "D_XT" << std::endl;
            for (size_t i = 0; i < N_y; i++){
                for (size_t j = 0; j < N_x; j++){
                    std::cout << (*D_XT)[i][j] << " ";
                }
                std::cout << std::endl;
            }
            */

            /*
            std::cout << "inv_X_XT" << std::endl;
            for (size_t i = 0; i < N_x; i++){
                for (size_t j = 0; j < N_x; j++){
                    std::cout << (*inv_X_XT)[i][j] << " ";
                }
                std::cout << std::endl;
            }
            */

            // D_XTとX_XTの疑似逆行列の積を計算してWoutを求める
            auto mul = m_matlib.matMuld(*D_XT, *inv_X_XT);

            std::cout << "cpp Wout" << std::endl;
            for (size_t i = 0; i < N_x; i++){
                std::cout << (*mul)[0][i] << " ";
            }
            std::cout << std::endl;

            /*
            for (auto &row : *mul){
                for (auto &elem : row){
                    std::cout << elem << " ";
                }
            }
            */
            std::cout << std::endl;

            set_Wout(*mul);
            std::cout << "end updating Wout" << std::endl;

            std::cout << "Finish Train" << std::endl;

            // yをnumpy型で返す
            return y;
        }
#endif

#ifdef USE_PYBIND
        void SetWout(py::array_t<double> w_out){
            std::cout << "Start SetWout" << std::endl;

            const auto &w_out_buf = w_out.request();
            const auto &w_out_shape = w_out_buf.shape;
            const auto &w_out_ndim = w_out_buf.ndim;
            double *ptr_w_out = static_cast<double *>(w_out_buf.ptr);
            vec_w_out.resize(N_y, std::vector<double>(N_x));
            if (w_out_ndim == 2 && (size_t)w_out_shape[1] == N_x) {
                for (size_t i = 0; i < N_y; i++){
                    for (size_t j = 0; j < N_x; j++){
                        vec_w_out[i][j] = ptr_w_out[i * N_x + j];
                    }
                }
            } else {
                std::cout << "w_out: shape error. ndim = " << w_out_ndim << ", shape[0]=" << w_out_shape[0] << ", shape[1]=" << w_out_shape[1] << std::endl;
            }
        }

        void SetWin(py::array_t<double> w_in){
            const auto &w_in_buf = w_in.request();
            const auto &w_in_shape = w_in_buf.shape;
            const auto &w_in_ndim = w_in_buf.ndim;
            double *ptr_w_in = static_cast<double *>(w_in_buf.ptr);
            vec_w_in.resize(N_x, std::vector<double>(N_u));
            if (w_in_ndim == 2 && (size_t)w_in_shape[1] == N_u) {
                for (size_t i = 0; i < N_x; i++){
                    for (size_t j = 0; j < N_u; j++){
                        vec_w_in[i][j] = ptr_w_in[i * N_u + j];
                    }
                }
            } else {
                std::cout << "w_in: shape error. ndim = " << w_in_ndim << ", shape[0]=" << w_in_shape[0] << ", shape[1]=" << w_in_shape[1] << std::endl;
            }
            
            std::cout << "End SetWout" << std::endl;
        }

        void SetW(py::array_t<double> w){
            std::cout << "Start SetW" << std::endl;

            const auto &w_buf = w.request();
            const auto &w_shape = w_buf.shape;
            const auto &w_ndim = w_buf.ndim;
            double *ptr_w = static_cast<double *>(w_buf.ptr);
            vec_w.resize(N_x, std::vector<double>(N_x));
            if (w_ndim == 2 && (size_t)w_shape[1] == N_x) {
                for (size_t i = 0; i < N_x; i++){
                    for (size_t j = 0; j < N_x; j++){
                        vec_w[i][j] = ptr_w[i * N_x + j];
                    }
                }
            } else {
                std::cout << "w: shape error. ndim = " << w_ndim << ", shape[0]=" << w_shape[0] << ", shape[1]=" << w_shape[1] << std::endl;
            }
            std::cout << "Finish SetW" << std::endl;
        }
#endif

#ifdef USE_PYBIND
        py::array_t<double> GetWout(){
            size_t row_size = vec_w_out.size();
            size_t col_size = vec_w_out[0].size();

            // NumPy配列用メモリ確保
            py::array_t<double> py_wout({row_size, col_size});

            // データへのポインタ取得
            auto buf = py_wout.request();
            double* ptr = static_cast<double*>(buf.ptr);

            // vector<vector<double>> の内容を NumPy 配列へコピー
            for (size_t i = 0; i < row_size; ++i) {
                for (size_t j = 0; j < col_size; ++j) {
                    ptr[i * col_size + j] = vec_w_out[i][j];
                }
            }

            return py_wout;
        }
#endif

        std::unique_ptr<std::vector<std::vector<float>>> make_connection_mat(size_t N_x, float density, float rho) {
            auto connection_matrix = std::make_unique<std::vector<std::vector<float>>>(N_x, std::vector<float>(N_x));
            auto w1 = m_matlib.generate_erdos_renyi(N_x, density);
            auto w2 = m_matlib.generate_uniform_random(N_x, N_x, 1.0);

            #pragma omp parallel for
            for (size_t i = 0; i < N_x; i++){
                for (size_t j = 0; j < N_x; j++){
                    (*connection_matrix)[i][j] = (*w1)[i][j] * (*w2)[i][j];
                }
            }

            // Eigen::MatrixXf に変換
            float sp_radius = 0.0;
            try {
                Eigen::MatrixXf eigenMat = m_matlib.vectorMatrixToEigenMatrix(*connection_matrix);

                Eigen::EigenSolver<Eigen::MatrixXf> solver(eigenMat);
                Eigen::VectorXcf eigenvalues = solver.eigenvalues();

                for (size_t i = 0; i < (size_t)eigenvalues.size(); ++i) {
                    float absVal = std::abs(eigenvalues[i]);  // 固有値の絶対値
                    if (absVal > sp_radius) {
                        sp_radius = absVal;
                    }
                }

            } catch (const std::exception& e) {
                std::cerr << "エラー: " << e.what() << std::endl;
            }

            #pragma omp parallel for
            for (size_t i = 0; i < N_x; i++){
                for (size_t j = 0; j < N_x; j++){
                    (*connection_matrix)[i][j] *= (rho / (1.0 * sp_radius));
                }
            }

            return connection_matrix;
        }

        std::unique_ptr<std::vector<std::vector<double>>> make_connection_matd(size_t N_x, float density, float rho) {
            auto connection_matrix = std::make_unique<std::vector<std::vector<double>>>(N_x, std::vector<double>(N_x));
            auto w1 = m_matlib.generate_erdos_renyi(N_x, density);
            auto w2 = m_matlib.generate_uniform_randomd(N_x, N_x, 1.0);

            #pragma omp parallel for
            for (size_t i = 0; i < N_x; i++){
                for (size_t j = 0; j < N_x; j++){
                    (*connection_matrix)[i][j] = (*w1)[i][j] * (*w2)[i][j];
                }
            }

            // Eigen::MatrixXf に変換
            double sp_radius = 0.0;
            try {
                Eigen::MatrixXd eigenMat = m_matlib.vectorMatrixToEigenMatrixd(*connection_matrix);

                Eigen::EigenSolver<Eigen::MatrixXd> solver(eigenMat);
                Eigen::VectorXcd eigenvalues = solver.eigenvalues();

                for (size_t i = 0; i < (size_t)eigenvalues.size(); ++i) {
                    double absVal = std::fabs(eigenvalues[i]);  // 固有値の絶対値
                    if (absVal > sp_radius) {
                        sp_radius = absVal;
                    }
                }

            } catch (const std::exception& e) {
                std::cerr << "エラー: " << e.what() << std::endl;
            }

            #pragma omp parallel for
            for (size_t i = 0; i < N_x; i++){
                for (size_t j = 0; j < N_x; j++){
                    (*connection_matrix)[i][j] *= (rho / (1.0 * sp_radius));
                }
            }

            return connection_matrix;
        }

#ifdef USE_PYBIND
        py::tuple GetInversePy2 (py::array_t<double> mat){
            SMatrix matlib = SMatrix();
            return matlib.GetInversePy2(mat);
        }
#endif
};


#ifdef TEST
TEST_CASE("[test] get inverse matrix") {
    std::cout << "[START] get inverse matrix" << std::endl;
    SMatrix matlib = SMatrix();

    std::vector<std::vector<float>> matrix = {
        {1.0f, -1.0f},
        {-1.0f, 2.0f},
        {2.0f, -1.0f}
    };

    auto inv = matlib.GetInverse(matrix);

    for (const auto &row : *inv){
        for (const auto &elem : row){
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "[PASS] get inverse matrix" << std::endl;
}

TEST_CASE("[test] matrix mul") {
    std::cout << "[START] matrix mul" << std::endl;
    SMatrix matlib = SMatrix();

    std::vector<std::vector<float>> A = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };

    std::vector<std::vector<float>> B = {
        {11.0f, 12.0f, 13.0f},
        {14.0f, 15.0f, 16.0f},
        {17.0f, 18.0f, 19.0f}
    };

    auto mul = matlib.matMul(A, B);

    for (const auto &row : *mul){
        for (const auto &elem : row){
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "[PASS] matrix mul" << std::endl;
}

TEST_CASE("[test] create init matrix") {
    std::cout << "[START] create init matrix" << std::endl;
    SMatrix matlib = SMatrix();
    ESN esn = ESN();

    size_t N_x = 500;
    size_t N_u = 60000;
    size_t N_y = 1;
    float input_scale = 1.0f;
    float density = 0.1;
    float rho = 0.9;

    auto w_in = matlib.generate_uniform_random(N_x, N_u, input_scale);
    auto w = esn.make_connection_mat(N_x, density, rho);
    auto w_out = matlib.generate_normal_distribution(N_y, N_x);

    std::cout << "[result] w_in" << std::endl;
    std::cout << "row_size: " << (*w_in).size() << std::endl;
    std::cout << "col_size: " << (*w_in)[0].size() << std::endl;
    float w_in_sum = 0;
    int outOfScaleCount = 0;
    for (const auto &row : *w_in){
        for (const auto &elem : row){
            //std::cout << elem << " ";
            w_in_sum += elem;
            //std::cout << std::endl;
            if (elem < -1.0 || elem > 1.0){
                outOfScaleCount++;
            }
        }
        //std::cout << std::endl;
    }
    float w_in_ave = w_in_sum / (1.0f * N_x * N_u);
    std::cout << "w_in average: " << w_in_ave << std::endl;

    float w_in_sum2 = 0;
    for (const auto &row : *w_in){
        for (const auto &elem : row){
            w_in_sum2 += (elem - w_in_ave) * (elem - w_in_ave);
            //std::cout << std::endl;
        }
    }
    float w_in_variance = w_in_sum2 / (1.0f * N_x * N_u);
    std::cout << "w_in_variance: " << w_in_variance << std::endl;
    std::cout << "outOfScaleCount: " << outOfScaleCount << std::endl;

    std::cout << std::endl;

    std::cout << "[result] w" << std::endl;
    std::cout << "row_size: " << (*w).size() << std::endl;
    std::cout << "col_size: " << (*w)[0].size() << std::endl;
    for (const auto &row : *w){
        for (const auto &elem : row){
            //std::cout << elem << " ";
        }
        //std::cout << std::endl;
    }

    std::cout << std::endl;

    std::cout << "[result] w_out" << std::endl;
    std::cout << "row_size: " << (*w_out).size() << std::endl;
    std::cout << "col_size: " << (*w_out)[0].size() << std::endl;
    for (const auto &row : *w_out){
        for (const auto &elem : row){
            //std::cout << elem << " ";
        }
        //std::cout << std::endl;
    }

    std::cout << "[PASS] create init matrix" << std::endl;
}

TEST_CASE("[test] inv matrix") {
    std::cout << "[START] inv matrix" << std::endl;
    SMatrix matlib = SMatrix();
    auto file_logger = spdlog::basic_logger_mt("test_logger", "./log/test_log.txt");
    spdlog::set_level(spdlog::level::debug);

    size_t N_x = 500;
    auto X_XT = std::make_unique<std::vector<std::vector<float>>>(N_x, std::vector<float>(N_x, 1497.0f));

    auto inv_X_XT = matlib.GetInverse(*X_XT);

    size_t inv_i = 0;
    for (auto& row : *inv_X_XT){
        size_t inv_j = 0;
        for (auto& elem : row){
            file_logger->debug("inv_X_XT[{}][{}] = {}", inv_i, inv_j, elem);
            inv_j++;
        }
        inv_i++;
    }

    auto r1 = matlib.matMul(*inv_X_XT, *X_XT);
    auto result1 = matlib.matMul(*X_XT, *r1);

    inv_i = 0;
    for (auto& row : *result1){
        size_t inv_j = 0;
        for (auto& elem : row){
            file_logger->debug("result1[{}][{}] = {}", inv_i, inv_j, elem);
            inv_j++;
        }
        inv_i++;
    }

    std::cout << "[PASS] inv matrix" << std::endl;
}

TEST_CASE("[test] SMatrix2") {
    std::cout << "[START] SMatrix2" << std::endl;
    SMatrix2 matlib2 = SMatrix2();

    ///////////////
    // test data //
    ///////////////
    size_t row_size = 10;
    size_t col_size = 10;
    double scale_d = 1.0;
    float scale_f = 1.0f;
    double mean_d = 0.0;
    float mean_f = 0.0f;
    double stddev_d = 1.0;
    float stddev_f = 1.0f;

    std::vector<std::vector<double>> A_d = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    std::vector<std::vector<double>> B_d = {
        {11.0, 12.0, 13.0},
        {14.0, 15.0, 16.0},
        {17.0, 18.0, 19.0}
    };
    std::vector<std::vector<float>> A_f = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };
    std::vector<std::vector<float>> B_f = {
        {11.0f, 12.0f, 13.0f},
        {14.0f, 15.0f, 16.0f},
        {17.0f, 18.0f, 19.0f}
    };

    size_t N_x = 20;
    double density_d = 0.6;
    float density_f = 0.6f;

    std::vector<double> vec_d = {1.0, -2.0, 3.0};
    std::vector<float> vec_f = {1.0f, -2.0f, 3.0f};

    std::cout << "generate_uniform_random" << std::endl;
    auto uniform_rand_mat_d = matlib2.generate_uniform_random(row_size, col_size, scale_d);
    std::cout << typeid((*uniform_rand_mat_d)[0][0]).name() << std::endl;
    CHECK(std::string(typeid((*uniform_rand_mat_d)[0][0]).name()) == "d");
    auto uniform_rand_mat_f = matlib2.generate_uniform_random(row_size, col_size, scale_f);
    std::cout << typeid((*uniform_rand_mat_f)[0][0]).name() << std::endl;
    CHECK(std::string(typeid((*uniform_rand_mat_f)[0][0]).name()) == "f");

    std::cout << "generate_normal_distribution" << std::endl;
    auto normal_rand_mat_d = matlib2.generate_normal_distribution(row_size, col_size, mean_d, stddev_d);
    std::cout << typeid((*normal_rand_mat_d)[0][0]).name() << std::endl;
    CHECK(std::string(typeid((*normal_rand_mat_d)[0][0]).name()) == "d");
    auto normal_rand_mat_f = matlib2.generate_normal_distribution(row_size, col_size, mean_f, stddev_f);
    std::cout << typeid((*normal_rand_mat_f)[0][0]).name() << std::endl;
    CHECK(std::string(typeid((*normal_rand_mat_f)[0][0]).name()) == "f");

    std::cout << "vectorMatrixToEigenMatrix" << std::endl;
    auto eigen_mat_d = matlib2.vectorMatrixToEigenMatrix<Eigen::MatrixXd, double>(A_d);
    std::cout << typeid(eigen_mat_d(0, 0)).name() << std::endl;
    CHECK(std::string(typeid(eigen_mat_d(0, 0)).name()) == "d");
    auto eigen_mat_f = matlib2.vectorMatrixToEigenMatrix<Eigen::MatrixXf, float>(A_f);
    std::cout << typeid(eigen_mat_f(0, 0)).name() << std::endl;
    CHECK(std::string(typeid(eigen_mat_f(0, 0)).name()) == "f");

    std::cout << "eigenMatrixToUniquePtr" << std::endl;
    auto unique_ptr_d = matlib2.eigenMatrixToUniquePtr<Eigen::MatrixXd, double>(eigen_mat_d);
    std::cout << typeid((*unique_ptr_d)[0][0]).name() << std::endl;
    CHECK(std::string(typeid((*unique_ptr_d)[0][0]).name()) == "d");
    auto unique_ptr_f = matlib2.eigenMatrixToUniquePtr<Eigen::MatrixXf, float>(eigen_mat_f);
    std::cout << typeid((*unique_ptr_f)[0][0]).name() << std::endl;
    CHECK(std::string(typeid((*unique_ptr_f)[0][0]).name()) == "f");

    std::cout << "generate_erdos_renyi" << std::endl;
    auto erdos_renyi_d = matlib2.generate_erdos_renyi(N_x, density_d);
    std::cout << typeid((*erdos_renyi_d)[0][0]).name() << std::endl;
    CHECK(std::string(typeid((*erdos_renyi_d)[0][0]).name()) == "h");
    auto erdos_renyi_f = matlib2.generate_erdos_renyi(N_x, density_f);
    std::cout << typeid((*erdos_renyi_f)[0][0]).name() << std::endl;
    CHECK(std::string(typeid((*erdos_renyi_f)[0][0]).name()) == "h");

    std::cout << "dot" << std::endl;
    auto dot_d = matlib2.dot(A_d, vec_d);
    std::cout << typeid((*dot_d)[0]).name() << std::endl;
    CHECK(std::string(typeid((*dot_d)[0]).name()) == "d");
    auto dot_f = matlib2.dot(A_f, vec_f);
    std::cout << typeid((*dot_f)[0]).name() << std::endl;
    CHECK(std::string(typeid((*dot_f)[0]).name()) == "f");

    std::cout << "matMul" << std::endl;
    auto matMul_d = matlib2.matMul(A_d, B_d);
    std::cout << typeid((*matMul_d)[0][0]).name() << std::endl;
    CHECK(std::string(typeid((*matMul_d)[0][0]).name()) == "d");
    auto matMul_f = matlib2.matMul(A_f, B_f);
    std::cout << typeid((*matMul_f)[0][0]).name() << std::endl;
    CHECK(std::string(typeid((*matMul_f)[0][0]).name()) == "f");


    std::cout << "[PASS] SMatrix2" << std::endl;
}

#endif

#ifdef USE_PYBIND
PYBIND11_MODULE(esn, m){
    py::class_<ESN>(m, "ESN", "ESN class made by pybind11")
        .def(py::init<py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, double>())
        .def(py::init<size_t, size_t, size_t, float, float, float, double>())
        .def(py::init())
        .def("SetWout", &ESN::SetWout)
        .def("SetWin", &ESN::SetWin)
        .def("SetW", &ESN::SetW)
        .def("Print", &ESN::Print)
        .def("Predict", &ESN::Predict)
        .def("Train", &ESN::Train)
        //.def("Randnumer_test", &ESN::Randnumer_test)
        //.def("Generate_erdos_renyi_test", &ESN::Generate_erdos_renyi_test)
        .def("GetWout", &ESN::GetWout)
        .def("GetInversePy2", &ESN::GetInversePy2);
}
#endif