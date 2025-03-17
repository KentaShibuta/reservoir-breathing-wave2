#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <omp.h>
#include <random>
#include <Dense> // Eigen
#include "SMatrix.hpp"

#ifdef USE_PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#endif

#ifdef TEST
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN // doctestの実装部とmain関数を有効化する
#include "doctest.h"
#endif

class ESN{
    private:
        SMatrix m_matlib;

        void set_Wout (const std::vector<std::vector<float>>& mat){
            size_t row_size = mat.size();
            size_t col_size = mat[0].size();

            for (size_t i = 0; i < row_size; i++){
                for (size_t j = 0; j < col_size; j++){
                    vec_w_out[i][j] = mat[i][j];
                }
            }
        }

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

        ESN(size_t n_u, size_t n_y, size_t n_x, float density, float input_scale, float rho, float leaking_rate){
            m_matlib = SMatrix();
            N_u = n_u;
            std::cout << "init N_u: " << N_u << std::endl;
            N_y = n_y;
            N_x = n_x;

            auto mat_w_in = m_matlib.generate_uniform_random(N_x, N_u, input_scale);

            auto mat_w = make_connection_mat(N_x, density, rho);
            auto mat_w_out = m_matlib.generate_normal_distribution(N_y, N_x);

            auto x_ptr = std::make_unique<std::vector<float>>(N_x, 0.0f);

            vec_w_in = *mat_w_in;
            vec_w = *mat_w;
            vec_w_out = *mat_w_out;
            vec_x = *x_ptr;
            a_alpha = leaking_rate;
        }

#ifdef USE_PYBIND
        ESN(py::array_t<float> u, py::array_t<float> w_in, py::array_t<float> w, py::array_t<float> w_out, py::array_t<float> x, float alpha){
            m_matlib = SMatrix();
            const auto &u_buf = u.request();
            const auto &u_shape = u_buf.shape;
            const auto &u_ndim = u_buf.ndim;
            N = u_shape[0];
            N_window = u_shape[1];
            N_u = u_shape[2];
            float *ptr_u = static_cast<float *>(u_buf.ptr);
            vec_u.resize(N, std::vector<std::vector<float>>(N_window, std::vector<float>(N_u)));
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
            float *ptr_w_in = static_cast<float *>(w_in_buf.ptr);
            vec_w_in.resize(N_x, std::vector<float>(N_u));
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
            float *ptr_w = static_cast<float *>(w_buf.ptr);
            vec_w.resize(N_x, std::vector<float>(N_x));
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
            float *ptr_w_out = static_cast<float *>(w_out_buf.ptr);
            vec_w_out.resize(N_y, std::vector<float>(N_x));
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
            float *ptr_x = static_cast<float *>(x_buf.ptr);
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
        py::array_t<float> Predict(py::array_t<float> u){
            const auto &u_buf = u.request();
            const auto &u_shape = u_buf.shape;
            const auto &u_ndim = u_buf.ndim;
            N = u_shape[0];
            N_window = u_shape[1];
            float *ptr_u = static_cast<float *>(u_buf.ptr);
            vec_u.resize(N, std::vector<std::vector<float>>(N_window, std::vector<float>(N_u)));
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
            py::array_t<float> y({N, N_y});
            
            std::cout << "Running Predict" << std::endl;
            size_t n = 0;
            for (const auto& input : vec_u){
                size_t step = 0;
                for (const auto& input_step : input){
                    auto x_in = m_matlib.dot(vec_w_in, input_step);
                    auto w_dot_x = m_matlib.dot(vec_w, vec_x);

                    // リザバー状態ベクトルの更新
                    for (size_t i = 0; i < N_x; i++){
                        vec_x[i] = (1.0 - a_alpha) * vec_x[i] + a_alpha * std::tanh((*w_dot_x)[i] + (*x_in)[i]);
                    }

                    step++;
                }

                auto y_pred = m_matlib.dot(vec_w_out, vec_x);
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
        py::array_t<float> Train(py::array_t<float> u, py::array_t<float> d){
            // read u
            std::cout << "start reading U" << std::endl;
            const auto &u_buf = u.request();
            const auto &u_shape = u_buf.shape;
            const auto &u_ndim = u_buf.ndim;
            N = u_shape[0];
            N_window = u_shape[1];
            //N_u = u_shape[2];
            float *ptr_u = static_cast<float *>(u_buf.ptr);
            vec_u.resize(N, std::vector<std::vector<float>>(N_window, std::vector<float>(N_u)));
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
            float *ptr_d = static_cast<float *>(d_buf.ptr);
            auto vec_d = std::make_unique<std::vector<float>>(N, 0.0f);
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
            py::array_t<float> y(N);

            // 時間発展
            std::cout << "Running Train" << std::endl;
            size_t n = 0;

            // N_x行、N_x列
            auto X_XT = std::make_unique<std::vector<std::vector<float>>>(N_x, std::vector<float>(N_x, 0.0f));
            auto D_XT = std::make_unique<std::vector<std::vector<float>>>(N_y, std::vector<float>(N_x, 0.0f));

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
                    auto x_in = m_matlib.dot(vec_w_in, input_step);
                    auto w_dot_x = m_matlib.dot(vec_w, vec_x);

                    // リザバー状態ベクトルの更新
                    for (size_t i = 0; i < N_x; i++){
                        vec_x[i] = (1.0 - a_alpha) * vec_x[i] + a_alpha * std::tanh((*w_dot_x)[i] + (*x_in)[i]);
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
                        }
                    }

                    for (size_t i = 0; i < N_y; i++){
                        for (size_t j = 0; j < N_x; j++){
                            (*D_XT)[i][j] += (*vec_d)[n] * vec_x[j];
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
            auto inv_X_XT = m_matlib.GetInverse(*X_XT);

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
            auto mul = m_matlib.matMul(*D_XT, *inv_X_XT);

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
        void SetWout(py::array_t<float> w_out){
            std::cout << "Start SetWout" << std::endl;

            const auto &w_out_buf = w_out.request();
            const auto &w_out_shape = w_out_buf.shape;
            const auto &w_out_ndim = w_out_buf.ndim;
            float *ptr_w_out = static_cast<float *>(w_out_buf.ptr);
            vec_w_out.resize(N_y, std::vector<float>(N_x));
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

        void SetWin(py::array_t<float> w_in){
            const auto &w_in_buf = w_in.request();
            const auto &w_in_shape = w_in_buf.shape;
            const auto &w_in_ndim = w_in_buf.ndim;
            float *ptr_w_in = static_cast<float *>(w_in_buf.ptr);
            vec_w_in.resize(N_x, std::vector<float>(N_u));
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

        void SetW(py::array_t<float> w){
            std::cout << "Start SetW" << std::endl;

            const auto &w_buf = w.request();
            const auto &w_shape = w_buf.shape;
            const auto &w_ndim = w_buf.ndim;
            float *ptr_w = static_cast<float *>(w_buf.ptr);
            vec_w.resize(N_x, std::vector<float>(N_x));
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
        py::array_t<float> GetWout(){
            size_t row_size = vec_w_out.size();
            size_t col_size = vec_w_out[0].size();

            // NumPy配列用メモリ確保
            py::array_t<float> py_wout({row_size, col_size});

            // データへのポインタ取得
            auto buf = py_wout.request();
            float* ptr = static_cast<float*>(buf.ptr);

            // vector<vector<float>> の内容を NumPy 配列へコピー
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

    size_t N_x = 10;
    size_t N_u = 15;
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
    for (const auto &row : *w_in){
        for (const auto &elem : row){
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "[result] w" << std::endl;
    std::cout << "row_size: " << (*w).size() << std::endl;
    std::cout << "col_size: " << (*w)[0].size() << std::endl;
    for (const auto &row : *w){
        for (const auto &elem : row){
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "[result] w_out" << std::endl;
    std::cout << "row_size: " << (*w_out).size() << std::endl;
    std::cout << "col_size: " << (*w_out)[0].size() << std::endl;
    for (const auto &row : *w_out){
        for (const auto &elem : row){
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "[PASS] create init matrix" << std::endl;
}
#endif

#ifdef USE_PYBIND
PYBIND11_MODULE(esn, m){
    py::class_<ESN>(m, "ESN", "ESN class made by pybind11")
        .def(py::init<py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, float>())
        .def(py::init<size_t, size_t, size_t, float, float, float, float>())
        .def("SetWout", &ESN::SetWout)
        .def("SetWin", &ESN::SetWin)
        .def("SetW", &ESN::SetW)
        .def("Print", &ESN::Print)
        .def("Predict", &ESN::Predict)
        .def("Train", &ESN::Train)
        //.def("Randnumer_test", &ESN::Randnumer_test)
        //.def("Generate_erdos_renyi_test", &ESN::Generate_erdos_renyi_test)
        .def("GetWout", &ESN::GetWout);
}
#endif