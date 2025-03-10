#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <omp.h>
#include <random>
#include <eigen3/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

class ESN{
    private:
        std::unique_ptr<std::vector<std::vector<float>>> generate_uniform_random(std::size_t row_size, std::size_t col_size, float scale) {
            uint_fast32_t seed = 0;
            std::mt19937 gen(seed); // メルセンヌ・ツイスター法による生成器
            std::uniform_real_distribution<float> dist(-1.0 * scale, std::nextafter(scale, std::numeric_limits<float>::max()));
        
            auto numbers = std::make_unique<std::vector<std::vector<float>>>(row_size, std::vector<float>(col_size));
            for (std::size_t i = 0; i < row_size; i++) {
                for (std::size_t j = 0; j < col_size; j++) {
                    (*numbers)[i][j] = dist(gen);
                }
            }
            return numbers;
        }

        std::unique_ptr<std::vector<std::vector<float>>> generate_normal_distribution(std::size_t row_size, std::size_t col_size, float mean = 0.0, float stddev = 1.0) {
            // 乱数生成器と正規分布設定
            uint_fast32_t seed = 0;
            std::mt19937 gen(seed);
            std::normal_distribution<float> dist(mean, stddev);
        
            // N_y行N_x列の2次元配列を作成
            auto numbers = std::make_unique<std::vector<std::vector<float>>>(row_size, std::vector<float>(col_size));
            // 乱数を生成して配列に格納
            for (std::size_t i = 0; i < row_size; i++) {
                for (std::size_t j = 0; j < col_size; j++) {
                    (*numbers)[i][j] = dist(gen);
                }
            }
            return numbers;
        }

        // std::unique_ptr<std::vector<std::vector<float>>> を Eigen::MatrixXf に変換
        Eigen::MatrixXf uniquePtrToEigenMatrix(const std::unique_ptr<std::vector<std::vector<float>>>& ptr) {
            if (!ptr || ptr->empty()) {
                throw std::invalid_argument("入力が空です。");
            }

            int rows = ptr->size();          // 行数
            int cols = ptr->at(0).size();    // 列数

            // 列数が揃っているか確認
            for (const auto& row : *ptr) {
                if (row.size() != static_cast<size_t>(cols)) {
                    throw std::invalid_argument("全ての行の列数が一致しません。");
                }
            }

            Eigen::MatrixXf eigenMat(rows, cols);
            
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    eigenMat(i, j) = (*ptr)[i][j];
                }
            }

            return eigenMat;
        }

        // ランダムなErdos-Renyiグラフを生成
        std::unique_ptr<std::vector<std::vector<uint8_t>>> generate_erdos_renyi(size_t N_x, float density) {
            uint_fast32_t seed = 0;
            std::mt19937 gen(seed);  // Mersenne Twister 乱数生成器
            std::uniform_int_distribution<size_t> dist(0, N_x - 1);  // 範囲 [0, N_x-1] の一様分布

            auto adjacency_matrix = std::make_unique<std::vector<std::vector<uint8_t>>>(N_x, std::vector<uint8_t>(N_x, 0));
            size_t m = static_cast<size_t>(N_x * (N_x - 1) * density / 2.0);  // 総結合数

            //std::cout << "m: " << m << std::endl;

            size_t edge_count = 0;
            while (edge_count < m) {
                size_t i = dist(gen);
                size_t j = dist(gen);
                
                // 自己ループを避け、重複エッジを作らない
                if (i != j && (*adjacency_matrix)[i][j] == 0) {
                    (*adjacency_matrix)[i][j] = 1;
                    (*adjacency_matrix)[j][i] = 1;  // 無向グラフ
                    edge_count++;
                }
            }

            return adjacency_matrix;
        }

        std::unique_ptr<std::vector<std::vector<float>>> make_connection_mat(size_t N_x, float density, float rho) {
            auto connection_matrix = std::make_unique<std::vector<std::vector<float>>>(N_x, std::vector<float>(N_x));
            auto w1 = generate_erdos_renyi(N_x, density);
            auto w2 = generate_uniform_random(N_x, N_x, 1.0);

            // print
            //////////////////////
            /*
            std::cout << "w1" << std::endl;
            for (size_t i = 0; i < N_x; i++){
                for (size_t j = 0; j < N_x; j++){
                    std::cout << (int)((*w1)[i][j]) << " ";
                }
                std::cout << std::endl;
            }

            std::cout << "w2" << std::endl;
            for (size_t i = 0; i < N_x; i++){
                for (size_t j = 0; j < N_x; j++){
                    std::cout << (*w2)[i][j] << " ";
                }
                std::cout << std::endl;
            }
            */

            //////////////////////

            #pragma omp parallel for
            for (size_t i = 0; i < N_x; i++){
                for (size_t j = 0; j < N_x; j++){
                    (*connection_matrix)[i][j] = (*w1)[i][j] * (*w2)[i][j];
                }
            }

            // Eigen::MatrixXf に変換
            float sp_radius = 0.0;
            try {
                Eigen::MatrixXf eigenMat = uniquePtrToEigenMatrix(connection_matrix);
                //std::cout << "Eigen::MatrixXf:\n" << eigenMat << std::endl;

                
                Eigen::EigenSolver<Eigen::MatrixXf> solver(eigenMat);
                Eigen::VectorXcf eigenvalues = solver.eigenvalues();

                for (size_t i = 0; i < (size_t)eigenvalues.size(); ++i) {
                    //std::cout << "seigenvalues[" << i << "]: " << eigenvalues[i] << std::endl;
                    //std::cout << "abs(seigenvalues[" << i << "]): " << std::abs(eigenvalues[i]) << std::endl;
                    float absVal = std::abs(eigenvalues[i]);  // 固有値の絶対値
                    if (absVal > sp_radius) {
                        sp_radius = absVal;
                    }
                }
                

            } catch (const std::exception& e) {
                std::cerr << "エラー: " << e.what() << std::endl;
            }

            //std::cout << "sp_radius: " << sp_radius << std::endl;

            #pragma omp parallel for
            for (size_t i = 0; i < N_x; i++){
                for (size_t j = 0; j < N_x; j++){
                    (*connection_matrix)[i][j] *= (rho / (1.0 * sp_radius));
                }
            }

            // print
            ////////////////////
            /*
            std::cout << "connection_matrix" << std::endl;
            for (size_t i = 0; i < N_x; i++){
                for (size_t j = 0; j < N_x; j++){
                    std::cout << (*connection_matrix)[i][j] << " ";
                }
                std::cout << std::endl;
            }
            */
            ////////////////////

            return connection_matrix;
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
            N_u = n_u;
            N_y = n_y;
            N_x = n_x;

            
            auto mat_w_in = generate_uniform_random(N_x, N_u, input_scale);
            /*
            std::cout << "mat_w_in: " << std::endl; 
            for (size_t i = 0; i < N_x; i++){
                for (size_t j = 0; j < N_u; j++){
                    std::cout << (*mat_w_in)[i][j] << " ";
                }
                std::cout << std::endl;
            }
            */

            auto mat_w = make_connection_mat(N_x, density, rho);
            auto mat_w_out = generate_normal_distribution(N_y, N_x);

            auto x_ptr = std::make_unique<std::vector<float>>(N_x, 0.0f);

            /*
            std::cout << "mat_w_out: " << std::endl; 
            for (size_t i = 0; i < N_y; i++){
                for (size_t j = 0; j < N_x; j++){
                    std::cout << (*mat_w_out)[i][j] << " ";
                }
                std::cout << std::endl;
            }
            */

            vec_w_in = *mat_w_in;
            vec_w = *mat_w;
            vec_w_out = *mat_w_out;
            vec_x = *x_ptr;
            a_alpha = leaking_rate;
        }

        void SetWout(py::array_t<float> w_out, py::array_t<float> w_in){
        //void SetInput(py::array_t<float> u, py::array_t<float> w_out){
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

        ESN(py::array_t<float> u, py::array_t<float> w_in, py::array_t<float> w, py::array_t<float> w_out, py::array_t<float> x, float alpha){
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

        void Print(){
            // print
            // vec_u
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
            // a_alpha
            std::cout << "a_alpha = " << a_alpha << std::endl;

            std::cout << "N = " << N << std::endl;
            std::cout << "N_u = " << N_u << std::endl;
            std::cout << "N_x = " << N_x << std::endl;
            std::cout << "N_y = " << N_y << std::endl;
        }

        py::array_t<float> Predict(py::array_t<float> u){
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
                    auto x_in = dot(vec_w_in, input_step);
                    auto w_dot_x = dot(vec_w, vec_x);

                    // リザバー状態ベクトルの更新
                    for (size_t i = 0; i < N_x; i++){
                        vec_x[i] = (1.0 - a_alpha) * vec_x[i] + a_alpha * std::tanh((*w_dot_x)[i] + (*x_in)[i]);
                    }

                    step++;
                }

                auto y_pred = dot(vec_w_out, vec_x);
                for (size_t j = 0; j < N_y; j++){
                    *y.mutable_data(n, j) = (*y_pred)[j];
                }

                n++;
            }

            std::cout << "Finish Predict" << std::endl;

            return y;
        }

        py::array_t<float> Train(py::array_t<float> u, py::array_t<float> d){
            // read u
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

            // read d
            const auto &d_buf = d.request();
            const auto &d_shape = d_buf.shape;
            const auto &d_ndim = d_buf.ndim;
            N_d = d_shape[0];
            float *ptr_d = static_cast<float *>(d_buf.ptr);
            auto vec_d = std::make_unique<std::vector<float>>(N_d, 0.0f);
            if (d_ndim == 1 && (size_t)d_shape[0] == N_d) {
                for (size_t i = 0; i < N_d; i++){
                    vec_d[i] = ptr_d[i];
                }
            } else {
                std::cout << "d: shape error. ndim = " << d_ndim << ", shape[0]=" << d_shape[0] << std::endl;
            }

            auto y = std::make_unique<std::vector<float>>(N, 0.0f);

            // 時間発展
            std::cout << "Running Train" << std::endl;
            size_t n = 0;
            for (const auto& input : vec_u){
                size_t step = 0;
                for (const auto& input_step : input){
                    auto x_in = dot(vec_w_in, input_step);
                    auto w_dot_x = dot(vec_w, vec_x);

                    // リザバー状態ベクトルの更新
                    for (size_t i = 0; i < N_x; i++){
                        vec_x[i] = (1.0 - a_alpha) * vec_x[i] + a_alpha * std::tanh((*w_dot_x)[i] + (*x_in)[i]);
                    }

                    step++;
                }

                // 目標値
                float d = vec_d[n];

                // 学習器
                if (n > 0){

                }

                auto y_pred = dot(vec_w_out, vec_x);
                for (size_t j = 0; j < N_y; j++){
                    *y.mutable_data(n, j) = (*y_pred)[j];
                }

                n++;
            }

            std::cout << "Finish Train" << std::endl;
        }
    
        std::unique_ptr<std::vector<float>> dot (const std::vector<std::vector<float>> &mat, const std::vector<float> &vec){
            size_t vec_size = vec.size();
            auto y = std::make_unique<std::vector<float>>(vec_size, 0.0f);

            #pragma omp parallel for
            for (size_t i = 0; i < mat.size(); i++) {
                for (size_t j = 0; j < vec_size; j++) {
                    (*y)[i] += mat[i][j] * vec[j];
                }
            }

            return y;
        }

        void Randnumer_test (size_t row_size, size_t col_size, float scale){
            auto numbers = generate_uniform_random(row_size, col_size, scale);

            for (const auto &row : *numbers){
                for (const auto &elem : row){
                    std::cout << elem << " ";
                }
                std::cout << std::endl;
            }
        }

        void Generate_erdos_renyi_test (size_t N_x, float density){
            auto numbers = generate_erdos_renyi(N_x, density);

            for (const auto &row : *numbers){
                for (const auto &elem : row){
                    std::cout << static_cast<int>(elem) << " ";
                }
                std::cout << std::endl;
            }
        }
};

PYBIND11_MODULE(esn, m){
    py::class_<ESN>(m, "ESN", "ESN class made by pybind11")
        .def(py::init<py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, float>())
        .def(py::init<size_t, size_t, size_t, float, float, float, float>())
        .def("SetWout", &ESN::SetWout)
        .def("Print", &ESN::Print)
        .def("Predict", &ESN::Predict)
        .def("Randnumer_test", &ESN::Randnumer_test)
        .def("Generate_erdos_renyi_test", &ESN::Generate_erdos_renyi_test);
}