#include "SMatrix2.hpp"

std::shared_ptr<spdlog::logger> SMatrix2::logger = nullptr;

template <typename T>
std::unique_ptr<std::vector<std::vector<T>>> SMatrix2::generate_uniform_random(std::size_t row_size, std::size_t col_size, T scale) {
    uint_fast32_t seed = 0;
    std::mt19937 gen(seed); // メルセンヌ・ツイスター法による生成器
    std::uniform_real_distribution<T> dist(-1.0 * scale, std::nextafter(scale, std::numeric_limits<T>::max()));

    auto numbers = std::make_unique<std::vector<std::vector<T>>>(row_size, std::vector<T>(col_size));
    for (std::size_t i = 0; i < row_size; i++) {
        for (std::size_t j = 0; j < col_size; j++) {
            (*numbers)[i][j] = dist(gen);
        }
    }
    return numbers;
}
template std::unique_ptr<std::vector<std::vector<double>>> SMatrix2::generate_uniform_random<double> (size_t, size_t, double);
template std::unique_ptr<std::vector<std::vector<float>>> SMatrix2::generate_uniform_random<float> (size_t, size_t, float);

template <typename T>
std::unique_ptr<std::vector<std::vector<T>>> SMatrix2::generate_normal_distribution(std::size_t row_size, std::size_t col_size, T mean, T stddev) {
    // 乱数生成器と正規分布設定
    uint_fast32_t seed = 0;
    std::mt19937 gen(seed);
    std::normal_distribution<T> dist(mean, stddev);

    // N_y行N_x列の2次元配列を作成
    auto numbers = std::make_unique<std::vector<std::vector<T>>>(row_size, std::vector<T>(col_size));
    // 乱数を生成して配列に格納
    for (std::size_t i = 0; i < row_size; i++) {
        for (std::size_t j = 0; j < col_size; j++) {
            (*numbers)[i][j] = dist(gen);
        }
    }
    return numbers;
}
template std::unique_ptr<std::vector<std::vector<double>>> SMatrix2::generate_normal_distribution<double> (size_t, size_t, double, double);
template std::unique_ptr<std::vector<std::vector<float>>> SMatrix2::generate_normal_distribution<float> (size_t, size_t, float, float);

// std::unique_ptr<std::vector<std::vector<float>>> を Eigen::MatrixXf に変換
template <typename MatrixType, typename T>
MatrixType SMatrix2::vectorMatrixToEigenMatrix(const std::vector<std::vector<T>>& vectorMatrix) {
    if (vectorMatrix.empty()) {
        throw std::invalid_argument("入力が空です。");
    }

    int rows = vectorMatrix.size();          // 行数
    int cols = vectorMatrix[0].size();    // 列数

    // 列数が揃っているか確認
    for (const auto& row : vectorMatrix) {
        if (row.size() != static_cast<size_t>(cols)) {
            throw std::invalid_argument("全ての行の列数が一致しません。");
        }
    }

    MatrixType eigenMat(rows, cols);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            eigenMat(i, j) = vectorMatrix[i][j];
        }
    }

    return eigenMat;
}
template Eigen::MatrixXd SMatrix2::vectorMatrixToEigenMatrix<Eigen::MatrixXd, double> (const std::vector<std::vector<double>>&);
template Eigen::MatrixXf SMatrix2::vectorMatrixToEigenMatrix<Eigen::MatrixXf, float> (const std::vector<std::vector<float>>&);


// Eigen::MatrixXf を std::unique_ptr<std::vector<std::vector<float>>>  に変換
template <typename MatrixType, typename T>
std::unique_ptr<std::vector<std::vector<T>>> SMatrix2::eigenMatrixToUniquePtr(const MatrixType& matrix) {
    // 行列のサイズを取得
    int rows = matrix.rows();
    int cols = matrix.cols();

    // std::unique_ptrで2次元ベクトルを作成
    auto vec = std::make_unique<std::vector<std::vector<T>>>(rows, std::vector<T>(cols));

    // EigenのMatrixXfからstd::vectorにデータをコピー
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            (*vec)[i][j] = matrix(i, j);
        }
    }

    return vec;
}
template std::unique_ptr<std::vector<std::vector<double>>> SMatrix2::eigenMatrixToUniquePtr<Eigen::MatrixXd, double> (const Eigen::MatrixXd&);
template std::unique_ptr<std::vector<std::vector<float>>> SMatrix2::eigenMatrixToUniquePtr<Eigen::MatrixXf, float> (const Eigen::MatrixXf&);

// ランダムなErdos-Renyiグラフを生成
template <typename T>
std::unique_ptr<std::vector<std::vector<uint8_t>>> SMatrix2::generate_erdos_renyi(size_t N_x, T density) {
    uint_fast32_t seed = 0;
    std::mt19937 gen(seed);  // Mersenne Twister 乱数生成器
    std::uniform_int_distribution<size_t> dist(0, N_x - 1);  // 範囲 [0, N_x-1] の一様分布

    auto adjacency_matrix = std::make_unique<std::vector<std::vector<uint8_t>>>(N_x, std::vector<uint8_t>(N_x, 0));
    size_t m = static_cast<size_t>(N_x * (N_x - 1) * density / 2.0);  // 総結合数

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
template std::unique_ptr<std::vector<std::vector<uint8_t>>> SMatrix2::generate_erdos_renyi<double> (size_t, double);
template std::unique_ptr<std::vector<std::vector<uint8_t>>> SMatrix2::generate_erdos_renyi<float> (size_t, float);

template <typename T>
std::unique_ptr<std::vector<T>> SMatrix2::dot (const std::vector<std::vector<T>> &mat, const std::vector<T> &vec){
    size_t vec_size = vec.size();
    auto y = std::make_unique<std::vector<T>>(vec_size, 0.0);

    #pragma omp parallel for
    for (size_t i = 0; i < mat.size(); i++) {
        for (size_t j = 0; j < vec_size; j++) {
            (*y)[i] += mat[i][j] * vec[j];
        }
    }

    return y;
}
template std::unique_ptr<std::vector<double>> SMatrix2::dot<double> (const std::vector<std::vector<double>>&, const std::vector<double>&);
template std::unique_ptr<std::vector<float>> SMatrix2::dot<float> (const std::vector<std::vector<float>>&, const std::vector<float>&);

// 行列Aと行列Bの積
template <typename T>
std::unique_ptr<std::vector<std::vector<T>>> SMatrix2::matMul (const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B){
    size_t m = A.size();    // Aの行数
    size_t n = A[0].size(); // Aの列数 (Bの行数と同じ)
    size_t p = B[0].size(); // Bの列数

    std::cout << "A: (" << m << ", " << n << ")" << std::endl;
    std::cout << "B: (" << n << ", " << p << ")" << std::endl;

    auto mul = std::make_unique<std::vector<std::vector<T>>>(m, std::vector<T>(p, 0.0));

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = 0; k < n; ++k) {
                (*mul)[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return mul;
}
template std::unique_ptr<std::vector<std::vector<double>>> SMatrix2::matMul<double> (const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&);
template std::unique_ptr<std::vector<std::vector<float>>> SMatrix2::matMul<float> (const std::vector<std::vector<float>>&, const std::vector<std::vector<float>>&);

#ifdef USE_PYBIND
// 擬似逆行列を求める
template <typename MatrixType, typename VectorType, typename T>
std::unique_ptr<std::vector<std::vector<T>>> SMatrix2::GetInversePy (const std::vector<std::vector<T>>& mat){
    // matをEigenに変換
    MatrixType eigenMat = vectorMatrixToEigenMatrix<MatrixType, T>(mat);

    std::cout << eigenMat << std::endl;

    // 特異値分解
    //Eigen::JacobiSVD<Eigen::MatrixXd> svd(eigenMat, Eigen::ComputeThinU | Eigen::ComputeThinV);

    /////////////
    /////////////
    // 1. Python interpreter initialization
    //py::scoped_interpreter guard{};

    // 3. Convert Eigen matrix to Numpy array
    py::array_t<T> np_matrix({eigenMat.rows(), eigenMat.cols()});
    for (size_t i = 0; i < (size_t)eigenMat.rows(); ++i) {
        for (size_t j = 0; j < (size_t)eigenMat.cols(); ++j) {
            *np_matrix.mutable_data(i, j) = eigenMat(i, j);
        }
    }

    // 4. Import numpy and call np.linalg.svd
    py::module_ np = py::module_::import("numpy");
    py::object svd = np.attr("linalg").attr("svd");
    py::object pinv = np.attr("linalg").attr("pinv");

    // 5. Perform SVD on the matrix
    py::tuple result = svd(np_matrix, py::arg("full_matrices") = false);

    // 6. Extract U, S, V from the result
    py::array_t<T> U_py = result[0].cast<py::array_t<T>>();
    py::array_t<T> S_py = result[1].cast<py::array_t<T>>();
    py::array_t<T> V_transpose = result[2].cast<py::array_t<T>>();

    // Transpose V to get V
    py::object transpose = np.attr("transpose");
    //py::array_t<T> V_py = transpose(V_transpose).cast<py::array_t<T>>();
    auto tmp_V_py = transpose(V_transpose);
    py::array_t<T> V_py = tmp_V_py.template cast<py::array_t<T>>();

    // 7. Convert Numpy arrays back to Eigen matrices
    MatrixType U(U_py.shape(0), U_py.shape(1));
    MatrixType V(V_py.shape(0), V_py.shape(1));
    VectorType S(S_py.shape(0));

    // Fill U
    for (size_t i = 0; i < (size_t)U_py.shape(0); ++i) {
        for (size_t j = 0; j < (size_t)U_py.shape(1); ++j) {
            U(i, j) = *U_py.data(i, j);
        }
    }

    // Fill V
    for (size_t i = 0; i < (size_t)V_py.shape(0); ++i) {
        for (size_t j = 0; j < (size_t)V_py.shape(1); ++j) {
            V(i, j) = *V_py.data(i, j);
        }
    }

    // Fill S
    for (size_t i = 0; i < (size_t)S_py.shape(0); ++i) {
        S(i) = *S_py.data(i);
    }

    /////////////
    /////////////

    // U, S, V の取得
    SMatrix2::logger->debug("Run SMatrix2::GetInversePy");
    SMatrix2::logger->debug("V:");
    for (int i = 0; i < V.rows(); ++i) {
        for (int j = 0; j < V.cols(); ++j) {
            SMatrix2::logger->debug("V[{}][{}] = {}", i, j, V(i, j));
        }
    }

    SMatrix2::logger->debug("S:");
    for (int i = 0; i < S.size(); ++i) {
        SMatrix2::logger->debug("S[{}] = {}", i, S(i));
    }

    SMatrix2::logger->debug("U:");
    for (int i = 0; i < U.rows(); ++i) {
        for (int j = 0; j < U.cols(); ++j) {
            SMatrix2::logger->debug("U[{}][{}] = {}", i, j, U(i, j));
        }
    }

    T epsilon = 1.0e-10;
    // 特異値の逆数を計算（小さすぎる値は0にする）
    VectorType S_inv(S.size());
    for (size_t i = 0; i < (size_t)S.size(); ++i) {
        S_inv(i) = (S(i) > epsilon) ? (1.0 / S(i)) : 0.0;
    }

    // S を対角行列に変換 (n × m の適切なサイズにする)
    MatrixType Sigma_pinv = MatrixType::Zero(S.size(), S.size());
    for (size_t i = 0; i < (size_t)S_inv.size(); ++i) {
        Sigma_pinv(i, i) = S_inv(i);
    }

    // 擬似逆行列を計算
    MatrixType Ainv = V * Sigma_pinv * U.transpose();

    std::cout << Ainv(0,0) << std::endl;
    for (int i = 0; i < Ainv.rows(); ++i) {
        for (int j = 0; j < Ainv.cols(); ++j) {
            SMatrix2::logger->debug("Ainv[{}][{}] = {}", i, j, Ainv(i, j));
        }
    }

    // 擬似逆行列をvectorに変換
    return eigenMatrixToUniquePtr<MatrixType, T>(Ainv);
}
template std::unique_ptr<std::vector<std::vector<double>>> SMatrix2::GetInversePy<Eigen::MatrixXd, Eigen::VectorXd, double> (const std::vector<std::vector<double>>&);
template std::unique_ptr<std::vector<std::vector<float>>> SMatrix2::GetInversePy<Eigen::MatrixXf, Eigen::VectorXf, float> (const std::vector<std::vector<float>>&);
#endif