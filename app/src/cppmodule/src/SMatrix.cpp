#include "SMatrix.hpp"

std::shared_ptr<spdlog::logger> SMatrix::logger = nullptr;

std::unique_ptr<std::vector<std::vector<float>>> SMatrix::generate_uniform_random(std::size_t row_size, std::size_t col_size, float scale) {
    uint_fast32_t seed = 0;
    std::mt19937 gen(seed); // メルセンヌ・ツイスター法による生成器
    std::uniform_real_distribution<float> dist(-1.0 * scale, std::nextafter(scale, std::numeric_limits<float>::max()));
    //std::uniform_real_distribution<float> dist(-1.0 * scale, scale);

    auto numbers = std::make_unique<std::vector<std::vector<float>>>(row_size, std::vector<float>(col_size));
    for (std::size_t i = 0; i < row_size; i++) {
        for (std::size_t j = 0; j < col_size; j++) {
            (*numbers)[i][j] = dist(gen);
        }
    }
    return numbers;
}

std::unique_ptr<std::vector<std::vector<double>>> SMatrix::generate_uniform_randomd(std::size_t row_size, std::size_t col_size, float scale) {
    uint_fast32_t seed = 0;
    std::mt19937 gen(seed); // メルセンヌ・ツイスター法による生成器
    std::uniform_real_distribution<double> dist(-1.0 * scale, std::nextafter(scale, std::numeric_limits<double>::max()));
    //std::uniform_real_distribution<float> dist(-1.0 * scale, scale);

    auto numbers = std::make_unique<std::vector<std::vector<double>>>(row_size, std::vector<double>(col_size));
    for (std::size_t i = 0; i < row_size; i++) {
        for (std::size_t j = 0; j < col_size; j++) {
            (*numbers)[i][j] = dist(gen);
        }
    }
    return numbers;
}

std::unique_ptr<std::vector<std::vector<float>>> SMatrix::generate_normal_distribution(std::size_t row_size, std::size_t col_size, float mean, float stddev) {
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

std::unique_ptr<std::vector<std::vector<double>>> SMatrix::generate_normal_distributiond(std::size_t row_size, std::size_t col_size, float mean, float stddev) {
    // 乱数生成器と正規分布設定
    uint_fast32_t seed = 0;
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(mean, stddev);

    // N_y行N_x列の2次元配列を作成
    auto numbers = std::make_unique<std::vector<std::vector<double>>>(row_size, std::vector<double>(col_size));
    // 乱数を生成して配列に格納
    for (std::size_t i = 0; i < row_size; i++) {
        for (std::size_t j = 0; j < col_size; j++) {
            (*numbers)[i][j] = dist(gen);
        }
    }
    return numbers;
}

// std::unique_ptr<std::vector<std::vector<float>>> を Eigen::MatrixXf に変換
Eigen::MatrixXf SMatrix::vectorMatrixToEigenMatrix(const std::vector<std::vector<float>>& vectorMatrix) {
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

    Eigen::MatrixXf eigenMat(rows, cols);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            eigenMat(i, j) = vectorMatrix[i][j];
        }
    }

    return eigenMat;
}

// std::unique_ptr<std::vector<std::vector<double>>> を Eigen::MatrixXd に変換
Eigen::MatrixXd SMatrix::vectorMatrixToEigenMatrixd(const std::vector<std::vector<double>>& vectorMatrix) {
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

    Eigen::MatrixXd eigenMat(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            eigenMat(i, j) = vectorMatrix[i][j];
        }
    }

    return eigenMat;
}

// Eigen::MatrixXf を std::unique_ptr<std::vector<std::vector<float>>>  に変換
std::unique_ptr<std::vector<std::vector<float>>> SMatrix::eigenMatrixToUniquePtr(const Eigen::MatrixXf& matrix) {
    // 行列のサイズを取得
    int rows = matrix.rows();
    int cols = matrix.cols();

    // std::unique_ptrで2次元ベクトルを作成
    auto vec = std::make_unique<std::vector<std::vector<float>>>(rows, std::vector<float>(cols));

    // EigenのMatrixXfからstd::vectorにデータをコピー
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            (*vec)[i][j] = matrix(i, j);
        }
    }

    return vec;
}

// Eigen::MatrixXd を std::unique_ptr<std::vector<std::vector<double>>>  に変換
std::unique_ptr<std::vector<std::vector<double>>> SMatrix::eigenMatrixToUniquePtrd(const Eigen::MatrixXd& matrix) {
    // 行列のサイズを取得
    int rows = matrix.rows();
    int cols = matrix.cols();

    // std::unique_ptrで2次元ベクトルを作成
    auto vec = std::make_unique<std::vector<std::vector<double>>>(rows, std::vector<double>(cols));

    // EigenのMatrixXfからstd::vectorにデータをコピー
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            (*vec)[i][j] = matrix(i, j);
        }
    }

    return vec;
}

// ランダムなErdos-Renyiグラフを生成
std::unique_ptr<std::vector<std::vector<uint8_t>>> SMatrix::generate_erdos_renyi(size_t N_x, float density) {
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

std::unique_ptr<std::vector<float>> SMatrix::dot (const std::vector<std::vector<float>> &mat, const std::vector<float> &vec){
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

std::unique_ptr<std::vector<double>> SMatrix::dotd (const std::vector<std::vector<double>> &mat, const std::vector<double> &vec){
    size_t vec_size = vec.size();
    auto y = std::make_unique<std::vector<double>>(vec_size, 0.0);

    #pragma omp parallel for
    for (size_t i = 0; i < mat.size(); i++) {
        for (size_t j = 0; j < vec_size; j++) {
            (*y)[i] += mat[i][j] * vec[j];
        }
    }

    return y;
}

// 行列Aと行列Bの積
std::unique_ptr<std::vector<std::vector<float>>> SMatrix::matMul (const std::vector<std::vector<float>> &A, const std::vector<std::vector<float>> &B){
    size_t m = A.size();    // Aの行数
    size_t n = A[0].size(); // Aの列数 (Bの行数と同じ)
    size_t p = B[0].size(); // Bの列数

    std::cout << "A: (" << m << ", " << n << ")" << std::endl;
    std::cout << "B: (" << n << ", " << p << ")" << std::endl;

    auto mul = std::make_unique<std::vector<std::vector<float>>>(m, std::vector<float>(p, 0.0f));

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = 0; k < n; ++k) {
                (*mul)[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return mul;
}

// 行列Aと行列Bの積
std::unique_ptr<std::vector<std::vector<double>>> SMatrix::matMuld (const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B){
    size_t m = A.size();    // Aの行数
    size_t n = A[0].size(); // Aの列数 (Bの行数と同じ)
    size_t p = B[0].size(); // Bの列数

    std::cout << "A: (" << m << ", " << n << ")" << std::endl;
    std::cout << "B: (" << n << ", " << p << ")" << std::endl;

    auto mul = std::make_unique<std::vector<std::vector<double>>>(m, std::vector<double>(p, 0.0));

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = 0; k < n; ++k) {
                (*mul)[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return mul;
}

// 擬似逆行列を求める
std::unique_ptr<std::vector<std::vector<float>>> SMatrix::GetInverse (const std::vector<std::vector<float>>& mat){
    // matをEigenに変換
    Eigen::MatrixXf eigenMat = vectorMatrixToEigenMatrix(mat);
    Eigen::MatrixXd eigenMat_d = eigenMat.cast<double>();

    std::cout << eigenMat_d << std::endl;

    // 特異値分解
    //Eigen::JacobiSVD<Eigen::MatrixXf> svd(eigenMat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(eigenMat_d, Eigen::ComputeThinU | Eigen::ComputeThinV);
    //Eigen::VectorXf s = svd.singularValues();
    //Eigen::VectorXd s = svd.singularValues();

    // U, S, V の取得
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::VectorXd S = svd.singularValues();
    Eigen::MatrixXd V = svd.matrixV();

    SMatrix::logger->debug("V:");
    for (int i = 0; i < V.rows(); ++i) {
        for (int j = 0; j < V.cols(); ++j) {
            SMatrix::logger->debug("V[{}][{}] = {}", i, j, V(i, j));
        }
    }

    SMatrix::logger->debug("S:");
    for (int i = 0; i < S.size(); ++i) {
        SMatrix::logger->debug("S[{}] = {}", i, S(i));
    }

    SMatrix::logger->debug("U:");
    for (int i = 0; i < U.rows(); ++i) {
        for (int j = 0; j < U.cols(); ++j) {
            SMatrix::logger->debug("U[{}][{}] = {}", i, j, U(i, j));
        }
    }

    /*
    double tolerance = 1.0e-15; // しきい値（例）
    for (size_t i = 0; i < (size_t)s.size(); ++i) {
        //if (std::fabs(s[i]) > tolerance)
        if (s[i] > tolerance)
            s[i] = 1.0 / (s[i] * 1.0);
        else
            s[i] = 0.0; // 小さすぎる特異値は無視
    }
    */

    double epsilon = 1.0e-15;
    // 特異値の逆数を計算（小さすぎる値は0にする）
    Eigen::VectorXd S_inv(S.size());
    for (int i = 0; i < S.size(); ++i) {
        S_inv(i) = (S(i) > epsilon) ? (1.0 / S(i)) : 0.0;
    }

    // S を対角行列に変換 (n × m の適切なサイズにする)
    Eigen::MatrixXd Sigma_pinv = Eigen::MatrixXd::Zero(V.rows(), U.cols());
    for (int i = 0; i < S_inv.size(); ++i) {
        Sigma_pinv(i, i) = S_inv(i);
    }

    // 擬似逆行列を計算
    //Eigen::MatrixXf Ainv = svd.matrixV() * s.asDiagonal() * svd.matrixU().transpose();
    //Eigen::MatrixXd Ainv_d = svd.matrixV().transpose() * s.asDiagonal() * svd.matrixU().transpose();
    Eigen::MatrixXd Ainv_d = V * Sigma_pinv * U.transpose();

    std::cout << Ainv_d(0,0) << std::endl;
    //Eigen::MatrixXd Ainv_d = svd.matrixU() * s.asDiagonal() * svd.matrixV().transpose();
    Eigen::MatrixXf Ainv = Ainv_d.cast<float>();

    // 擬似逆行列をvectorに変換
    return eigenMatrixToUniquePtr(Ainv);
    /*
    std::unique_ptr<std::vector<std::vector<float>>> a;
    return a;
    */
}

// 擬似逆行列を求める
std::unique_ptr<std::vector<std::vector<double>>> SMatrix::GetInversePy (const std::vector<std::vector<double>>& mat){
    // matをEigenに変換
    Eigen::MatrixXd eigenMat_d = vectorMatrixToEigenMatrixd(mat);
    //Eigen::MatrixXd eigenMat_d = eigenMat.cast<double>();

    std::cout << eigenMat_d << std::endl;

    // 特異値分解
    //Eigen::JacobiSVD<Eigen::MatrixXd> svd(eigenMat_d, Eigen::ComputeThinU | Eigen::ComputeThinV);

    /////////////
    /////////////
    // 1. Python interpreter initialization
    //py::scoped_interpreter guard{};

    // 3. Convert Eigen matrix to Numpy array
    py::array_t<double> np_matrix({eigenMat_d.rows(), eigenMat_d.cols()});
    for (size_t i = 0; i < (size_t)eigenMat_d.rows(); ++i) {
        for (size_t j = 0; j < (size_t)eigenMat_d.cols(); ++j) {
            *np_matrix.mutable_data(i, j) = eigenMat_d(i, j);
        }
    }

    // 4. Import numpy and call np.linalg.svd
    py::module_ np = py::module_::import("numpy");
    py::object svd = np.attr("linalg").attr("svd");
    py::object pinv = np.attr("linalg").attr("pinv");

    // 5. Perform SVD on the matrix
    py::tuple result = svd(np_matrix, py::arg("full_matrices") = false);

    // 6. Extract U, S, V from the result
    py::array_t<double> U_py = result[0].cast<py::array_t<double>>();
    py::array_t<double> S_py = result[1].cast<py::array_t<double>>();
    py::array_t<double> V_transpose = result[2].cast<py::array_t<double>>();

    // Transpose V to get V
    py::object transpose = np.attr("transpose");
    py::array_t<double> V_py = transpose(V_transpose).cast<py::array_t<double>>();

    // 7. Convert Numpy arrays back to Eigen matrices
    Eigen::MatrixXd U(U_py.shape(0), U_py.shape(1));
    Eigen::MatrixXd V(V_py.shape(0), V_py.shape(1));
    Eigen::VectorXd S(S_py.shape(0));

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
    SMatrix::logger->debug("Run SMatrix::GetInversePy");
    SMatrix::logger->debug("V:");
    for (int i = 0; i < V.rows(); ++i) {
        for (int j = 0; j < V.cols(); ++j) {
            SMatrix::logger->debug("V[{}][{}] = {}", i, j, V(i, j));
        }
    }

    SMatrix::logger->debug("S:");
    for (int i = 0; i < S.size(); ++i) {
        SMatrix::logger->debug("S[{}] = {}", i, S(i));
    }

    SMatrix::logger->debug("U:");
    for (int i = 0; i < U.rows(); ++i) {
        for (int j = 0; j < U.cols(); ++j) {
            SMatrix::logger->debug("U[{}][{}] = {}", i, j, U(i, j));
        }
    }

    double epsilon = 1.0e-10;
    // 特異値の逆数を計算（小さすぎる値は0にする）
    Eigen::VectorXd S_inv(S.size());
    for (int i = 0; i < S.size(); ++i) {
        S_inv(i) = (S(i) > epsilon) ? (1.0 / S(i)) : 0.0;
    }

    // S を対角行列に変換 (n × m の適切なサイズにする)
    Eigen::MatrixXd Sigma_pinv = Eigen::MatrixXd::Zero(S.size(), S.size());
    for (int i = 0; i < S_inv.size(); ++i) {
        Sigma_pinv(i, i) = S_inv(i);
    }

    // 擬似逆行列を計算
    Eigen::MatrixXd Ainv_d = V * Sigma_pinv * U.transpose();

    /*
    py::array_t<double> Ainv_py = pinv(np_matrix).cast<py::array_t<double>>();
    Eigen::MatrixXd Ainv_d(Ainv_py.shape(0), Ainv_py.shape(1));
    */

    std::cout << Ainv_d(0,0) << std::endl;
    //Eigen::MatrixXf Ainv = Ainv_d.cast<float>();
    for (int i = 0; i < Ainv_d.rows(); ++i) {
        for (int j = 0; j < Ainv_d.cols(); ++j) {
            SMatrix::logger->debug("Ainv_d[{}][{}] = {}", i, j, Ainv_d(i, j));
        }
    }

    // 擬似逆行列をvectorに変換
    //return eigenMatrixToUniquePtr(Ainv);
    return eigenMatrixToUniquePtrd(Ainv_d);
}

// 擬似逆行列を求める
py::tuple SMatrix::GetInversePy2 (py::array_t<double> mat){

    // 4. Import numpy and call np.linalg.svd
    py::module_ np = py::module_::import("numpy");
    py::object svd = np.attr("linalg").attr("svd");

    // 5. Perform SVD on the matrix
    py::tuple result = svd(mat, py::arg("full_matrices") = false);

    // 6. Extract U, S, V from the result
    py::array_t<double> U_py = result[0].cast<py::array_t<double>>();
    py::array_t<double> S_py = result[1].cast<py::array_t<double>>();
    py::array_t<double> V_transpose = result[2].cast<py::array_t<double>>();

    // Transpose V to get V
    py::object transpose = np.attr("transpose");
    py::array_t<double> V_py = transpose(V_transpose).cast<py::array_t<double>>();

    // 7. Convert Numpy arrays back to Eigen matrices
    Eigen::MatrixXd U(U_py.shape(0), U_py.shape(1));
    Eigen::MatrixXd V(V_py.shape(0), V_py.shape(1));
    Eigen::VectorXd S(S_py.shape(0));

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
    SMatrix::logger->debug("Run SMatrix::GetInversePy2");
    SMatrix::logger->debug("V:");
    for (int i = 0; i < V.rows(); ++i) {
        for (int j = 0; j < V.cols(); ++j) {
            SMatrix::logger->debug("V[{}][{}] = {}", i, j, V(i, j));
        }
    }

    SMatrix::logger->debug("S:");
    for (int i = 0; i < S.size(); ++i) {
        SMatrix::logger->debug("S[{}] = {}", i, S(i));
    }

    SMatrix::logger->debug("U:");
    for (int i = 0; i < U.rows(); ++i) {
        for (int j = 0; j < U.cols(); ++j) {
            SMatrix::logger->debug("U[{}][{}] = {}", i, j, U(i, j));
        }
    }

    double epsilon = 1.0e-15;
    // 特異値の逆数を計算（小さすぎる値は0にする）
    Eigen::VectorXd S_inv(S.size());
    for (int i = 0; i < S.size(); ++i) {
        S_inv(i) = (S(i) > epsilon) ? (1.0 / S(i)) : 0.0;
    }

    // S を対角行列に変換 (n × m の適切なサイズにする)
    Eigen::MatrixXd Sigma_pinv = Eigen::MatrixXd::Zero(S.size(), S.size());
    for (int i = 0; i < S_inv.size(); ++i) {
        Sigma_pinv(i, i) = S_inv(i);
    }

    // 擬似逆行列を計算
    Eigen::MatrixXd Ainv_d = V * Sigma_pinv * U.transpose();

    std::cout << Ainv_d(0,0) << std::endl;
    Eigen::MatrixXf Ainv = Ainv_d.cast<float>();
    for (int i = 0; i < Ainv.rows(); ++i) {
        for (int j = 0; j < Ainv.cols(); ++j) {
            SMatrix::logger->debug("Ainv[{}][{}] = {}", i, j, Ainv(i, j));
        }
    }

    // 擬似逆行列をvectorに変換
    return py::make_tuple(U_py, S_py, V_py);
}
