#ifndef SMATRIX_H_
#define SMATRIX_H_

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <omp.h>
#include <random>
#include <Dense> // Eigen
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

class SMatrix{
    private:
        static std::shared_ptr<spdlog::logger> logger;
    public:
        SMatrix(){
            if (!logger) {  // loggerがまだ初期化されていない場合のみ初期化
                try {
                    // ロガーを作成
                    logger = spdlog::basic_logger_mt("Smat_logger", "./log/log.txt");
                    logger->set_level(spdlog::level::debug);
                }
                catch (const spdlog::spdlog_ex &e) {
                    std::cerr << "Log initialization failed: " << e.what() << std::endl;
                }
            }
        };
        std::unique_ptr<std::vector<std::vector<float>>> generate_uniform_random(std::size_t row_size, std::size_t col_size, float scale);
        std::unique_ptr<std::vector<std::vector<float>>> generate_normal_distribution(std::size_t row_size, std::size_t col_size, float mean = 0.0, float stddev = 1.0);
        Eigen::MatrixXf vectorMatrixToEigenMatrix(const std::vector<std::vector<float>>& vectorMatrix);
        Eigen::MatrixXd vectorMatrixToEigenMatrixd(const std::vector<std::vector<double>>& vectorMatrix);
        std::unique_ptr<std::vector<std::vector<float>>> eigenMatrixToUniquePtr(const Eigen::MatrixXf& matrix);
        std::unique_ptr<std::vector<std::vector<double>>> eigenMatrixToUniquePtrd(const Eigen::MatrixXd& matrix);
        std::unique_ptr<std::vector<std::vector<uint8_t>>> generate_erdos_renyi(size_t N_x, float density);
        std::unique_ptr<std::vector<float>> dot (const std::vector<std::vector<float>> &mat, const std::vector<float> &vec);
        std::unique_ptr<std::vector<std::vector<float>>> matMul (const std::vector<std::vector<float>> &A, const std::vector<std::vector<float>> &B);
        std::unique_ptr<std::vector<std::vector<double>>> matMuld (const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B);
        std::unique_ptr<std::vector<std::vector<float>>> GetInverse (const std::vector<std::vector<float>>& mat);
        std::unique_ptr<std::vector<std::vector<double>>> GetInversePy (const std::vector<std::vector<double>>& mat);
        py::tuple GetInversePy2 (py::array_t<double> mat);
};

#endif // SMATRIX_H_