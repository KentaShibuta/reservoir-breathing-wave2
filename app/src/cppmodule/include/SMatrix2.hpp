#ifndef SMATRIX2_H_
#define SMATRIX2_H_

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <omp.h>
#include <random>
#include <Dense> // Eigen
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#ifdef USE_PYBIND
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#endif

class SMatrix2{
    private:
        static std::shared_ptr<spdlog::logger> logger;
    public:
        SMatrix2(){
            if (!logger) {  // loggerがまだ初期化されていない場合のみ初期化
                try {
                    // ロガーを作成
                    logger = spdlog::basic_logger_mt("Smat2_logger", "./log/log.txt");
                    logger->set_level(spdlog::level::debug);
                }
                catch (const spdlog::spdlog_ex &e) {
                    std::cerr << "Log initialization failed: " << e.what() << std::endl;
                }
            }
        };
#ifdef USE_PYBIND
        template <typename T>
        std::unique_ptr<std::vector<std::vector<T>>> generate_uniform_random(std::size_t row_size, std::size_t col_size, T scale);
#endif
        template <typename T>
        std::unique_ptr<std::vector<std::vector<T>>> generate_normal_distribution(std::size_t row_size, std::size_t col_size, T mean, T stddev);
        template <typename MatrixType, typename T>
        MatrixType vectorMatrixToEigenMatrix(const std::vector<std::vector<T>>& vectorMatrix);
        template <typename MatrixType, typename T>
        std::unique_ptr<std::vector<std::vector<T>>> eigenMatrixToUniquePtr(const MatrixType& matrix);
        template <typename T>
        std::unique_ptr<std::vector<std::vector<uint8_t>>> generate_erdos_renyi(size_t N_x, T density);
        template <typename T>
        std::unique_ptr<std::vector<T>> dot (const std::vector<std::vector<T>> &mat, const std::vector<T> &vec);
        template <typename T>
        std::unique_ptr<std::vector<std::vector<T>>> matMul (const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B);
        template <typename T>
        std::unique_ptr<std::vector<std::vector<T>>> matAdd (const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B);
        template <typename MatrixType, typename VectorType, typename T>
        std::unique_ptr<std::vector<std::vector<T>>> GetInverse (const std::vector<std::vector<T>>& mat, T epsilon);
#ifdef USE_PYBIND
        template <typename MatrixType, typename VectorType, typename T>
        std::unique_ptr<std::vector<std::vector<T>>> GetInverseSVD (const std::vector<std::vector<T>>& mat, T epsilon);
        template <typename MatrixType, typename VectorType, typename T>
        std::unique_ptr<std::vector<std::vector<T>>> GetInverseNumpy (const std::vector<std::vector<T>>& mat, bool isPinv=true);
#endif
};

#endif // SMATRIX2_H_