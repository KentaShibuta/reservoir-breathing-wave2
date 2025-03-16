#ifndef SMATRIX_H_
#define SMATRIX_H_

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <omp.h>
#include <random>
#include <Dense> // Eigen

class SMatrix{
    public:
        SMatrix(){};
        std::unique_ptr<std::vector<std::vector<float>>> generate_uniform_random(std::size_t row_size, std::size_t col_size, float scale);
        std::unique_ptr<std::vector<std::vector<float>>> generate_normal_distribution(std::size_t row_size, std::size_t col_size, float mean = 0.0, float stddev = 1.0);
        Eigen::MatrixXf vectorMatrixToEigenMatrix(const std::vector<std::vector<float>>& vectorMatrix);
        std::unique_ptr<std::vector<std::vector<float>>> eigenMatrixToUniquePtr(const Eigen::MatrixXf& matrix);
        std::unique_ptr<std::vector<std::vector<uint8_t>>> generate_erdos_renyi(size_t N_x, float density);
        std::unique_ptr<std::vector<float>> dot (const std::vector<std::vector<float>> &mat, const std::vector<float> &vec);
        std::unique_ptr<std::vector<std::vector<float>>> matMul (const std::vector<std::vector<float>> &A, const std::vector<std::vector<float>> &B);
        std::unique_ptr<std::vector<std::vector<float>>> GetInverse (const std::vector<std::vector<float>>& mat);
};

#endif // SMATRIX_H_