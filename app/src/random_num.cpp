#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

template <typename T>
void print(const std::vector<std::vector<T>> &vec){
    for (const auto& row : vec){
        for (const auto& elem : row){
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void print1d(const std::vector<T> &vec){
    std::cout << std::scientific << std::setprecision(17);
    for (const auto& elem : vec){
        std::cout << elem << std::endl;
    }
}

template <typename T>
void outputcsv1d(const std::vector<T> &vec){
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(17);

    for (const auto& elem : vec){
        oss << elem << std::endl;
    }

    std::string str = oss.str();

    std::ofstream outputfile("./random_c.csv");
    outputfile << str;
    outputfile.close();
}

int main(void){
    /*
    uint_fast32_t seed = 0;
    std::mt19937 gen(seed);

    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<std::vector<double>> vec(3, std::vector<double>(3, 0.0));
    for (std::size_t i = 0; i < 3; i++) {
        for (std::size_t j = 0; j < 3; j++) {
            vec[i][j] = dist(gen);
        }
    }

    print(vec);
    */

    uint_fast32_t seed = 0;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    //std::vector<std::vector<double>> vec(3, std::vector<double>(3, 0.0));
    //std::vector<std::vector<uint32_t>> vec_int(3, std::vector<uint32_t>(3, 0.0));
    //double low = -1.0;
    //double high = 1.0;

    //std::cout << gen.max() << std::endl;

    size_t N_x = 500;
    size_t N_u = 60000;
    size_t rand_num = N_x * N_u;
    std::vector<double> vec(rand_num, 0.0);

    for (std::size_t i = 0; i < vec.size(); i++) {
        vec[i] = dist(gen);
    }

    std::sort(vec.begin(), vec.end());

    //print1d(vec);
    outputcsv1d(vec);

    plt::plot(vec);
    plt::show();
}