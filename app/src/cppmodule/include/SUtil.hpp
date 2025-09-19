#ifndef SUTIL_H_
#define SUTIL_H_

#include <ctime>
#include <iomanip>
#include <sstream>
#include <fstream>

inline std::string currentDateTime() {
    std::time_t t = std::time(nullptr);
    std::tm* now = std::localtime(&t);
 
    char buffer[128];
    strftime(buffer, sizeof(buffer), "%m-%d-%Y %X", now);
    return buffer;
}

template <typename T>
inline void outputcsv1d(const std::vector<T> &vec){
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(17);

    for (const auto& elem : vec){
        oss << elem << std::endl;
    }

    std::string str = oss.str();

    std::string fname = "./log/random_c_"+ currentDateTime() +".csv";
    std::ofstream outputfile(fname);
    outputfile << str;
    outputfile.close();
}

inline std::string bool_to_string(bool b) {
    return b ? "true" : "false";
}

#endif // SUTIL_H_