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

inline std::unique_ptr<std::vector<std::vector<float>>> readCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ファイルを開けませんでした: " << filename << std::endl;
        return nullptr;
    }

    // 空の2次元vectorをunique_ptrで作成
    auto data = std::make_unique<std::vector<std::vector<float>>>();

    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;

        // カンマ区切りで1行を分割
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stof(cell));  // float型に変換
            } catch (const std::invalid_argument&) {
                std::cerr << "数値変換エラー: " << cell << std::endl;
            }
        }

        if (!row.empty()) {
            data->push_back(std::move(row)); // unique_ptr内のvectorに追加
        }
    }

    file.close();
    return data;
}

// === 2次元vectorのブロックを抽出する関数 ===
// Eigenの block(row, col, numRows, numCols) に相当
inline std::unique_ptr<std::vector<std::vector<float>>> extractBlock(const std::vector<std::vector<float>>& data, size_t startRow, size_t startCol, size_t numRows, size_t numCols)
{
    auto result = std::make_unique<std::vector<std::vector<float>>>();

    if (data.empty()) return result;
    size_t totalRows = data.size();
    size_t totalCols = data[0].size();

    // 範囲チェック
    if (startRow + numRows > totalRows || startCol + numCols > totalCols) {
        std::cerr << "ブロック範囲がデータのサイズを超えています。" << std::endl;
        return result;
    }

    // 部分ブロックを抽出
    result->reserve(numRows);
    for (size_t i = startRow; i < startRow + numRows; ++i) {
        std::vector<float> row;
        row.reserve(numCols);
        for (size_t j = startCol; j < startCol + numCols; ++j) {
            row.push_back(data[i][j]);
        }
        result->push_back(std::move(row));
    }

    return result;
}


// 2次元ベクトルをCSVファイルに書き出す関数
inline void Write2DVectorToFile(const std::string& filename, const std::vector<std::vector<float>>& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: ファイルを開けませんでした: " << filename << std::endl;
        return;
    }

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i != row.size() - 1) {
                file << ",";  // 列の区切り
            }
        }
        file << "\n";  // 行の区切り
    }

    file.close();
}

#endif // SUTIL_H_