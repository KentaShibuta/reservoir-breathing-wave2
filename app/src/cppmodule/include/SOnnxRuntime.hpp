#ifndef SONNXRUNTIME_H_
#define SONNXRUNTIME_H_

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <thread>
#include <onnxruntime_cxx_api.h>


class SOnnxRuntime{
    public:
        inline SOnnxRuntime (){};

        static std::pair<std::vector<float>, std::vector<int64_t>> runInference(
            const std::string& model_path,
            const std::vector<float>& input_data,
            const std::vector<int64_t>& input_dims);

};

#endif // SONNXRUNTIME_H_