#ifndef OBJECTDETECTION_H_
#define OBJECTDETECTION_H_

#include "SOnnxRuntime.hpp"
#include "SImage.hpp"
#include "SIOBinary.hpp"
#include "ESN.hpp"

class ObjectDetection{
    public:
        void Run(const std::string& inputImgDir, const std::string& outputImgDir, const std::string& modelDir, const std::string& inputImg, const std::string& scaler, const std::string& esnWeight, const std::string& cnnWeight, const std::string& logDir);
};

#endif // OBJECTDETECTION_H_
