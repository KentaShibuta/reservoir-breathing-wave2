#include "ObjectDetection.hpp"

// --------------------------------------------------------------
// プログラムのエントリーポイント
// --------------------------------------------------------------
int main() {
    std::string inputImgDir = "/root/app/data/object_detection_test_data";
    std::string outputImgDir = "/root/app/data/object_detection_test_data";
    std::string modelDir = "/root/app/model";
    std::string inputImg = "LINE_ALBUM_ネコ写真資料２_250518_52.jpg";
    std::string scaler = "20250919_014211_scale.dat";
    std::string esnWeight = "20250919_014415_700_wout.dat";
    std::string cnnWeight = "vgg16_block5_conv3.onnx";
    std::string logDir = "/root/app/src/cppmodule/log";

    ObjectDetection objdtc = ObjectDetection();
    objdtc.Run(inputImgDir, outputImgDir, modelDir, inputImg, scaler, esnWeight, cnnWeight, logDir);
}
