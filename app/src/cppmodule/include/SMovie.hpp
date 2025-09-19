#ifndef SMOVIE_H_
#define SMOVIE_H_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include "SThread.hpp"

namespace fs = std::filesystem;

class SMovie {
    private:
        static constexpr double s_scale = 1.0;                    // 動画の画角のサイズのスケール
        static constexpr int s_warp_type = cv::MOTION_HOMOGRAPHY; // findTransformECC()で使うモーションタイプ
        cv::Mat m_warp = cv::Mat::eye(3, 3, CV_32F);              // findTransformECC()で推定される変換行列を格納する変数
        cv::Mat m_base;                                           // findTransformECC()で使う動画の先頭フレーム
        cv::VideoCapture m_inputVideo;                            // ビデオインプット
        cv::VideoWriter m_outputVideo;                            // ビデオアウトプット
        std::vector<cv::Mat> m_frames;                            // 変換前のフレーム画像のベクターコンテナ
        std::vector<cv::Mat> m_dstImages;                         // 変換後のフレーム画像のベクターコンテナ
        std::string m_outputPath;                                 // 出力パス
        double m_fps;                                             // 入力動画のFPS
        cv::Size m_size;                                          // 入力動画の画角のサイズ
        int m_totalFrameNum;                                      // 入力動画のフレーム数
        std::string m_fourccStr;                                  // 入力動画のfourcc
        std::mutex m_mtx;

        cv::Mat CreateStabilizeImage(int index);
        cv::Mat CreateStabilizeImage(int index, cv::Mat& warp);

    public:
        SMovie(){};
        void Read(const std::string& fPath);
        void CreateFrames();
        void Stabilize();
        void CreateOutputVideo();
        void Release();
};

#endif // SMOVIE_H_