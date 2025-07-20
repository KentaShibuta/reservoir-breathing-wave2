#include "SMovie.hpp"
#include <chrono>

// 動画読み込み
void SMovie::Read(const std::string& fPath) {
    m_inputVideo.open(fPath);
    if (!m_inputVideo.isOpened()) {
        std::cerr << "動画が開けませんでした。" << std::endl;
        return;
    }

    m_fps = m_inputVideo.get(cv::CAP_PROP_FPS);
    m_size = cv::Size(static_cast<int>(m_inputVideo.get(cv::CAP_PROP_FRAME_WIDTH)),
                    static_cast<int>(m_inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT)));
    m_totalFrameNum = static_cast<int>(m_inputVideo.get(cv::CAP_PROP_FRAME_COUNT));

    int codec = static_cast<int>(m_inputVideo.get(cv::CAP_PROP_FOURCC));
    char ext[] = {
        (char)(codec & 0XFF),
        (char)((codec & 0XFF00) >> 8),
        (char)((codec & 0XFF0000) >> 16),
        (char)((codec & 0XFF000000) >> 24),
        '\0'
    };
    m_fourccStr = std::string(ext);

    std::cout << "fps: " << m_fps << std::endl;
    std::cout << "(width, height): " << m_size << std::endl;
    std::cout << "total frame num: " << m_totalFrameNum << std::endl;
    std::cout << "codec: " << m_fourccStr << std::endl;

    std::string fName = fs::path(fPath).stem().string();
    fs::create_directories("../data/output");
    m_outputPath = "/root/app/data/output/" + fName + "_stabilization.mp4";
}

// 動画からフレーム画像を取得
void SMovie::CreateFrames() {
    cv::Mat baseFrame;
    m_inputVideo.read(baseFrame);
    m_size = cv::Size(int(m_size.width * s_scale), int(m_size.height * s_scale));
    std::cout << "scaled (width, height): " << m_size << std::endl;

    cv::resize(baseFrame, baseFrame, m_size);
    cv::cvtColor(baseFrame, m_base, cv::COLOR_BGR2GRAY);

    int m_fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    m_outputVideo.open(m_outputPath, m_fourcc, m_fps, m_size);
    if (!m_outputVideo.isOpened()) {
        std::cerr << "出力動画の作成に失敗しました。" << std::endl;
        return;
    }

    m_inputVideo.set(cv::CAP_PROP_POS_FRAMES, 0);
    m_frames.clear();
    m_dstImages.resize(m_totalFrameNum);

    for (int i = 0; i < m_totalFrameNum; ++i) {
        cv::Mat frame;
        if (!m_inputVideo.read(frame)) break;
        m_frames.push_back(frame);
    }
}
/*
void SMovie::CreateFrames() {
    // フレーム間引き間隔（例: 2で1/2に間引き）
    const int skip_rate = 1;

    // 動画のサイズやFPSなどを設定済み前提（Readで実行済み）
    m_size = cv::Size(int(m_size.width * s_scale), int(m_size.height * s_scale));
    std::cout << "scaled (width, height): " << m_size << std::endl;

    // 出力用FPSを間引きに合わせて調整（再生速度維持のため）
    m_fps /= skip_rate;

    // 出力用VideoWriterを初期化
    int m_fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    m_outputVideo.open(m_outputPath, m_fourcc, m_fps, m_size);
    if (!m_outputVideo.isOpened()) {
        std::cerr << "出力動画の作成に失敗しました。" << std::endl;
        return;
    }

    // 入力動画の先頭に戻る
    m_inputVideo.set(cv::CAP_PROP_POS_FRAMES, 0);
    m_frames.clear();

    // フレームを間引きながら読み込み
    int frameIndex = 0;
    while (frameIndex < m_totalFrameNum) {
        m_inputVideo.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
        cv::Mat frame;
        if (!m_inputVideo.read(frame)) break;

        // 最初の1フレームを基準フレームに設定
        if (m_frames.empty()) {
            cv::Mat baseFrame;
            cv::resize(frame, baseFrame, m_size);
            cv::cvtColor(baseFrame, m_base, cv::COLOR_BGR2GRAY);
        }

        m_frames.push_back(frame);
        frameIndex += skip_rate;
    }

    // 更新された総フレーム数に基づき出力バッファを確保
    m_totalFrameNum = static_cast<int>(m_frames.size());
    m_dstImages.resize(m_totalFrameNum);
}
*/

// 全フレーム画像から手ぶれ補正後の画像を作成
/*
// シングルスレッド用
void SMovie::Stabilize() {
    //#pragma omp parallel for
    for (int i = 0; i < m_totalFrameNum; ++i) {
        if (i == 0 || i % 10 == 0 || i == m_totalFrameNum - 1)
            std::cout << "progressCount: " << i + 1 << " / " << m_totalFrameNum << std::endl;

        m_dstImages[i] = CreateStabilizeImage(i);
    }
}
*/


// OpenMPによるマルチスレッド用
/*
void SMovie::Stabilize() {
    int mid = m_totalFrameNum / 2;

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            cv::Mat warp = cv::Mat::eye(3, 3, CV_32F);
            for (int i = 0; i < mid; ++i) {
                if (i == 0 || i % 10 == 0 || i == m_totalFrameNum - 1)
                    std::cout << "progressCount: " << i + 1 << " / " << m_totalFrameNum << std::endl;

                m_dstImages[i] = CreateStabilizeImage(i, warp);
            }
        }

        #pragma omp section
        {
            cv::Mat warp = cv::Mat::eye(3, 3, CV_32F);
            for (int i = mid; i < m_totalFrameNum; ++i) {
                if (i == 0 || i % 10 == 0 || i == m_totalFrameNum - 1)
                    std::cout << "progressCount: " << i + 1 << " / " << m_totalFrameNum << std::endl;

                m_dstImages[i] = CreateStabilizeImage(i, warp);
            }
        }
    }
}
*/

/*
void SMovie::Stabilize() {
    int mid = m_totalFrameNum / 2;

    // スレッド関数（ラムダ）1: 0 ～ mid-1
    auto process_first_half = [this, mid]() {
        cv::Mat warp = cv::Mat::eye(3, 3, CV_32F);
        for (int i = 0; i < mid; ++i) {
            if (i == 0 || i % 10 == 0 || i == m_totalFrameNum - 1)
                std::cout << "progressCount: " << i + 1 << " / " << m_totalFrameNum << std::endl;

            m_dstImages[i] = CreateStabilizeImage(i, warp);
        }
    };

    // スレッド関数（ラムダ）2: mid ～ m_totalFrameNum-1
    auto process_second_half = [this, mid]() {
        cv::Mat warp = cv::Mat::eye(3, 3, CV_32F);
        for (int i = mid; i < m_totalFrameNum; ++i) {
            if (i == 0 || i % 10 == 0 || i == m_totalFrameNum - 1)
                std::cout << "progressCount: " << i + 1 << " / " << m_totalFrameNum << std::endl;

            m_dstImages[i] = CreateStabilizeImage(i, warp);
        }
    };

    // スレッドの生成と実行
    std::thread thread1(process_first_half);
    std::thread thread2(process_second_half);

    // スレッドの終了待機
    thread1.join();
    thread2.join();
}
*/

/*
// 各スレッドで使うデータをインスタンスで保持するVer
void SMovie::Stabilize() {
    int numThreads = 6;

    if (numThreads <= 0) numThreads = 1;
    if (numThreads > m_totalFrameNum) numThreads = m_totalFrameNum;

    std::vector<std::thread> threads;
    int framesPerThread = m_totalFrameNum / numThreads;
    int remainingFrames = m_totalFrameNum % numThreads;

    std::vector<SThread> threadData(numThreads);

    int start = 0;
    for (int t = 0; t < numThreads; ++t) {
        int end = start + framesPerThread + (t < remainingFrames ? 1 : 0);

        threadData[t].Init(start, end);
        
        start = end;
    }

    for (auto& elem : threadData){
        threads.emplace_back([this, &elem]() {
            cv::Mat warp = cv::Mat(cv::Mat::eye(3, 3, CV_32F)).clone();
            for (size_t i = elem.GetStart(); i < elem.GetEnd(); ++i) {
                elem.SetDstImg(i-elem.GetStart(), CreateStabilizeImage(i, warp));
            }
        });
    }

    // すべてのスレッドの終了を待機
    for (auto& thread : threads) {
        thread.join();
    }

    m_dstImages.clear();
    for (const auto& elem : threadData){
        auto data = elem.GetDstImg();
        
        for (size_t i = 0; i < data.size(); i++){
            if (data[i].empty()) {
                std::cerr << "Warning: empty frame at index " << i << std::endl;
            }
        }

        m_dstImages.insert(m_dstImages.end(), data.begin(), data.end());
    }
}
*/

void SMovie::Stabilize() {
    int numThreads = 6;

    if (numThreads <= 0) numThreads = 1;
    if (numThreads > m_totalFrameNum) numThreads = m_totalFrameNum;

    std::vector<std::thread> threads;
    int framesPerThread = m_totalFrameNum / numThreads;
    int remainingFrames = m_totalFrameNum % numThreads;

    int start = 0;
    for (int t = 0; t < numThreads; ++t) {
        int end = start + framesPerThread + (t < remainingFrames ? 1 : 0);

        threads.emplace_back([this, start, end]() {
            //cv::Mat warp = cv::Mat::eye(3, 3, CV_32F);
            cv::Mat warp = cv::Mat(cv::Mat::eye(3, 3, CV_32F)).clone();
            for (int i = start; i < end; ++i) {
                //if (i == 0 || i % 10 == 0 || i == m_totalFrameNum - 1)
                    //std::cout << "progressCount: " << i + 1 << " / " << m_totalFrameNum << std::endl;

                //m_mtx.lock(); 
                m_dstImages[i] = CreateStabilizeImage(i, warp);
                //m_mtx.unlock();

                // optional: デバッグ用
                //if (m_dstImages[i].empty()) {
                //    std::cerr << "Warning: empty frame at index " << i << std::endl;
                //}
            }
        });

        start = end;
    }

    // すべてのスレッドの終了を待機
    for (auto& thread : threads) {
        thread.join();
    }
}

/*
void SMovie::Stabilize() {
    const int numThreads = std::thread::hardware_concurrency(); // 例：8コアなら8
    const int framesPerThread = m_totalFrameNum / numThreads;

    std::vector<std::thread> threads;
    std::mutex coutMutex;

    auto worker = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            if (i % 10 == 0 || i == m_totalFrameNum - 1) {
                std::lock_guard<std::mutex> lock(coutMutex);
                std::cout << "progressCount: " << i + 1 << " / " << m_totalFrameNum << std::endl;
            }
            m_dstImages[i] = CreateStabilizeImage(i);
        }
    };

    for (int t = 0; t < numThreads; ++t) {
        int start = t * framesPerThread;
        int end = (t == numThreads - 1) ? m_totalFrameNum : start + framesPerThread;
        threads.emplace_back(worker, start, end);
    }

    for (auto& th : threads) {
        th.join();
    }
}
*/

// 手ブレ補正後の画像を結合して手ブレ補正後の動画を作成
void SMovie::CreateOutputVideo() {
    for (const auto& frame : m_dstImages) {
        m_outputVideo.write(frame);
    }
    std::cout << "finish." << std::endl;
}

// indexを指定したフレーム画像から手ブレ補正後の画像を作成する
// シングルスレッド
cv::Mat SMovie::CreateStabilizeImage(int index) {
    cv::Mat frameResized, gray;
    cv::resize(m_frames[index], frameResized, m_size);
    cv::cvtColor(frameResized, gray, cv::COLOR_BGR2GRAY);

    //cv::Mat warp = cv::Mat::eye(3, 3, CV_32F);
    cv::findTransformECC(gray, m_base, m_warp, s_warp_type);
    cv::Mat stabilized;
    cv::warpPerspective(frameResized, stabilized, m_warp, m_size);
    return stabilized;
}

// マルチスレッド
cv::Mat SMovie::CreateStabilizeImage(int index, cv::Mat& warp) {
    if (m_frames[index].empty()) {
        std::cerr << "Empty input frame at index " << index << std::endl;
        return cv::Mat();
    }

    if (m_base.empty()) {
        std::cerr << "Error: m_base is empty." << std::endl;
        return cv::Mat();
    }

    cv::Mat frameResized, gray;
    cv::resize(m_frames[index], frameResized, m_size);
    cv::cvtColor(frameResized, gray, cv::COLOR_BGR2GRAY);

    //cv::findTransformECC(gray, m_base, warp, s_warp_type);
    double ecc = cv::findTransformECC(gray, m_base, warp, s_warp_type);
    if (ecc <= 0) {
        std::cerr << "ECC failed at index " << index << std::endl;
        return cv::Mat(); // 空のMatを返す
    }

    cv::Mat stabilized;
    cv::warpPerspective(frameResized, stabilized, warp, m_size);
    return stabilized;
}

// インプットビデオとアウトプットビデオを解放する
void SMovie::Release() {
    m_inputVideo.release();
    m_outputVideo.release();
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    std::string fName = "/root/app/data/LINE_ALBUM_25621つくば🐶ランドよ_250707_1.mp4";

    SMovie movie;
    movie.Read(fName);
    movie.CreateFrames();
    movie.Stabilize();
    movie.CreateOutputVideo();
    movie.Release();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_sec = end - start;
    std::cout << "実行時間: " << elapsed_sec.count() << " 秒" << std::endl;
    return 0;
}
