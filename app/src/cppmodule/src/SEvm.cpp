#include <opencv2/opencv.hpp>
#include <stdint.h>




int main()
{
    // params
    float alpha = 10.0;
    uint32_t level = 4;
    float f_lo = 10.0 / 60.0; // 1分(60秒)間に10回
    float f_hi = 300.0 / 60.0; // 1分(60秒)間に300回
    float scale_factor = 0.3;

    // image.pngをimgに代入
    //cv::Mat inputImg = cv::imread("/root/app/src/cppmodule/src/chris-lynch-DkCGxSPNowg-unsplash.jpg");

    // 動画読み込み
    cv::VideoCapture cap;
    cap.open("/root/app/data/770125907.349203_stabilization.mp4");
    if (cap.isOpened() == false) {
		// 動画ファイルが開けなかったときは終了する
		return 0;
	}

    // 作成する動画ファイルの諸設定
	uint32_t width, height, fourcc;
	double fps;
 
	width  = (uint32_t)cap.get(cv::CAP_PROP_FRAME_WIDTH);	// フレーム横幅を取得
	height = (uint32_t)cap.get(cv::CAP_PROP_FRAME_HEIGHT);	// フレーム縦幅を取得
	fps    = cap.get(cv::CAP_PROP_FPS);					// フレームレートを取得

    //fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
    //fourcc = cv::VideoWriter::fourcc('a','v','c','1');
    fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');	// ISO MPEG-4 / .mp4

    // 動画ファイルを書き出すためのオブジェクトを宣言する
	cv::VideoWriter writer;
	//writer.open("video.mp4", fourcc, fps, cv::Size(width, height));
    writer.open("video.mp4", fourcc, fps, cv::Size(width, height));

    cv::Mat frame, dst;

    for (;;) {
		// cap から frame へ1フレームを取り込む
		cap >> frame;
 
		// 画像から空のとき、無限ループを抜ける
		if (frame.empty() == true) {
			break;
		}
 
        cv::cvtColor(frame, dst, cv::COLOR_BGR2GRAY);
        cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR); // 3チャンネルに戻す
 
		// 画像 dst を動画ファイルへ書き出す
		writer << dst;
 
		// 1ミリ秒待つ
		//cv::waitKey(1);
	}

    // imgの表示
    //cv::Mat gray;
    //cv::cvtColor(inputImg, gray, cv::COLOR_BGR2GRAY);
    //cv::imwrite("output.jpg", gray);

    return 0;
}