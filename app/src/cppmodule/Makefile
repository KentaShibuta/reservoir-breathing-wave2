# CPUアーキテクチャ
ARCH := $(shell uname -m)

# =============================
# コンパイラとオプション
# =============================
CC := g++

ifeq ($(ARCH), x86_64)
    CFLAGS := -O3 -Wall -std=c++23 -fopenmp
    CLINK := -L/root/app/lib/spdlog/build/x86_x64 -lspdlog
else ifeq ($(ARCH), aarch64)
    CFLAGS := -O3 -Wall -std=c++23 -fopenmp
    CLINK := -L/root/app/lib/spdlog/build/aarch64 -lspdlog
else
    $(error Unsupported architecture: $(ARCH))
endif

DEBUG_CFLAGS := -g

ONNX_CLINK := -L/root/app/lib/onnxruntime/build/Linux/RelWithDebInfo -lonnxruntime -Xlinker -rpath -Xlinker /root/app/lib/onnxruntime/build/Linux/RelWithDebInfo
OPENCV_CLINK := -L/root/app/lib/opencv-4.x/build/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lopencv_video -Xlinker -rpath -Xlinker /root/app/lib/opencv-4.x/build/lib

PY_CFLAGS :=  -lpython3.12 -shared -fPIC -fvisibility=hidden $(shell python3 -m pybind11 --includes) -DUSE_PYBIND

PY_INCLUDE := -I/usr/local/include/python3.12 -I/usr/local/lib/python3.12/site-packages/numpy/_core/include -I/root/app/lib/matplotlibcpp

INCLUDES := -I /root/app/lib/eigen3 -I /root/app/src/cppmodule/include -I /root/app/lib/spdlog/include
ONNX_INCLUDES := -I/root/app/lib/onnxruntime/include/onnxruntime/core/session
OPENCV_INCLUDES := -I/root/app/lib/opencv-4.x/include/modules/core/include -I/root/app/lib/opencv-4.x/include/modules/calib3d/include -I/root/app/lib/opencv-4.x/include/modules/features2d/include -I/root/app/lib/opencv-4.x/include/modules/flann/include -I/root/app/lib/opencv-4.x/include/modules/dnn/include -I/root/app/lib/opencv-4.x/include/modules/highgui/include -I/root/app/lib/opencv-4.x/include/modules/imgcodecs/include -I/root/app/lib/opencv-4.x/include/modules/videoio/include -I/root/app/lib/opencv-4.x/include/modules/imgproc/include -I/root/app/lib/opencv-4.x/include/modules/ml/include -I/root/app/lib/opencv-4.x/include/modules/objdetect/include -I/root/app/lib/opencv-4.x/include/modules/photo/include -I/root/app/lib/opencv-4.x/include/modules/stitching/include -I/root/app/lib/opencv-4.x/include/modules/video/include -I/root/app/lib/opencv-4.x/include/include -I/root/app/lib/opencv-4.x/build
TEST_INCLUDES := -I/root/app/lib/doctest
TEST_CFLAGS := -DTEST
EXT_SUFFIX := $(shell python3-config --extension-suffix)

# =============================
# ソースファイル
# =============================
SRC := /root/app/src/cppmodule/src/ESN.cpp /root/app/src/cppmodule/src/SMatrix.cpp /root/app/src/cppmodule/src/SMatrix2.cpp
ONNX_SRC := /root/app/src/cppmodule/src/SOnnxRuntime.cpp
OPENCV_SRC := /root/app/src/cppmodule/src/SMovie.cpp
MAINSRC := /root/app/src/cppmodule/src/ObjectDetection.cpp /root/app/src/cppmodule/src/main.cpp


# =============================
# 出力ファイル
# =============================
TARGET_CPP := esn.out
TARGET_DEBUG_CPP := esn.debug
TARGET_PY := esn$(EXT_SUFFIX)


# =============================
# デフォルトターゲット（通常C++ビルド）
# =============================
all: cpp

# =============================
# 通常C++ビルド
# =============================
cpp: $(SRC) $(ONNX_SRC) $(OPENCV_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $(ONNX_INCLUDES) $(OPENCV_INCLUDES) $(SRC) $(ONNX_SRC) $(OPENCV_SRC) $(MAINSRC) $(CLINK) $(ONNX_CLINK) $(OPENCV_CLINK) -o $(TARGET_CPP)

# =============================
# デバッグC++ビルド
# =============================
cpp-debug: $(SRC) $(ONNX_SRC) $(OPENCV_SRC)
	$(CC) $(CFLAGS) $(DEBUG_CFLAGS) $(INCLUDES) $(ONNX_INCLUDES) $(OPENCV_INCLUDES) $(SRC) $(ONNX_SRC) $(OPENCV_SRC) $(MAINSRC) $(CLINK) $(ONNX_CLINK) $(OPENCV_CLINK) -o $(TARGET_DEBUG_CPP)

# =============================
# Python拡張モジュールビルド
# =============================
python: $(SRC)
	$(CC) $(CFLAGS) $(PY_CFLAGS) $(INCLUDES) $(PY_INCLUDE) $(SRC) $(CLINK) -o $(TARGET_PY)

# =============================
# Doctestビルド＆実行
# =============================
test: $(SRC) $(TEST_SRC)
	$(CC) $(CFLAGS) $(TEST_CFLAGS) $(INCLUDES) $(TEST_INCLUDES) $(SRC) -o $(TARGET_CPP)

# =============================
# クリーン
# =============================
clean:
	rm -f $(TARGET_CPP) $(TARGET_PY) $(TARGET_TEST)
