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
    CLINK := -L/root/app/lib/spdlog/build/aarch64 -lspdlog -L/root/app/lib/onnxruntime/build/Linux/RelWithDebInfo -lonnxruntime -Xlinker -rpath -Xlinker /root/app/lib/onnxruntime/build/Linux/RelWithDebInfo
else
    $(error Unsupported architecture: $(ARCH))
endif

PY_CFLAGS :=  -lpython3.12 -shared -fPIC -fvisibility=hidden $(shell python3 -m pybind11 --includes) -DUSE_PYBIND

PY_INCLUDE := -I/usr/local/include/python3.12 -I/usr/local/lib/python3.12/site-packages/numpy/_core/include -I/root/app/lib/matplotlibcpp

INCLUDES := -I /root/app/lib/eigen3 -I /root/app/src/cppmodule/include -I /root/app/lib/spdlog/include -I/root/app/lib/onnxruntime/include/onnxruntime/core/session
TEST_INCLUDES := -I/root/app/lib/doctest
TEST_CFLAGS := -DTEST
EXT_SUFFIX := $(shell python3-config --extension-suffix)

# =============================
# ソースファイル
# =============================
SRC := /root/app/src/cppmodule/src/ESN.cpp /root/app/src/cppmodule/src/SMatrix.cpp /root/app/src/cppmodule/src/SMatrix2.cpp /root/app/src/cppmodule/src/SOnnxRuntime.cpp
MAINSRC := /root/app/src/cppmodule/src/main.cpp
#TEST_SRC := main.cpp  # doctest用のテストファイル

# =============================
# 出力ファイル
# =============================
TARGET_CPP := esn.out
TARGET_PY := esn$(EXT_SUFFIX)
#TARGET_TEST := main.out

# =============================
# デフォルトターゲット（通常C++ビルド）
# =============================
all: cpp

# =============================
# 通常C++ビルド
# =============================
cpp: $(SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) $(MAINSRC) $(CLINK) -o $(TARGET_CPP)

# =============================
# Python拡張モジュールビルド
# =============================
python: $(SRC)
	$(CC) $(CFLAGS) $(PY_CFLAGS) $(INCLUDES) $(PY_INCLUDE) $(SRC) $(CLINK) -o $(TARGET_PY)

# =============================
# Doctestビルド＆実行
# =============================
test: $(SRC) $(TEST_SRC)
#	$(CC) $(CFLAGS) $(TEST_CFLAGS) $(INCLUDES) $(TEST_INCLUDES) $(SRC) $(TEST_SRC) -o $(TARGET_TEST)
	$(CC) $(CFLAGS) $(TEST_CFLAGS) $(INCLUDES) $(TEST_INCLUDES) $(SRC) -o $(TARGET_CPP)
#	./$(TARGET_TEST)

# =============================
# クリーン
# =============================
clean:
	rm -f $(TARGET_CPP) $(TARGET_PY) $(TARGET_TEST)
