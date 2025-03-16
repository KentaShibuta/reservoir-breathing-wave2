# =============================
# コンパイラとオプション
# =============================
CC := g++
CFLAGS := -O3 -Wall -std=c++23 -fopenmp
INCLUDES := -I /root/app/lib/eigen3 -I /root/app/src/cppmodule/include
PY_CFLAGS := -shared -fPIC -fvisibility=hidden $(shell python3 -m pybind11 --includes) -DUSE_PYBIND
TEST_INCLUDES := -I/root/app/lib/doctest
TEST_CFLAGS := -DTEST
EXT_SUFFIX := $(shell python3-config --extension-suffix)

# =============================
# ソースファイル
# =============================
SRC := /root/app/src/cppmodule/src/esn.cpp /root/app/src/cppmodule/src/SMatrix.cpp
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
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) -o $(TARGET_CPP)

# =============================
# Python拡張モジュールビルド
# =============================
python: $(SRC)
	$(CC) $(CFLAGS) $(PY_CFLAGS) $(INCLUDES) -o $(TARGET_PY) $(SRC)

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
