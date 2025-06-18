# reservoir-breathing-wave2
* ESN(Echo State Network)を用いて、動物の安静時の動画から呼吸数を求めるプログラム
* 開発環境は、dockerフォルダにて下記コマンドを実行してコンテナを起動する
   - docker on x86_64
      - ```
        docker compose up -d
        ```
   - docker on Apple silicon
      - ```
        docker compose --env-file .env.mac up -d
        ```
* RSNの学習と推論は、下記文献の付録にあるコードを参考にした
   - リザバーコンピューティング: 時系列パターン認識のための高速機械学習の理論とハードウェア
      - https://www.morikita.co.jp/books/mid/085531
   - 学習と推論処理は、C++で書いたコードをpybind11で共有ライブラリ化したものに置き換えている
      - 行列とベクトルの積は、OpenMPを用いて並列化している
      - 作成した共有ライブラリは、付録の推論処理と同等の結果を出力することを確認済み
      - 作成した共有ライブラリは、Pythonのスクリプトから呼び出して使用可能
      - 付録のコードとC++で書いた共有ライブラリで実行時間を比較した結果
         - 実行時間の測定に使用した環境
            - CPU: AMD Ryzen 5 5600X 6-Core Processor 3.70 GHz
            - メモリ: 32 GB
         - リザバーのノード数：500
         - 入力で与える動画データ
            - 1分程度の動画
            - FPS：30
         - 学習
            - C++：9 秒
            - Python：12 秒
         - 推論
            - C++：6 秒
            - Python：15 秒
   - 学習済みモデルは、.pickleファイルとして「/root/app/model」ディレクトリに保存される
   - 推論で使う学習済みモデルを変更するには、main.pyの「`model_file = "/root/app/model/20250405_013307.pickle"`」を書き換える
* main.py中の変数isTrainについて
   - isTrainをTrueにするとESNの学習が実行される
      - 学習で使用する動画ファイルは、デフォルトで1つdataフォルダに保存している
      - 動画ファイルをフレーム画像に分割し、フレーム画像から、ESNに与えるデータセットを作成する
         - データセットの作成時に、フレーム画像をトリミングしている
         - トリミングの範囲は、デフォルトの動画ファイルに合わせて、コードに定数で埋め込んでいる
            - もし別の動画ファイルを使用する場合は、トリミング範囲を書き換えて実行する
   - isTrainをFalseにするとESNの推論が実行される
      - デフォルトではリポジトリに保存している学習済みモデルを使用して推論する
      - 実行後にターミナルに表示される「peakNum: 」に続く値が呼吸数である
   - 学習と推論のデモ動画
      - https://youtu.be/aZmtaYk__Vk
* C++で書き直したESNモジュールのビルド方法
   - Dockerコンテナにアタッチした後に、下記コマンドを実行する
      - ```
        cd /root/app/src/cppmodule/
        ```
      - ```
        make python
        ```

# License
The source code is licensed [BSD 3-Clause License](LICENSE).  
This source code incorporates the following software.
* Eigen
   * https://eigen.tuxfamily.org/index.php?title=Main_Page
   * [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/)
* doctest
   * https://github.com/doctest/doctest
   * [MIT license](https://github.com/doctest/doctest/blob/master/LICENSE.txt)
* spdlog
   * https://github.com/gabime/spdlog
   * [MIT License](https://github.com/gabime/spdlog?tab=License-1-ov-file#readme)
* matplotlib-cpp
   * https://github.com/lava/matplotlib-cpp
   * [MIT License](https://github.com/lava/matplotlib-cpp/blob/master/LICENSE)
* NumPy
   * https://numpy.org/
   * [BSD-3-Clause license](https://numpy.org/doc/stable/license.html)
* Matplitlib
   * https://matplotlib.org/
   * [Python Software Foundation License](https://matplotlib.org/stable/project/license.html)
* pickle
   * https://docs.python.org/3/library/pickle.html
   * [Python Software Foundation License](https://docs.python.org/3/license.html)
* SciPy
   * https://scipy.org/
   * [BSD-3-Clause license](https://github.com/scipy/scipy?tab=BSD-3-Clause-1-ov-file#readme)
* pandas
   * https://pandas.pydata.org/
   * [BSD 3-Clause License](https://github.com/pandas-dev/pandas?tab=BSD-3-Clause-1-ov-file#readme)
* scikit-learn
   * https://scikit-learn.org/stable/
   * [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn?tab=BSD-3-Clause-1-ov-file#readme)
* NetworkX
   * https://networkx.org/
   * [BSD 3-Clause License](https://raw.githubusercontent.com/networkx/networkx/master/LICENSE.txt)
* OpenCV
   * https://opencv.org/
   * [Apache License 2.0](https://github.com/opencv/opencv/blob/master/LICENSE)
* Pillow
   * https://python-pillow.github.io/
   * [MIT-CMU License](https://github.com/python-pillow/Pillow/tree/main?tab=License-1-ov-file#readme)
* scikit-image
   * https://github.com/scikit-image/scikit-image
   * [BSD-3-Clause](https://github.com/scikit-image/scikit-image?tab=License-1-ov-file)
* streamlit
   * https://github.com/streamlit/streamlit/tree/develop
   * [Apache License 2.0](https://github.com/streamlit/streamlit/tree/develop?tab=Apache-2.0-1-ov-file#readme)
* streamlit-drawable-canvas
   * https://github.com/andfanilo/streamlit-drawable-canvas
   * [MIT License](https://github.com/andfanilo/streamlit-drawable-canvas?tab=MIT-1-ov-file#readme)
* streamlit-image-coordinates
   * https://github.com/blackary/streamlit-image-coordinates/
   * [MIT License](https://github.com/blackary/streamlit-image-coordinates/?tab=MIT-1-ov-file#readme)