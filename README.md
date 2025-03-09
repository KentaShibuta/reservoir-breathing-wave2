# reservoir-breathing-wave2
* ESN(Echo State Network)を用いて、動物の安静時の動画から呼吸数を求めるプログラム
* 開発環境は、dockerフォルダにて下記コマンドを実行してコンテナを起動する
   - ```
     docker compose up -d 
     ```
* RSNの学習と推論は、下記文献の付録にあるコードを参考にした
   - リザバーコンピューティング: 時系列パターン認識のための高速機械学習の理論とハードウェア
      - https://www.morikita.co.jp/books/mid/085531
   - 付録のコードは入力と出力がOne to Oneであり、これをMany to Oneに変更した
   - 推論処理は、C++で書いたコードをpybind11で共有ライブラリ化したものに置き換えている
      - 推論の行列とベクトルの積は、OpenMPを用いて並列化している
      - 作成した共有ライブラリは、付録の推論処理と同等の結果を出力することを確認済み
      - 作成した共有ライブラリは、Pythonのスクリプトから呼び出して使用可能
      - 付録のコードとC++で書いた共有ライブラリで実行時間を比較するとC++の方が付録のコードを半分の時間で推論結果を得られた
      - 行列の固有値計算には、Eigenを使用している
         - https://eigen.tuxfamily.org/index.php?title=Main_Page
   - 近日中に、学習処理もC++で書いた共有ライブラリに置き換える予定
* main.py中の変数isTrainについて
   - isTrainをTrueにするとESNの学習が実行される
      - 学習で使用する動画ファイルは、デフォルトで1つdataフォルダに保存している
      - 動画ファイルをフレーム画像に分割し、フレーム画像から、ESNに与えるデータセットを作成する
         - データセットの作成時に、フレーム画像をトリミングしている
         - トリミングの範囲は、デフォルトの動画ファイルに合わせて、コードに定数で埋め込んでいる
            - (学習と推論実行時に、入力された任意の動画ファイルに合わせて、トリミング範囲を設定できるよう改良する)
      - 学習済みモデルは、modelフォルダにpickleファイルが作成される
   - isTrainをFalseにするとESNの推論が実行される
      - デフォルトではリポジトリに保存している学習済みモデルを使用して推論する
* C++で書き直したESNモジュールのビルド方法
   - Dockerコンテナにアタッチした後に、下記コマンドを実行する
      - ```
        cd /root/app/src/
        ```
      - ```
        g++ -O3 -Wall -shared -std=c++23 -I /root/app/lib -fopenmp -fPIC -fvisibility=hidden `python3 -m pybind11 --includes` esn.cpp -o esn`python3-config --extension-suffix`
        ```
