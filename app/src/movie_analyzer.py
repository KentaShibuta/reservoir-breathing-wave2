from movie import Movie
from movie import MaltiProcess
import time
import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks
import datetime
import csv
from sklearn.preprocessing import MinMaxScaler
import pickle
from PIL import Image

class Frame:
    def __init__(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
    
    def get(self):
        return self.top, self.bottom, self.left, self.right


class MovieAnalyzer:
    def __init__(self, movie_file):
        
        self.movie = Movie()
        self.frame = Frame(top=460, bottom=660, left=160, right=560)
        self.movie.Read(movie_file)
        self.movie.SplitFrame(self.frame)

        self.image = self.movie.GetImage()
    
    def CreateBreathingWave(self, compression_ratio, show = False):
        print("[START] getting breathing wave")

        # グレースケールで画像データを取得
        #grayImages = [cv2.imread(files[i], cv2.IMREAD_GRAYSCALE) for i in range(len(files))]
        grayImages = [cv2.cvtColor(self.image[i], cv2.COLOR_BGR2GRAY) for i in range(self.movie.GetTotalFrameNum())]


        height, width = grayImages[0].shape
        print(f"image num: {len(grayImages)}")
        print(f"original width: {width}")
        print(f"original height: {height}")
        print(f"number of original pixels: {width * height}")

        # 画像サイズを圧縮
        grayImages = [cv2.resize(grayImages[i] , (int(width*compression_ratio), int(height*compression_ratio))) for i in range(self.movie.GetTotalFrameNum())]
        height, width = grayImages[0].shape
        print(f"number of comprssed pixels: {width * height}")

        # 2値化
        th = [cv2.threshold(grayImages[i], 100, 255,cv2.THRESH_BINARY)[1] // 255 for i in range(len(grayImages))]
        th_array = np.array(th)

        # 特徴量に変換
        # 画像ごとに全ピクセルの2値化の値の和を計算
        breathing_wave = [np.sum(th[i]) for i in range(len(th))]



        # 正規化
        scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
        breathing_wave = np.array(breathing_wave)
        breathing_wave = scaler.fit_transform(breathing_wave.reshape(-1, 1)).flatten()

        peaks, _ = find_peaks(breathing_wave, distance=30)
        print(f"peakNum: {len(peaks)}")

        print("[END] getting breathing wave")

        if show == True:
            x = range(len(th))
            plt.plot(x, breathing_wave)

            plt.xlabel("time step")
            plt.ylabel("breathing wave")
            plt.show()

        return th_array, breathing_wave
    
    def CreateColorHistogram(self, compression_ratio):
        print("[START] getting color histogram")

        # カラーで画像データを取得
        #colorImages = [cv2.imread(files[i]) for i in range(len(files))]
        colorImages = self.image
        height, width, channel = colorImages[0].shape
        print(f"image num: {len(colorImages)}")
        print(f"original width: {width}")
        print(f"original height: {height}")
        print(f"number of original pixels: {width * height}")

        # 画像サイズを圧縮
        colorImages = [cv2.resize(colorImages[i] , (int(width*compression_ratio), int(height*compression_ratio))) for i in range(self.movie.GetTotalFrameNum())]
        height, width, channel = colorImages[0].shape
        print(f"number of comprssed pixels: {width * height}")

        color_array = np.array(colorImages)
        print(color_array[0])
        color_array = np.array(colorImages, dtype=np.float32)
        print(color_array[0])
        print(f"color_array.shape:{color_array.shape}")

        print("[END] getting color histogram")

        return color_array
    
    def GetGray(self, show = False):
        # フレーム間差分法
        compression_ratio = 0.5
        #files = glob.glob("../data/output/frame/image_wave-*.png")
        #files.sort()

        image_array, breathing_wave = self.CreateBreathingWave(compression_ratio, show)

        image_array = np.concatenate((image_array.reshape(image_array.shape[0], -1), breathing_wave.reshape(-1, 1)), axis=1)
        print(image_array.shape)
        
        now = datetime.datetime.now()
        filename = 'data/' + now.strftime('%Y%m%d_%H%M%S') + '.pickle'
        with open(filename, mode='wb') as fo:
            pickle.dump(image_array, fo)

    def GetColor(self, show = False, save = False, isTrain = True):
        # フレーム間差分法
        compression_ratio = 0.5
        #files = glob.glob("../data/output/frame/image_wave-*.png")
        #files.sort()

        _, breathing_wave = self.CreateBreathingWave(compression_ratio, show) if isTrain == True else (None, None)
        image_array = self.CreateColorHistogram(compression_ratio)
        image_array /= 255.0
        
        print(image_array.shape)
        print(image_array[0, 0].dtype)

        if isTrain == True:
            image_array = np.concatenate((image_array.reshape(image_array.shape[0], -1), breathing_wave.reshape(-1, 1)), axis=1)
            print(image_array.shape)
        else:
            image_array = image_array.reshape(image_array.shape[0], -1)

        if save == True:
            now = datetime.datetime.now()
            filename = '../data/' + now.strftime('%Y%m%d_%H%M%S') + '.pickle'
            with open(filename, mode='wb') as fo:
                pickle.dump(image_array, fo)
        
        return image_array

    """
    def Stabilize(fName, isMulti = False):
        movie = Movie()
        movie.Read(fName)
        movie.CreateFrames()

        if isMulti:
            #マルチプロセス
            base = movie.GetBase()
            size = movie.GetSize()
            frames = movie.GetFrames()
            totalFrameNum = movie.GetTotalFrameNum()
            mulp = MaltiProcess()
            mulp.SetBase(base)
            mulp.SetSize(size)
            mulp.SetFrames(frames)
            mulp.SetTotalFrameNum(totalFrameNum)
            mulp.Stabilize()
            dstImages = mulp.GetDstImages()
            movie.SetDstImages(dstImages)
        else:
            # シングルプロセス
            movie.Stabilize()

        movie.CreateOutputVideo()
        movie.Release()
    """


"""
# 動画をフレームごとに画像に変換
fMovieName = "../data/output/20241227_sophie_1_stabilization.mp4"
#fMovieName = "../data/input/20241227_sophie_1.mp4"
GetFrame(fMovieName)


#GetGray(show=True)
GetColor(show=True)
"""