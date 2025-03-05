import cv2
import numpy as np
import os
from multiprocessing import Pool
from multiprocessing import Process
import math

class Movie:
    scale = 1
    warp_type = cv2.MOTION_HOMOGRAPHY
    warp = np.eye(3,3,dtype=np.float32)
    warpTransform = cv2.warpPerspective

    def __init__(self):
        pass

    def Read(self, fPath):
        fName = os.path.splitext(os.path.basename(fPath))[0]
        outputDir = "../data/output"
        if not os.path.exists("../data/output"):
            os.makedirs(outputDir)

        self.outputPath = outputDir + "/" + fName + "_stabilization.mp4"
        self.inputVideo = cv2.VideoCapture(fPath)
        self.fps = self.inputVideo.get(cv2.CAP_PROP_FPS)
        self.size = (int(self.inputVideo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.inputVideo.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.totalFrameNum = int(self.inputVideo.get(cv2.CAP_PROP_FRAME_COUNT))
        self.codec = int(self.inputVideo.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("utf-8").upper()
        self.fourcc = "mp4v" if self.codec == "H264" else "other"
        print(f"fps: {self.fps}")
        print(f"(width, height): {self.size}")
        print(f"total flame num: {self.totalFrameNum}")
        print(f"codec: {self.codec}")
    
    def CreateFrames(self):
        ret, base = self.inputVideo.read()

        self.size = tuple(map(lambda x: int(x * Movie.scale), self.size))
        print(f"scaled (width, height): {self.size}")

        base = cv2.resize(base, self.size)
        self.base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

        if self.fourcc != "mp4v":
            print("invaild codec.")
            return

        self.outputVideo = cv2.VideoWriter(self.outputPath, cv2.VideoWriter_fourcc(*self.fourcc), self.fps, self.size)

        progressCount = 0

        self.dstImages = np.zeros((self.totalFrameNum, self.size[1], self.size[0], 3), dtype=np.uint8)
        #print(dstImages)
        print(self.dstImages.shape)

        self.inputVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.frames = [self.inputVideo.read() for i in range(self.totalFrameNum)]

    def Stabilize(self):
        ##############################################
        for i in range(self.totalFrameNum):
            if i == 0 or i % 10 == 0 or i == self.totalFrameNum - 1:
                print(f"progressCount: {i + 1} / {self.totalFrameNum}")
            tmp = self._CreateStablizeImage(i)
            self.dstImages[i] = tmp[1]
            #print(tmp[0])
        ##############################################

    def CreateOutputVideo(self):
        for i in range(self.totalFrameNum):
            self.outputVideo.write(self.dstImages[i])
        
        print("finish.")

    def _CreateStablizeImage(self, progressCount):
        frame = cv2.resize(self.frames[progressCount][1], self.size)
        tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.findTransformECC(tmp, self.base, Movie.warp, Movie.warp_type)

        return progressCount, Movie.warpTransform(frame, Movie.warp, self.size)
    
    def GetBase(self):
        return self.base
    
    def GetSize(self):
        return self.size
    
    def GetFrames(self):
        return self.frames
    
    def GetTotalFrameNum(self):
        return self.totalFrameNum

    def Release(self):
        self.inputVideo.release()
        self.outputVideo.release()
        cv2.destroyAllWindows()
    
    def SetDstImages(self, dstImages):
        self.dstImages = dstImages

    """
    def SplitFrame(self):
        outputDir = "../data/output/frame"
        if not os.path.exists("../data/output/frame"):
            os.makedirs(outputDir)
        outputFileName = 'image_wave-%s.png'

        self.inputVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)

        top = 460
        bottom = 660
        left = 160
        right = 560
        for i in range(self.totalFrameNum):
            if i == 0 or i % 10 == 0 or i == self.totalFrameNum - 1:
                print(f"progressCount: {i + 1} / {self.totalFrameNum}")
            flag, frame = self.inputVideo.read()
            if flag == False:
                break
            cv2.imwrite(outputDir + "/" + outputFileName % str(i).zfill(6), frame[top : bottom, left : right])

        self.inputVideo.release()
    """
    def SplitFrame(self, frame):
        #outputDir = "../data/output/frame"
        #if not os.path.exists("../data/output/frame"):
        #    os.makedirs(outputDir)
        #outputFileName = 'image_wave-%s.png'

        self.inputVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.image= []
        top, bottom, left, right = frame.get()
        width = right - left
        height = bottom - top
        for i in range(self.totalFrameNum):
            if i == 0 or i % 10 == 0 or i == self.totalFrameNum - 1:
                print(f"progressCount: {i + 1} / {self.totalFrameNum}")
            flag, frame = self.inputVideo.read()
            if flag == False:
                break
            #cv2.imwrite(outputDir + "/" + outputFileName % str(i).zfill(6), frame[top : bottom, left : right])

            roi_frame = frame[top:top+height, left:left+width]  # 指定範囲のフレームを抽出
            
            self.image.append(roi_frame)
        self.inputVideo.release()

    def GetImage(self):
        return self.image

class MaltiProcess:
    warp_type = cv2.MOTION_HOMOGRAPHY
    warp = np.eye(3,3,dtype=np.float32)
    warpTransform = cv2.warpPerspective

    def RunMultiProcess(self):
        """
        p = Pool(4)
        args = list(range(16))
        res = p.map(self.square, args)
        print(res)
        """
        p = Pool(4)
        for i in range(16):
            res = p.apply_async(self._pp, (i,))
        p.close()
        p.join()

    def _pp(self, progressCount):
        print(progressCount)
    
    def square(self, n):
        return n * n
    
    def SetBase(self, base):
        self.base = base
    
    def SetSize(self, size):
        self.size = size
    
    def SetFrames(self, frames):
        self.frames = frames
    
    def SetTotalFrameNum(self, totalFrameNum):
        self.totalFrameNum = totalFrameNum
    
    def Stabilize(self):
        self.dstImages = np.zeros((self.totalFrameNum, self.size[1], self.size[0], 3), dtype=np.uint8)
        
        
        # Poolを使う場合
        processNum = 4
        p = Pool(processNum)
        chunkSize = math.ceil(self.totalFrameNum / processNum)
        print(chunkSize)
        args = list(range(self.totalFrameNum))
        
        for r in p.imap_unordered(self._CreateStablizeImage, args, chunkSize):
            self.dstImages[r[0]] = r[1]
        
        """
        # asyncを使う場合
        p = Pool(8)
        for i in range(self.totalFrameNum):
            res = p.apply_async(self._CreateStablizeImage, (i,))
        p.close()
        p.join()
        """

        """
        for i in range(self.totalFrameNum):
            res = self._CreateStablizeImage(i)
            self.dstImages[res[0]] = res[1]
        """
    
    def _CreateStablizeImage(self, progressCount):
        if progressCount == 0 or progressCount % 10 == 0 or progressCount == self.totalFrameNum - 1:
            print(f"progressCount: {progressCount + 1} / {self.totalFrameNum}")
        frame = cv2.resize(self.frames[progressCount][1], self.size)
        tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.findTransformECC(tmp, self.base, MaltiProcess.warp, MaltiProcess.warp_type)

        return progressCount, MaltiProcess.warpTransform(frame, MaltiProcess.warp, self.size)
    
    def GetDstImages(self):
        return self.dstImages