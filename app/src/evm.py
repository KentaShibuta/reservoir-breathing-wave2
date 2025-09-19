import os
from glob import glob
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import scipy.signal as signal
import json
import sys

class Frame:
    def __init__(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
    
    def get(self):
        return self.top, self.bottom, self.left, self.right

# 動画読み込み
DATA_PATH = "/root/app/data"
#VIDEO_NAME = "20241227_sophie_1_stabilization.mp4"
VIDEO_NAME = "IMG_1693.mp4"
VIDEO_PATH = os.path.join(DATA_PATH, VIDEO_NAME)
os.path.exists(VIDEO_PATH)

# ハイパーパラメータの設定
# video magnification factor
ALPHA = 50.0
# Gaussian Pyramid Level of which to apply magnfication
LEVEL = 4
# Temporal Filter parameters
#f_lo = 50/60
#f_hi = 60/60
f_lo = 10/60
f_hi = 50/60
# OPTIONAL: override fs
MANUAL_FS = None
VIDEO_FS = None
# video frame scale factor
SCALE_FACTOR = 1.0

# Colorspace Functions
def rgb2yiq(rgb):
    """ Converts an RGB image to YIQ using FCC NTSC format.
        This is a numpy version of the colorsys implementation
        https://github.com/python/cpython/blob/main/Lib/colorsys.py
        Inputs:
            rgb - (N,M,3) rgb image
        Outputs
            yiq - (N,M,3) YIQ image
        """
    # compute Luma Channel
    y = rgb @ np.array([[0.30], [0.59], [0.11]])

    # subtract y channel from red and blue channels
    rby = rgb[:, :, (0,2)] - y

    i = np.sum(rby * np.array([[[0.74, -0.27]]]), axis=-1)
    q = np.sum(rby * np.array([[[0.48, 0.41]]]), axis=-1)

    yiq = np.dstack((y.squeeze(), i, q))
    
    return yiq

def bgr2yiq(bgr):
    """ Coverts a BGR image to float32 YIQ """
    # get normalized YIQ frame
    rgb = np.float32(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    yiq = rgb2yiq(rgb)

    return yiq

def yiq2rgb(yiq):
    """ Converts a YIQ image to RGB.
        Inputs:
            yiq - (N,M,3) YIQ image
        Outputs:
            rgb - (N,M,3) rgb image
        """
    r = yiq @ np.array([1.0, 0.9468822170900693, 0.6235565819861433])
    g = yiq @ np.array([1.0, -0.27478764629897834, -0.6356910791873801])
    b = yiq @ np.array([1.0, -1.1085450346420322, 1.7090069284064666])
    rgb = np.clip(np.dstack((r, g, b)), 0, 1)
    return rgb

# ガウシアンピラミッド
def gaussian_pyramid(image, level):
    """ Obtains single band of a Gaussian Pyramid Decomposition
        Inputs: 
            image - single channel input image
            num_levels - number of pyramid levels
        Outputs:
            pyramid - Pyramid decomposition tensor
        """ 
    rows, cols, colors = image.shape
    scale = 2**level
    pyramid = np.zeros((colors, rows//scale, cols//scale))

    for i in range(0, level):
        # image = cv2.pyrDown(image)

        image = cv2.pyrDown(image, dstsize=(cols//2, rows//2))
        rows, cols, _ = image.shape

        if i == (level - 1):
            for c in range(colors):
                pyramid[c, :, :] = image[:, :, c]

    return pyramid

# -----------------------------------------------------------------

inv_colorspace = lambda x: cv2.normalize(
    yiq2rgb(x), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)

# Get Video Frames
frames = [] # frames for processing
cap = cv2.VideoCapture(VIDEO_PATH)

# video sampling rate
fs = cap.get(cv2.CAP_PROP_FPS)

idx = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        break

    rect_file = "/root/app/data/input_rect.json"
    if os.path.exists(rect_file):
        with open(rect_file) as f:
            rect = json.load(f)
            ROI = Frame(top=rect["y1"], bottom=rect["y2"], left=rect["x1"], right=rect["x2"])
    else:
        print(f"{rect_file} は存在しません。")
        sys.exit(1)

    top, bottom, left, right = ROI.get()
    width = right - left
    height = bottom - top
    frame = frame[top:top+height, left:left+width]

    if idx == 0:
        og_h, og_w, _ = frame.shape
        w = int(og_w*SCALE_FACTOR)
        h = int(og_h*SCALE_FACTOR)

    # convert normalized uint8 BGR to the desired color space
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = bgr2yiq(np.float32(frame/255))

    # append resized frame
    frames.append(cv2.resize(frame, (w, h)))

    idx += 1
    
    
cap.release()
cv2.destroyAllWindows()
del cap

NUM_FRAMES = len(frames)
print(f"frames:{NUM_FRAMES}")

print(f"Detected Video Sampling rate: {fs}")

if MANUAL_FS:
    print(f"Overriding to: {MANUAL_FS}")
    fs = MANUAL_FS
    VIDEO_FS = fs
else:
    VIDEO_FS = fs

# 時間フィルタの定義
bandpass = signal.firwin(numtaps=NUM_FRAMES,
                         cutoff=(f_lo, f_hi),
                         fs=fs,
                         pass_zero=False)
transfer_function = np.fft.fft(np.fft.ifftshift(bandpass))

# ガウシアンピラミッド
rows, cols, colors = frames[0].shape
scale = 2**LEVEL
pyramid_stack = np.zeros((NUM_FRAMES, colors, rows//scale, cols//scale))

for i, frame in enumerate(frames):
    pyramid = gaussian_pyramid(frame, LEVEL)
    pyramid_stack[i, :, :, :] = pyramid

# 時間フィルタを適用
pyr_stack_fft = np.fft.fft(pyramid_stack, axis=0).astype(np.complex64)
_filtered_pyramid = pyr_stack_fft * transfer_function[:, None, None, None].astype(np.complex64)
filtered_pyramid = np.fft.ifft(_filtered_pyramid, axis=0).real

# 増幅してビデオを再構築
magnified_pyramid = filtered_pyramid * ALPHA

magnified = []
magnified_only = []

for i in range(NUM_FRAMES):
    y_chan = frames[i][:, :, 0]
    i_chan = frames[i][:, :, 1] 
    q_chan = frames[i][:, :, 2] 
    
    fy_chan = cv2.resize(magnified_pyramid[i, 0, :, :], (cols, rows))
    fi_chan = cv2.resize(magnified_pyramid[i, 1, :, :], (cols, rows))
    fq_chan = cv2.resize(magnified_pyramid[i, 2, :, :], (cols, rows))

    # apply magnification
    mag = np.dstack((
        y_chan + fy_chan,
        i_chan + fi_chan,
        q_chan + fq_chan,
    ))
    
    # normalize and convert to RGB
    mag = inv_colorspace(mag)

    # store magnified frames
    magnified.append(mag)

    # store magified only for reference
    magnified_only.append(np.dstack((fy_chan, fi_chan, fq_chan)))


og_reds = []
og_blues = []
og_greens = []

reds = []
blues = []
greens = []
for i in range(NUM_FRAMES):
    # convert YIQ to RGB
    frame = inv_colorspace(frames[i])
    og_reds.append(frame[0, :, :].sum())
    og_blues.append(frame[1, :, :].sum())
    og_greens.append(frame[2, :, :].sum())

    reds.append(magnified[i][0, :, :].sum())
    blues.append(magnified[i][1, :, :].sum())
    greens.append(magnified[i][2, :, :].sum())

times = np.arange(0, NUM_FRAMES)/fs

fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
ax[0].plot(times, og_reds, color='red')
ax[0].plot(times, og_blues, color='blue')
ax[0].plot(times, og_greens, color='green')
ax[0].set_title("Original", size=18)
ax[0].set_xlabel("Time", size=16)
ax[0].set_ylabel("Intensity", size=16)

ax[1].plot(times, reds, color='red')
ax[1].plot(times, blues, color='blue')
ax[1].plot(times, greens, color='green')
ax[1].set_title("Filtered", size=18)
ax[1].set_xlabel("Time", size=16)

# グラフを表示
plt.tight_layout()
plt.show()


# 増幅後のみの動画
# get width and height for video frames
_h, _w, _ = magnified_only[-1].shape

# save to mp4
out = cv2.VideoWriter(f"stacked_{int(ALPHA)}x_AMP.mp4",
                      cv2.VideoWriter_fourcc(*'MP4V'), 
                      int(fs), 
                      (_w, _h))

sums = []
for frame in magnified:
    sums.append(frame.sum(axis=1).sum(axis=0))
    #frame = yiq2rgb(frame)
    #frame = cv2.cvtColor(
    #    cv2.normalize(frame*20, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
    #    cv2.COLOR_RGB2BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)

out.release()
del out


"""
stacked_frames = []
middle = np.zeros((rows, 3, 3)).astype(np.uint8)

for vid_idx in range(NUM_FRAMES):
    og_frame = cv2.normalize(yiq2rgb(frames[vid_idx]), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
    frame = np.hstack((cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR), 
                       middle, 
                       cv2.cvtColor(magnified[vid_idx], cv2.COLOR_RGB2BGR)))
    stacked_frames.append(frame)

# get width and height for video frames
_h, _w, _ = stacked_frames[-1].shape

# save to mp4
out = cv2.VideoWriter(f"stacked_{int(ALPHA)}x.mp4",
                      cv2.VideoWriter_fourcc(*'MP4V'), 
                      int(fs), 
                      (_w, _h))
 
for frame in stacked_frames:
    out.write(frame)

out.release()
del out
"""