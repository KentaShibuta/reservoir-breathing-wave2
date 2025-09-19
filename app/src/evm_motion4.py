import os
from glob import glob
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import scipy.signal as signal
import json
import sys
from scipy.signal import butter, filtfilt

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
#VIDEO_NAME = "IMG_1693.mp4" # 90～110で切り取る ROI {"x1": 0, "y1": 117, "x2": 706, "y2": 1266}
#VIDEO_NAME = "IMG_1055.mp4"
#VIDEO_NAME = "770125907.349203.mp4"
#VIDEO_NAME = "770125907.349203_stabilization.mp4" # 20～40 ROI {"x1": 146, "y1": 452, "x2": 636, "y2": 790}
#VIDEO_NAME = "770125907.235399.mp4"
VIDEO_NAME = "770125907.235399_stabilization.mp4" # 5～25 ROI {"x1": 250, "y1": 523, "x2": 564, "y2": 708}
# 動画のフレーム画像を中心でy方向にスライスしてそれを時間軸で並べた画像における、y方向のトリミング範囲top～bottom
y_crop_from_center_strip_top = 5 # 
y_crop_from_center_strip_bottom = 25

#VIDEO_NAME = "baby.mp4"
VIDEO_PATH = os.path.join(DATA_PATH, VIDEO_NAME)
os.path.exists(VIDEO_PATH)

# ハイパーパラメータの設定
# video magnification factor
ALPHA = 10.0
# Gaussian Pyramid Level of which to apply magnfication
LEVEL = 4
# Temporal Filter parameters
#f_lo = 18/60
#f_hi = 240/60
f_lo = 10/60
f_hi = 300/60
# OPTIONAL: override fs
MANUAL_FS = None
VIDEO_FS = None
# video frame scale factor
SCALE_FACTOR = 0.3

# ラプラシアンピラミッド(最後のレベルだけ)
def laplacian_pyramid(image, level):
    rows, cols, colors = image.shape
    scale = 2**level
    pyramid = np.zeros((colors, rows//scale, cols//scale))

    current = image.copy()
    for i in range(0, level+1):
        down = cv2.pyrDown(current, dstsize=(cols//2, rows//2))
        up = cv2.pyrUp(down, dstsize=current.shape[:2][::-1])
        lap = current - up
        

        if i == (level):
            for c in range(colors):
                pyramid[c, :, :] = lap[:, :, c]
        
        current = down
        rows, cols, _ = current.shape

    return pyramid

# ラプラシアンピラミッド(全てのレベルを保持する)
def laplacian_pyramid_all_level(image, level):
    rows, cols, colors = image.shape
    scale = 2**level
    pyramid = []

    current = image.copy()
    for i in range(0, level):
        down = cv2.pyrDown(current, dstsize=(cols//2, rows//2))
        up = cv2.pyrUp(down, dstsize=current.shape[:2][::-1])
        lap = current - up
        
        pyramid.append(lap)
        
        current = down
        rows, cols, _ = current.shape
    
    pyramid.append(current)

    return pyramid

# ピラミッドを復元
def reconstruct_from_laplacian(pyramid):
    current = pyramid[-1]
    for i in reversed(range(len(pyramid) - 1)):
        upsampled = cv2.pyrUp(current, dstsize=pyramid[i].shape[:2][::-1])
        current = cv2.add(upsampled, pyramid[i])
    return current

def temporal_bandpass_filter(num_frames, data, f_lo, f_hi, fs):
    # 時間フィルタの定義
    """
    bandpass = signal.firwin(numtaps=num_frames,
                            cutoff=(f_lo, f_hi),
                            fs=fs,
                            pass_zero=False)
    transfer_function = np.fft.fft(np.fft.ifftshift(bandpass))

    pyr_stack_fft = np.fft.fft(data, axis=0).astype(np.complex64)
    _filtered_pyramid = pyr_stack_fft * transfer_function[:, None, None, None].astype(np.complex64)
    filtered_pyramid = np.fft.ifft(_filtered_pyramid, axis=0).real
    """
    b, a = butter(N=2, Wn=[f_lo, f_hi], fs=fs, btype='band')

    # フィルタ適用
    filtered_pyramid = filtfilt(b, a, data, axis=0)

    return filtered_pyramid

# -----------------------------------------------------------------

# Get Video Frames
frames = [] # frames for processing
cap = cv2.VideoCapture(VIDEO_PATH)

# video sampling rate
fs = cap.get(cv2.CAP_PROP_FPS)

# Get ROI
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

idx = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        break

    # crop frame by ROI
    frame = frame[top:top+height, left:left+width]

    if idx == 0:
        og_h, og_w, _ = frame.shape
        w = int(og_w*SCALE_FACTOR)
        h = int(og_h*SCALE_FACTOR)

    # convert BGR to YUV
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # convert normalized uint8 BGR to the desired color space
    frame_normalize = np.float32(frame_yuv/255)

    # append resized frame
    frames.append(cv2.resize(frame_normalize, (w, h)))

    idx += 1

    # 1800フレームで打ち切り(FPS = 30の場合は60 sec分)
    if idx == 1800:
        break
    
    
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

# ガウシアンピラミッド
rows, cols, colors = frames[0].shape
scale = 2**LEVEL
pyramid_stack = [laplacian_pyramid_all_level(frame, LEVEL) for frame in frames]

# 各レベルを時間方向にまとめる
filtered = []
for level in range(LEVEL):
    level_frames = np.array([p[level] for p in pyramid_stack])
    print(f"Filtering level {level}...")
    filtered_level = temporal_bandpass_filter(NUM_FRAMES, level_frames, f_lo, f_hi, fs)
    #per_level_alpha = ALPHA / (2 ** level)
    per_level_alpha = ALPHA
    #per_level_alpha = ALPHA * (2 ** level) / (2 ** (LEVEL-1))
    amplified = filtered_level * per_level_alpha
    filtered.append(amplified)

# 再構成
print("Reconstructing frames...")
magnified = []
for i in range(len(frames)):
    new_pyramid = []
    for level in range(LEVEL):
        original = pyramid_stack[i][level]

        
        if (level < 2):
            new_pyramid.append(original)
        else:
            amplified = original.copy()
            # Y成分（チャンネル0）のみ増幅
            amplified[:, :, 0] += filtered[level][i][:, :, 0]  # Y only
            new_pyramid.append(amplified)
        
        #new_pyramid.append(original)

    new_pyramid.append(pyramid_stack[i][-1])  # 最終レベルはそのまま
    reconstructed = reconstruct_from_laplacian(new_pyramid)
    magnified.append(np.clip(reconstructed * 255.0, 0, 255).astype(np.uint8))

# 0～255のスケール, BGRに戻す
og_frames = []
mag_frames = []
for vid_idx in range(NUM_FRAMES):
    og_frame = np.clip(frames[vid_idx] * 255.0, 0, 255).astype(np.uint8)
    og_frames.append(cv2.cvtColor(og_frame, cv2.COLOR_YUV2BGR))
    mag_frames.append(cv2.cvtColor(magnified[vid_idx], cv2.COLOR_YUV2BGR))

og_gray_frames = []
mag_gray_frames = []
for vid_idx in range(NUM_FRAMES):
    og_gray_frames.append(cv2.cvtColor(og_frames[vid_idx], cv2.COLOR_BGR2GRAY))
    mag_gray_frames.append(cv2.cvtColor(mag_frames[vid_idx], cv2.COLOR_BGR2GRAY))

# グラフ作成
og_grays = []
grays = []
for i in range(NUM_FRAMES):
    og_grays.append(og_gray_frames[i][:, :].mean())
    grays.append(mag_gray_frames[i][:, :].mean())
    

times = np.arange(0, NUM_FRAMES)/fs

fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
ax[0].plot(times, og_grays, color='gray')
ax[0].set_title("Original", size=18)
ax[0].set_xlabel("Time", size=16)
ax[0].set_ylabel("Intensity", size=16)

ax[1].plot(times, grays, color='gray')
ax[1].set_title("Filtered", size=18)
ax[1].set_xlabel("Time", size=16)

# グラフを表示
plt.tight_layout()
plt.show()

# フィルターする
bandpass = signal.firwin(numtaps=NUM_FRAMES,
                        cutoff=(f_lo, f_hi),
                        fs=fs,
                        pass_zero=False)
transfer_function = np.fft.fft(np.fft.ifftshift(bandpass))

grays_fft = np.fft.fft(np.array(grays), axis=0).astype(np.complex64)
_filtered_grays = grays_fft * transfer_function
filtered_grays = np.fft.ifft(_filtered_grays, axis=0).real

plt.plot(times, filtered_grays, color='gray')
plt.show()

# 動画から縦1列を取り出して、時系列でつなげる(元画像)
center_x = cols // 2
column_images = []
for i in range(NUM_FRAMES):
    # 中央列（縦1列）を抽出 → shape=(height,)
    column = og_gray_frames[i][:, center_x]

    # 縦ベクトルを列ベクトルに変換 → shape=(height, 1)
    column = column.reshape(-1, 1)

    # リストに追加
    column_images.append(column)

# 横に連結（x方向＝時間、y方向＝空間）
result = np.hstack(column_images)

split_horizon = []
for i in range(result.shape[1]):
    split_horizon.append(result[5:25, i].sum())

# 表示 or 保存
cv2.imshow('Space-Time Slice (Center Column)', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 動画から縦1列を取り出して、時系列でつなげる(増幅後)
column_images_amp = []
for i in range(NUM_FRAMES):
    # 中央列（縦1列）を抽出 → shape=(height,)
    column_amp = mag_gray_frames[i][:, center_x]

    # 縦ベクトルを列ベクトルに変換 → shape=(height, 1)
    column_amp = column_amp.reshape(-1, 1)

    # リストに追加
    column_images_amp.append(column_amp)

# 横に連結（x方向＝時間、y方向＝空間）
result_amp = np.hstack(column_images_amp)

split_horizon_amp = []
for i in range(result_amp.shape[1]):
    split_horizon_amp.append(result_amp[y_crop_from_center_strip_top:y_crop_from_center_strip_bottom, i].sum())

# 表示 or 保存
cv2.imshow('Space-Time Slice (Center Column)', result_amp)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.plot(times, split_horizon, color='red')
plt.plot(times, split_horizon_amp, color='blue')
plt.show()

# フレームを結合して動画を作成
stacked_frames = []
middle = np.zeros((rows, 3)).astype(np.uint8)
for vid_idx in range(NUM_FRAMES):
    frame = np.hstack((og_gray_frames[vid_idx], 
                       middle, 
                       mag_gray_frames[vid_idx]))
    stacked_frames.append(frame)

# get width and height for video frames
_h, _w = stacked_frames[-1].shape

# save to mp4
out = cv2.VideoWriter(f"stacked_{int(ALPHA)}x.mp4",
                      cv2.VideoWriter_fourcc(*'MP4V'), 
                      int(fs), 
                      (_w, _h))
 
for frame in stacked_frames:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    out.write(frame_bgr)

out.release()
del out
