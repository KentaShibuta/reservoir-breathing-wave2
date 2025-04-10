import streamlit as st
import cv2
from PIL import Image
import numpy as np
import cv2

st.title("画像の矩形選択とトリミング")

# 動画アップロード
uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # 動画ファイルを OpenCV で読み込み
    tfile = open("temp_video.mp4", "wb")
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("動画の読み込みに失敗しました。")
    else:
        # BGR → RGB に変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        image_np = np.array(image)
        st.image(image, caption="元の画像", use_container_width=False)

        st.write("トリミングしたい領域を選択してください。")

        # 左上と右下の座標を入力（手動選択）
        x1 = st.number_input("左上のX", min_value=0, max_value=image_np.shape[1], value=0)
        y1 = st.number_input("左上のY", min_value=0, max_value=image_np.shape[0], value=0)
        x2 = st.number_input("右下のX", min_value=0, max_value=image_np.shape[1], value=image_np.shape[1])
        y2 = st.number_input("右下のY", min_value=0, max_value=image_np.shape[0], value=image_np.shape[0])

        if x1 < x2 and y1 < y2:
            cropped_image = image_np[int(y1):int(y2), int(x1):int(x2)]
            st.image(cropped_image, caption="トリミングされた画像", use_container_width=False)

            # 画像保存オプション
            save = st.button("トリミング画像を保存")
            if save:
                cropped_pil = Image.fromarray(cropped_image)
                cropped_pil.save("cropped_image.png")
                st.success("cropped_image.png を保存しました")
        else:
            st.warning("X1 < X2 かつ Y1 < Y2 になるようにしてください。")