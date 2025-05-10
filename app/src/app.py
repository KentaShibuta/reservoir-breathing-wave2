import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from PIL import Image
import numpy as np
import cv2
import json

st.title("画像の矩形選択とトリミング")

# 動画アップロード
uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # 動画ファイルを OpenCV で読み込み
    tfile = open("/root/app/data/input_video.mp4", "wb")
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture("/root/app/data/input_video.mp4")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("動画の読み込みに失敗しました。")
    else:
        # BGR → RGB に変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        image_np = np.array(image)
        orig_width = image_np.shape[1]
        orig_height = image_np.shape[0]

        display_width = 600
        scale = display_width / orig_width
        display_height = int(orig_height * scale)

        # セッションステートで画像管理（初回のみ初期化）
        if "current_image" not in st.session_state:
            st.session_state.origin_image = image_np
            st.session_state.current_image = image_np
            st.session_state.reload_count = 0

        # NumPy画像をリサイズ
        image_resized = cv2.resize(st.session_state.current_image, (display_width, display_height))

        coords = streamlit_image_coordinates(
                                                image_resized,
                                                key="numpy",
                                                use_column_width=display_width,
                                                click_and_drag=True,
                                            )
        # 座標を補正
        if coords is not None:
            if coords and "x1" in coords and "y1" in coords and "x2" in coords and "y2" in coords:
                scale2 = orig_width / display_width
                corrected_x1 = int(coords["x1"] * scale2) if coords["x1"] <= coords["x2"] else int(coords["x2"] * scale2)
                corrected_y1 = int(coords["y1"] * scale2) if coords["y1"] <= coords["y2"] else int(coords["y2"] * scale2)
                corrected_x2 = int(coords["x2"] * scale2) if coords["x1"] <= coords["x2"] else int(coords["x1"] * scale2)
                corrected_y2 = int(coords["y2"] * scale2) if coords["y1"] <= coords["y2"] else int(coords["y1"] * scale2)
                corrected_width = int(coords["width"] * scale2)
                corrected_height = int(coords["height"] * scale2)

                if corrected_x1 < 0:
                    corrected_x1 = 0
                elif corrected_x1 > orig_width:
                    corrected_x1 = orig_width

                if corrected_x2 < 0:
                    corrected_x2 = 0
                elif corrected_x2 > orig_width:
                    corrected_x2 = orig_width

                if corrected_y1 < 0:
                    corrected_y1 = 0
                elif corrected_y1 > orig_height:
                    corrected_y1 = orig_height

                if corrected_y2 < 0:
                    corrected_y2 = 0
                elif corrected_y2 > orig_height:
                    corrected_y2 = orig_height

                pt1 = (corrected_x1, corrected_y1)
                pt2 = (corrected_x2, corrected_y2)

                tmp = st.session_state.origin_image.copy()
                cv2.rectangle(tmp, pt1, pt2, color=(255, 0, 0), thickness=3)

                # セッションステートの画像を更新
                st.session_state.current_image = tmp

                st.write(f"サイズ: width={orig_width}, height={orig_height}")
                st.write(f"始点(pixel): (x1, y1) = ({corrected_x1}, {corrected_y1})")
                st.write(f"終点(pixel): (x2, y2) = ({corrected_x2}, {corrected_y2})")

                # 強制再描画
                if st.session_state.reload_count == 0:
                    st.session_state.reload_count = st.session_state.reload_count + 1
                    st.rerun()
                else:
                    rect = {
                        "x1": corrected_x1,
                        "y1": corrected_y1,
                        "x2": corrected_x2,
                        "y2": corrected_y2
                    }

                    with open("/root/app/data/input_rect.json", mode="w", encoding="utf-8") as f:
                        json.dump(rect, f)

                    st.session_state.reload_count = 0
            else:
                st.write("まだ画像をクリックしていません。")
