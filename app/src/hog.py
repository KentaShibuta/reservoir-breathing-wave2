from skimage.feature import hog
from skimage import io, img_as_ubyte, exposure
import cv2

def get_hog():
    # グレースケールで読み込み
    img = cv2.imread('/root/app/data/frames/30_bin.png', cv2.IMREAD_GRAYSCALE)
    io.imsave("./gray-30.png", img)

    #hogとhog画像の取得
    fd, hog_image = hog(
        img, orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(3, 3), 
        visualize=True, 
        feature_vector=True
    )

    #print(hog_image)
    #cv2.imwrite("./hog-30.png", hog_image)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    hog_image_uint8 = img_as_ubyte(hog_image_rescaled)
    io.imsave("./hog-30.png", hog_image_uint8)