import cv2
import numpy as np
import pandas as pd

# ファイル名, このファイルだけなので引数の使用はなし
filename = 'image/milkdrop.bmp'

#画像で使用されている色を検出
def extract_unique_color(img_path, rgb=False):
    bgr_array = cv2.imread(img_path)
    row, col, _ = bgr_array.shape
    reshaped_bgr_array = bgr_array.reshape(row * col, 3)
    unique_color_array = np.unique(reshaped_bgr_array, axis=0)
    if rgb:
        # rgbに要素を並び替え
        unique_color_array = unique_color_array[:, [2, 1, 0]]
    return unique_color_array

#検出した色から閾値を決定
'''color = extract_unique_color(filename)
df = pd.DataFrame(index=[], columns=["b", "g", "r"])
for uniq in color:
    u = uniq.tolist()
    df = pd.concat([df, pd.DataFrame({"b": [u[0]], "g": [u[1]], "r": [u[2]]})])
min_b = df["b"].quantile(0.6)
min_g = df["g"].quantile(0.6)
min_r = df["r"].quantile(0.9)'''

#課題１①
img = cv2.imread(filename)
cv2.imshow("Show Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#画像のサイズ取得
height, width = img.shape[:2]

# 白色検出
# 課題1②
#min = np.array([int(min_b), int(min_g), int(min_r)])
min = np.array([80, 80, 80])
max = np.array([255, 255, 255])
mask = cv2.inRange(img, min, max)    # マスクを作成'''
'''hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_min = np.array([0,0,90])
hsv_max = np.array([180, 45, 255])
mask = cv2.inRange(hsv, hsv_min, hsv_max)    # HSVからマスクを作成'''
cv2.imshow("Show Image", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

#課題１③
masked_img = cv2.bitwise_and(img, img, mask=mask)
masked_img = cv2.cvtColor(masked_img,cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(masked_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
range_hierarchy = (hierarchy[0][-2].tolist()[0]+1)

area=0
h_max=0
# 輪郭を内部を塗りつぶし、面積がもっとも大きなものをミルククラウンと認識
for h in range(1, range_hierarchy):
    img_ground = np.zeros((height, width, 3), np.uint8)
    cv2.drawContours(img_ground, contours, h, color=(255, 255, 255), thickness=-1)
    img_ground = cv2.cvtColor(img_ground, cv2.COLOR_BGR2GRAY)
    if area < cv2.countNonZero(img_ground):
        area = cv2.countNonZero(img_ground)
        h_max = h

img_ground = np.zeros((height, width, 3), np.uint8)
cv2.drawContours(img_ground, contours, h_max, color=(255, 255, 255), thickness=-1)
# 元画像からクラウンの水滴が一つ除外されていることを確認
# ミルククラウンの主領域との距離で識別しようとしたが、うまくいかず目視で追加
cv2.drawContours(img_ground, contours, 19, color=(255, 255, 255), thickness=-1)
cv2.imshow("Show Image", img_ground)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 課題１④
img_onlymilk = cv2.bitwise_and(img, img_ground)
cv2.imshow("Show Image", img_onlymilk)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 目視確認用
'''for h in range(-1, h_max):
    img_ground = np.zeros((height, width, 3))
    cv2.drawContours(img_ground, contours, h, color=(255, 255, 255), thickness=2)
    cv2.imshow("Show Image"+str(h), img_ground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
#cv2.imwrite(filename.split('.')[0]+"_outline_img.jpg", img_ground)
