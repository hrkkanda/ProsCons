import cv2
import numpy as np
import pandas as pd

# ファイル名, このファイルだけなので引数の使用はなし
filename = 'image/metal_panel.jpg'
# 画像の読み込み
img_rgb = cv2.imread(filename)
height, width = img_rgb.shape[:2]
# 画像が大きすぎるので、1/10にスケール
img_rgb = cv2.resize(img_rgb, (int(width/10), int(height/10)))
# グレースケール化
gry = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# 閾値130で二値化
ret, thresh = cv2.threshold(gry, 100, 255, 0)

# 表示
# cv2.imshow("Show Image", gry)
cv2.imshow("Show Image", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
range_hierarchy = (hierarchy[0][-2].tolist()[0]+1)
h_max=0
area=0
for h in range(1, range_hierarchy):
    img_ground = np.zeros((int(height/10), int(width/10), 3), np.uint8)
    cv2.drawContours(img_ground, contours, h, color=(255, 255, 255), thickness=-1)
    img_ground = cv2.cvtColor(img_ground, cv2.COLOR_BGR2GRAY)
    if area < cv2.countNonZero(img_ground):
        area = cv2.countNonZero(img_ground)
        h_max = h
img_ground = np.zeros((int(height/10), int(width/10), 3), np.uint8)

cv2.drawContours(img_ground, contours, h_max, color=(255, 255, 255), thickness=-1)
img_only1 = cv2.bitwise_and(img_rgb, img_ground)
cv2.imshow("Show Image"+str(h_max), img_only1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 輪郭内部をの面積が最大値となるもの以外を削除
grey = cv2.cvtColor(img_only1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grey, 140, 255, 0)
cv2.imshow("Show Image", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
range_hierarchy = (hierarchy[0][-2].tolist()[0]+1)
area = 0
h_max = 0
for h in range(1, range_hierarchy):
    img_ground = np.zeros((int(height/10), int(width/10), 3), np.uint8)
    cv2.drawContours(img_ground, contours, h, color=(255, 255, 255), thickness=-1)
    img_ground = cv2.cvtColor(img_ground, cv2.COLOR_BGR2GRAY)
    if area < cv2.countNonZero(img_ground):
        area = cv2.countNonZero(img_ground)
        h_max = h

img_ground = np.zeros((int(height/10), int(width/10), 3), np.uint8)+255
for h in range(1, range_hierarchy):
    if h != h_max:
        cv2.drawContours(img_ground, contours, h, color=(0, 0, 0), thickness=-1)

cv2.imshow("Show Image", img_ground)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_only2 = cv2.bitwise_and(img_only1, img_ground)
cv2.imshow("Show Image"+str(h_max), img_only2)
cv2.waitKey(0)
cv2.destroyAllWindows()