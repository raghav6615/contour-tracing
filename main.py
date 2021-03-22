import cv2
from Harris import HarrisCorner
# import numpy as np

# given
path = r'E:\study\DIP\DIP_PROJECT\venv\resource\attachments\test2.jpg'
threshold = 20000

# reading and resizing image
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# print(img.shape)
corners, img_out = HarrisCorner(img, threshold)
cv2.imshow("out", img_out)
cv2.waitKey(0)