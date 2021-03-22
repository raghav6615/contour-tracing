import cv2
import numpy as np


def HarrisCorner(img, threshold, kernel_size=3, k=0.04):
    corners = []
    img_out = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
    # preprocessing
    img_size = 225
    img = cv2.resize(img, (img_size, img_size))  # print(img.shape)
    img = cv2.blur(img, (3, 3))
    """
    cv2.imshow('1', img)
    # blur
    img = cv2.blur(img, (3, 3))
    cv2.imshow('2', img)
    cv2.waitKey(0)
    """


    offset = int(kernel_size / 2)
    img_range = img_size - offset + 1
    print(img_range)
    # gradient
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.absolute(sobelx)
    sobelx = np.uint8(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = np.absolute(sobely)
    sobely = np.uint8(sobely)

    Ixx = sobelx ** 2
    Ixy = sobely * sobelx
    Iyy = sobely ** 2

    for x in range(offset, img_range):
        for y in range(offset, img_range):
            # Values of sliding window
            start_x = x - offset
            end_x = x + offset + 1
            start_y = y - offset
            end_y = y + offset + 1

            # The variable names are representative to
            # the variable of the Harris corner equation
            tempIxx = Ixx[start_x: end_x, start_y: end_y]
            tempIxy = Ixy[start_x: end_x, start_y: end_y]
            tempIyy = Iyy[start_x: end_x, start_y: end_y]

            # Sum of squares of intensities of partial derivatives
            Sxx = tempIxx.sum()
            Sxy = tempIxy.sum()
            Syy = tempIyy.sum()

            # Calculate determinant and trace of the matrix
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy

            # Calculate r for Harris Corner equation
            r = det - k * (trace ** 2)

            if r > threshold:
                corners.append([x, y, r])
                img_out[x, y] = (0, 0, 255)
    return corners, img_out
