import cv2
import numpy as np
import argparse
import glob
import cv2

from matplotlib import pyplot as plt


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

if __name__ == '__main__':
    img = cv2.imread('C:/Users/nons3ns/Desktop/Edgedetector/source.jpg', 1)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("gray.jpg", img_gray)

    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)

    cv2.imwrite("edges0.jpg", edges)

    edges = cv2.Canny(img_gray, 50, 150, apertureSize=5)

    cv2.imwrite("edges2.jpg", edges)

    edges = cv2.Canny(img_gray, 50, 150, apertureSize=7)

    cv2.imwrite("edges4.jpg", edges)

    img_blurry = cv2.blur(edges, (5, 5))

    cv2.imwrite("edges41.jpg", img_blurry)

    level = 1001
    factor = (259 * (level + 255)) / (255 * (259 - level))
    for x in range(img_blurry.shape[0]):
        for y in range(img_blurry.shape[1]):
            img_blurry[x, y] = factor * img_blurry[x, y]

    cv2.imwrite("edges42.jpg", img_blurry)


    # ret, thresh = cv2.threshold(edges, 127, 255, 0)
    #
    # cv2.imwrite("thresh.jpg", thresh)

    (_, contours, _) = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    # cv2.drawContours(edges, contours, -1, (255, 255, 255), 3)
    cv2.drawContours(edges, contours, -1, (255, 255, 255), 3)
    # cv2.drawContours(edges, contours, 0, (255, 255, 255), 3)

    cv2.imwrite("contours.jpg", edges)
    # print(contours)
    # print(hierarchy)
