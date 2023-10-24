import cv2
import math
import imutils
import numpy as np

image = cv2.imread('C:/Users/1/Desktop/COS_img/lab2/foto.jpg', 1)
img = image.copy()
M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 60, 2.0)
image_scaled_rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

img = image.copy()
img_scaled_rotated = image_scaled_rotated.copy()

polar_image = cv2.warpPolar(img, (img.shape[0], 360),
                              (img.shape[0] / 2, img.shape[1] / 2),
                              np.sqrt((img.shape[0] ** 2.0 + img.shape[1] ** 2.0)) / 2,
                              cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG)

polar_image_scaled_rotated = cv2.warpPolar(img_scaled_rotated, (img_scaled_rotated.shape[0], 360),
                                    (img_scaled_rotated.shape[0] / 2, img_scaled_rotated.shape[1] / 2),
                                    np.sqrt((img_scaled_rotated.shape[0] ** 2.0 + img_scaled_rotated.shape[1] ** 2.0)) / 2,
                                    cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG)

gray_polar_image = cv2.cvtColor(polar_image, cv2.COLOR_BGR2GRAY)
gray_polar_image_scaled_rotated = cv2.cvtColor(polar_image_scaled_rotated, cv2.COLOR_BGR2GRAY)

collage = np.vstack((gray_polar_image, gray_polar_image))
collage = np.hstack((collage, collage))

res = cv2.matchTemplate(collage, gray_polar_image_scaled_rotated, cv2.TM_CCORR_NORMED)
(_, _, _, maxLoc) = cv2.minMaxLoc(res, None)

w = img.shape[0]
R = np.sqrt((img.shape[0] ** 2.0 + img.shape[1] ** 2.0)) / 2
print ('angle =', maxLoc[1])
print ('scale =', math.exp((w - maxLoc[0]) * (math.log(R) / w)))

cv2.imshow("Input Image", image)
cv2.imshow("Scaled And Rotated Image", image_scaled_rotated)
cv2.imshow("Polar Input Image", polar_image)
cv2.imshow("Polar Scaled And Rotated Image", polar_image_scaled_rotated)
cv2.imshow('Collage', collage)
cv2.imshow('Output', res)

cv2.waitKey(0)