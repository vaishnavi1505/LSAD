import cv2
'''OpenCV is a cross-platform library using which we can develop real-time computer
 vision applications. It mainly focuses on image processing, video capture and 
 analysis including features like face detection and object detection.'''

import numpy as np
'''Numpy provides a high-performance multidimensional array and basic 
tools to compute with and manipulate these arrays.'''

import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)  # Get the gray image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Smooth the image; reduce the noise ;(5,5) kernel size
    canny = cv2.Canny(blur, 50, 150) # threshold
    return canny, blur, gray

image = cv2.imread('D:\jinna\Python\source/test_image.jpg')
lane_image = np.copy(image) # Protect the original image
canny, blur, gray= canny(lane_image)

cv2.imshow('3_canny', canny)
cv2.imshow('2_blur', blur)
cv2.imshow('1_gray', gray)
cv2.imshow('0_original_image', image)  # The image is shown by imshow
cv2.waitKey(0) #The image keep showing until the keyboard is pressed