import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image): #Get the boundary of image
    gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)  # Get the gray image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Smooth the image, reduce the noise
    canny = cv2.Canny(blur, 50, 150)
    #cv2.imshow('result', image)  # The image is shown by imshow
    #cv2.imshow('gray', gray)
    #cv2.imshow('blur', blur)
    return canny

def region_of_interest(image): # Select the region of interest
    height = image.shape[0] # Height of image
    polygons = np.array([
    [(200, height), (1100, height),(550, 250)] # The coordinate of RIO
    ])
    mask = np.zeros_like(image) # Same size black image
    cv2.fillPoly(mask, polygons, 255) # The interested area will be white 255
    return mask

image = cv2.imread('D:\jinna\Python\source/test_image.jpg')
lane_image = np.copy(image)# Protect the original image
canny = canny(lane_image)

plt.imshow(canny) #Coordinate  system
plt.show()

cv2.imshow("Rigion of Interest", region_of_interest(canny))
cv2.waitKey(0)



