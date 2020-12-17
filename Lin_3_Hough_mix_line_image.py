import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image): #Get the boundary of image
    gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)  # Get the gray image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Smooth the image, reduce the  noice
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image): # Select the  region of interest
    height = image.shape[0]
    polygons = np.array([
    [(200, height), (1100, height),(550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255) # The interested area will be white
    masked_image = cv2.bitwise_and(image, mask) # crop the canny image by mask
    return masked_image

def display_lines(image, lines): # Display lines in black image
    line_image = np.zeros_like(image) # Same size black image
    if lines is not None:
        for line in lines:
            #print(line)
            x1, y1, x2, y2 = line.reshape(4) #Get (x1,y1) (x2,y2) of each line
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3) # Draw lines
    return line_image


image = cv2.imread('D:\jinna\Python\source/test_image.jpg')
lane_image = np.copy(image)# Protect the original image
canny = canny(lane_image)
cropped_image = region_of_interest(canny)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100 , np.array([]), minLineLength= 40, maxLineGap= 5 )
line_image = display_lines(lane_image, lines)
comb_img = cv2.addWeighted(lane_image, 0.8, line_image , 1 , 1) #Blend the original image and lines


cv2.imshow("Mix", comb_img )
cv2.imshow("Line_imge", line_image )
cv2.imshow("Cropped_image", cropped_image )
cv2.waitKey(0)