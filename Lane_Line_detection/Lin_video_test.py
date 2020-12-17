import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinate(image, line_parameters):  # make_coordinate of left and right lines
    slope, intercept = line_parameters
    print(image.shape) # height ,width, channel
    y1 = image.shape[0] #The buttom of y 704
    y2 = int(y1*(3/5)) # The upper of y 420
    x1 = int((y1 -intercept )/slope)
    x2 = int((y2 - intercept) /slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines): #Combine two lines into single line
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2),1) #Get slope and intercept
        #print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope <0:
            left_fit.append((slope, intercept)) # left tuple
        else:
            right_fit.append((slope,intercept)) # right tuple
        #print(left_fit) #Left line is negative value
        #print(right_fit)
    left_fit_average =  np.average(left_fit, axis=0)
    right_fit_average =  np.average(right_fit, axis=0)
    print(left_fit_average,'L')  # left average tuple
    print(right_fit_average,'R')
    left_line = make_coordinate(image,  left_fit_average)
    right_line = make_coordinate(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image): #Get the boundary of image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Get the gray image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Smooth the image, reduce the  noice
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines): # Line detection using HoughLinesP
    line_image = np.zeros_like(image) # Same size black image
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            #print(line)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 0), 10) #draw line
    return line_image

def region_of_interest(image): # Select the region of interest
    height = image.shape[0]
    polygons = np.array([
    [(100, height), (1100, height),(550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255) # The interested area will be white
    masked_image = cv2.bitwise_and(image, mask) # crop the canny image by mask
    return masked_image

# image = cv2.imread('D:\jinna\Python\source/test_image.jpg')
# lane_image = np.copy(image)# Protect the original image
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100 , np.array([]), minLineLength= 40, maxLineGap= 5 )
# average_line = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, average_line)# Modified line image
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image , 1 ,1) #blend the original image and lines
# #cv2.imshow("Line_imge", line_image )
# cv2.imshow("Mix", combo_image )
# cv2.waitKey(0)

cap = cv2.VideoCapture("D:\jinna\Python\source/road.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap= 50)
    average_line = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, average_line)  # Modified line image
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)  # blend the original image and lines
    # cv2.imshow("Line_imge", line_image )
    cv2.imshow("Mix", combo_image)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


