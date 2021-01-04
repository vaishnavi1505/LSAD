import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import numpy as np
import torch
import model_Le
import torch.nn as nn
import pandas as pd
import math


classes = 'E:\ex_python\sign_detection\\backup\signnames.csv'
classes_index = pd.read_csv(classes)
# print(classes_index['SignName'][0])

train_weights = 'E:\ex_python\sign_detection\\backup\\trained_model.pth'
model = model_Le.LeNet()
state_dict = torch.load(train_weights)
model.load_state_dict(state_dict)
model.eval()


def get_probability(img):
    img = np.asanyarray(img)
    img = cv2.resize(img, (32, 32),interpolation=cv2.INTER_CUBIC)

    img = torch.from_numpy(img.transpose((2, 0, 1)))
    img = img.float().div(255).unsqueeze(0)
    # img = img.unsqueeze(0)

    with torch.no_grad():
        out = model(img)

    probability = torch.nn.functional.softmax(out, dim=1)

    probabilityValue = probability.data.max(1)[0]
    class_pred = probability.data.max(1)[1]
    label = classes_index['SignName'][class_pred.item()]

    return probabilityValue, label



def blue_color_select(img):
    cnt_list =[]
    label_list =[]
    # h_img, w_img, c_img = img.shape
    h_img,w_img,c_img =img.shape
    # (3024, 4032, 3)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # convert the image to HSV format for color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #BLUE
    blue_lower=np.array([94,127,20])
    blue_upper=np.array([126,255,200])
    mask_b=cv2.inRange(hsv,blue_lower,blue_upper)
    blue_mask = cv2.bitwise_and(img_output,img_output,mask=mask_b)

    # seperate out the channels
    r_channel = blue_mask[:, :, 2]
    g_channel = blue_mask[:, :, 1]
    b_channel = blue_mask[:, :, 0]

    # filter out
    filtered_r = cv2.medianBlur(r_channel, 5)
    filtered_g = cv2.medianBlur(g_channel, 5)
    filtered_b = cv2.medianBlur(b_channel, 5)

    # create a blue gray space
    filtered_b = -0.5 * filtered_r + 3 * filtered_b - 2 * filtered_g


    _, b_thresh = cv2.threshold(np.uint8(filtered_b), 60, 255, cv2.THRESH_BINARY)

    r_thresh = cv2.GaussianBlur(b_thresh, (5, 5), 0)

    kernel_1 = np.ones((3, 3), np.uint8)
    kernel_2 = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(r_thresh, kernel_1, iterations=1)
    dilation = cv2.dilate(erosion, kernel_2, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_2)


    # Do MSER
    mser_red = cv2.MSER_create(8, 100, 10000)
    regions, _ = mser_red.detectRegions(np.uint8(opening))
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    blank_im = np.zeros_like(r_thresh)
    cv2.fillPoly(np.uint8(blank_im), hulls, (255, 255, 255))  # fill a blank image with the detected hulls

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,6))
    closed = cv2.morphologyEx(blank_im, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts_circle =[]

    for cnt in cnts:
         x_, y_, weight, height = cv2.boundingRect(cnt)
         # if x_ < 0.5 * w_img:
         #    continue
         #contour area and perimeter
         area = cv2.contourArea(cnt)
         perimeter = cv2.arcLength(cnt, True)

         if 500< area < 10000:         # further filter by area
             Roundness = (area * 4 * math.pi) / (cv2.arcLength(cnt,True)**2)
             ((cx, cy), (w, h), theta) = cv2.minAreaRect(cnt)
             # (x, y, w, h) = cv2.boundingRect(cnt)
             area_rect = w * h
             perimeter_rect = 2*(w+h)
             Rectangularity = area / area_rect
             Elongation = min(w,h) / max(w,h)

             #Minimum circumscribed circle
             (x, y), radius = cv2.minEnclosingCircle(cnt)
             (x, y, radius) = map(int, (x, y, radius))
             #circumference(perimeter)
             C = 2*math.pi*radius
             S = math.pi*radius*radius

             # conditions for circle
             if Roundness > 0.85 and Rectangularity > 0.70 and Elongation>0.75 and area/S > 0.7 and perimeter/C >0.7 and 0.7< area/area_rect <0.9 and 0.7<perimeter/perimeter_rect<0.9:  # possible circle sign
                 cnts_circle.append(cnt)




    ## for visulization blue circle  signs
    # for cnt in cnts_circle:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     if x > 0.5*w_img:
    #      cv2.rectangle(img, (x, y), (int(x + w), int(y + h)), (0, 0, 255), 2)
    #      # cv2.rectangle(mask, (x, y), (int(x + w), int(y + h)), (255, 255, 255), -1)


    out = np.zeros_like(img)
    for cnt in cnts_circle:
        x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(img, (x, y), (int(x + w), int(y + h)), (0, 0, 255), 2)

        out = img[y:y+h,x:x+w]  # crop the region of traffic sign
        # load classfier
        probability,label = get_probability(out)
        cv2.rectangle(img, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(img, str(label), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.50, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.namedWindow('original', cv2.WINDOW_NORMAL)
        cv2.imshow('original', img)
        cv2.waitKey(4000)


import os
if __name__=="__main__":
    # a for loop for testing images
    # for file_img in os.listdir('E:\ex_python\sign_detection\CODE\\images'):
    #     # print(file_img)
    #     img = cv2.imread('E:\ex_python\sign_detection\CODE\\images'+'\\'+file_img)
    #     img = img.copy()
    #     h_img, w_img, c_img = img.shape
    #     img = cv2.resize(img,(int(w_img/3),int(h_img/3)),interpolation=cv2.INTER_CUBIC)
    #     a = red_color_select(img)

    #signle image
    img = cv2.imread('E:\ex_python\sign_detection\detecteor\\example9_1.jpg')
    # (3024, 4032, 3) image took by the smartphone
    img = img.copy()
    h_img, w_img, c_img = img.shape
    img = cv2.resize(img,(int(w_img/3),int(h_img/3)),interpolation=cv2.INTER_CUBIC)
    a = blue_color_select(img)