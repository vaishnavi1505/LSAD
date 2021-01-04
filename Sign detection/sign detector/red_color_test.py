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



def red_color_select(img):
    cnt_list =[]
    label_list =[]

    h_img,w_img,c_img =img.shape
    # (3024, 4032, 3)

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Since many believe that the RGB color space is very fragile with regards to changes in lighting
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 70, 60])
    upper_red_1 = np.array([10, 255, 255])
    mask_1 = cv2.inRange(img_hsv, lower_red_1, upper_red_1)

    lower_red_2 = np.array([170, 70, 60])
    upper_red_2 = np.array([180, 255, 255])
    mask_2 = cv2.inRange(img_hsv, lower_red_2, upper_red_2)

    mask = cv2.bitwise_or(mask_1, mask_2)
    red_mask_ = cv2.bitwise_and(img_output, img_output, mask=mask)

    red_mask = red_mask_[:int((1/2)*h_img), :]
    # separating channels
    r_channel = red_mask[:, :, 2]
    g_channel = red_mask[:, :, 1]
    b_channel = red_mask[:, :, 0]

    # filtering
    filtered_r = cv2.medianBlur(r_channel, 5)
    filtered_g = cv2.medianBlur(g_channel, 5)
    filtered_b = cv2.medianBlur(b_channel, 5)
    # create a red gray space
    filtered_r = 4 * filtered_r - 0.5 * filtered_b - 2 * filtered_g

    # cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    # cv2.imshow('original', filtered_r)
    # cv2.waitKey(10000)
    # filtered_r = cv2.GaussianBlur(filtered_r, (5, 5), 0)
    _, r_thresh = cv2.threshold(np.uint8(filtered_r), 20, 255, cv2.THRESH_BINARY)
    r_thresh = cv2.GaussianBlur(r_thresh, (5, 5), 0)


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
    cnts_triangle =[]
    cnts_all =[]
    for cnt in cnts:
         x_, y_, weight, height = cv2.boundingRect(cnt)
         if x_ < 0.5 * w_img:   # red traffic signs are always on the right hand
            continue
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
                 cnts_all.append(cnt)

            #conditions for triangle
             elif 0.35 < Roundness < 0.70 and 0.4< Rectangularity < 0.70 and Elongation>0.70: #possible triangle sign
                 cnts_triangle.append(cnt)
                 cnts_all.append(cnt)


    ## for visulization red circle and triangle signs
    # for cnt in cnts_circle:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     if x > 0.5*w_img:
    #      cv2.rectangle(img, (x, y), (int(x + w), int(y + h)), (0, 0, 255), 2)
    #      # cv2.rectangle(mask, (x, y), (int(x + w), int(y + h)), (255, 255, 255), -1)
    # for cnt in cnts_triangle:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     if x > 0.5*w_img:
    #      cv2.rectangle(img, (x, y), (int(x + w), int(y + h)), (255, 0, 0), 2)
    #      # cv2.rectangle(mask, (x, y), (int(x + w), int(y + h)), (255, 255, 255), -1)

    out = np.zeros_like(img)
    for cnt in cnts_all:
        x, y, w, h = cv2.boundingRect(cnt)
    #     cv2.rectangle(img, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)

        out = img[y:y+h,x:x+w]  # crop the region of traffic sign
        # load classfier
        probability,label = get_probability(out)
        cv2.rectangle(img, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(img, str(label), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.50, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.namedWindow('original', cv2.WINDOW_NORMAL)
        cv2.imshow('original', img)
        cv2.waitKey(5000)


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
    img = cv2.imread('E:\ex_python\sign_detection\CODE\\images\\20201216_131249.jpg')
    # (3024, 4032, 3) image took by the smartphone
    img = img.copy()
    h_img, w_img, c_img = img.shape
    img = cv2.resize(img,(int(w_img/3),int(h_img/3)),interpolation=cv2.INTER_CUBIC)
    a = red_color_select(img)