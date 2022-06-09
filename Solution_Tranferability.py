#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
from pyModbusTCP.client import ModbusClient


# In[2]:


conversion_factor_x = 0.26
conversion_factor_y = 0.26


# In[3]:


def setCamera(cam):
    cap = cv.VideoCapture(cam)
    cap.set(cv.CAP_PROP_SETTINGS, 1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    
    return cap


# In[4]:


def startModbusProtocol(rotation_angle, x_robot, y_world):
    client = ModbusClient(host="194.94.86.6", port=502)
    client.open()
    print(client.is_open(), "For checking the connection")
    print("Sending values......")
    client.write_single_register(24640, int(rotation_angle * 10))
    client.write_single_register(24641, int(x_robot * 10))
    client.write_single_register(24642, int(y_world * 10))
    client.write_single_register(24643, int(1000))
    print("Values sent successfully")


# In[5]:


def getCalibratedImage(img):
    
    camera_matrix = np.array([[1.25203035e+03, 0.00000000e+00, 6.66075086e+02],
                     [0.00000000e+00, 1.25370911e+03, 3.62197565e+02],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
    distortion = np.array([[ 0.05018599,  0.63090903,  0.00656325,  0.00949236, -1.97458389]])
    
    h, w = img.shape[:2]
    optimal_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion, (w, h), 1, (w, h))
    
    undistorted_image = cv.undistort(img, camera_matrix, distortion, None, optimal_camera_matrix)
    
    cv.imshow("Undistorted Image", undistorted_image)
    
    return undistorted_image


# In[23]:


cam = setCamera(0)

for i in range(120):
    ret, image = cam.read()
    cv.imshow("focusing_image", image)
    cv.waitKey(1)
    
result, image = cam.read()
calibrated_img = getCalibratedImage(image)

gray = cv.cvtColor(calibrated_img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(calibrated_img, (5, 5), 0)

canny = cv.Canny(blur, 100, 255)
cv.imshow("Canny", canny)

contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
print("Number of contours" + str(len(contours)))

cntsSorted = sorted(contours, key=lambda x: cv.contourArea(x))

for each in contours:
    area = cv.contourArea(each)
    peri = cv.arcLength(each, True)
    approx = cv.approxPolyDP(each, 0.03 * peri, True)
    if len(approx) == 4:
        print(area)
        if 550 < area < 800: 
            rect = cv.minAreaRect(each)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(calibrated_img, [box], 0, (0, 0, 255), 2)
            x, y = rect[0]
            width, height = rect[1]
            
            if width < height:
                rotation_angle = 90 - rect[-1]
            else:
                rotation_angle = 180 - rect[-1]
                
            rotation_angle1 = 180 - rotation_angle
            
            x_world = x * conversion_factor_x
            y_world = y * conversion_factor_y
            
            x_robot = x_world + 90
            print("The co-ordinates in mm are: ", (x_world))
            print("The co-ordinates in mm are: ", (x_world, y_world))
            print("The ROBOT co-ordinates in mm are: ", (x_robot, y_world))
            print("The rotation angle is :", rotation_angle)
            
            startModbusProtocol(rotation_angle, x_robot, y_world)
cv.imshow("Original", calibrated_img)
cv.imshow("contour", calibrated_img)
if cv.waitKey(0) & 0xFF == ord("s"):
    cv.destroyAllWindows()


# In[ ]:





# In[ ]:




