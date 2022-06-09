#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import time
from pyModbusTCP.client import ModbusClient


# In[2]:


def getWorldCoordinates(cameraCoordinates):
    
    x1 = cameraCoordinates[0]*0.38
    y1 = cameraCoordinates[1]*0.37
    
    print("Coordinates in mm - ", x1, y1)
    
    return (x1, y1)


# In[3]:


def addRobotOffset(world_center):
    
    x2 = world_center[0]+19.6
    y2 = world_center[1]
    
    return (x2, y2)


# In[4]:


def rotateYBy4(robot_coordinates_raw):

    cosT = 0.99756405026
    sinT = 0.06975647374
    x = robot_coordinates_raw[0]
    y = robot_coordinates_raw[1]
    
    print("x - ", x)
    print("y - ", y)
    
    x_new = x
    y_new = y
    
    return (x_new, y_new)


# In[5]:


def findRotationAngle(input_image):
    
    img = input_image.copy()
    
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 245, 245])
    mask2 = cv.inRange(hsv, lower_red, upper_red)

    mask = mask1 | mask2

    contour, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    cntsSorted = sorted(contour, key=lambda x: cv.contourArea(x))

    blank = img.copy()
    rotation_angle = 0
    center = (0, 0)

    for cnt in cntsSorted:
        area = cv.contourArea(cnt)
        #print(area)
        if 8000 < area < 12000:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            center = rect[0]
            w, h = rect[1]
            if w < h:
                rotation_angle = 90-rect[-1]
            else:
                rotation_angle = 180-rect[-1]
            cv.drawContours(img,[box],0,(0,0,255),2)
            cv.drawContours(blank, [cnt], -1, (0, 255, 0), 2)
            
    cv.imshow("contours_image", blank)
    
    return rotation_angle


# In[6]:


def setCamera(cam):
    
    cap = cv.VideoCapture(cam)
    cap.set(cv.CAP_PROP_SETTINGS, 1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    
    return cap


# In[7]:


def configureModbus(server, port):

    client = ModbusClient()

    client.host(server)
    client.port(port)
    client.open()
        
    if client.is_open():
        print("Modbus connection successfully established")
    else:
        print("Connection not established")
    
    return client


# In[8]:


def sendValue(client, address, val):
    val = round(val*10)
    print("Value - ", val, " sent at address - ", address)
    client.write_single_register(address, val)


# In[9]:


def findCenterCoord(img):
    
    original = img.copy()
    gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (7, 7), 0)
    
    canny = cv.Canny(blur, 70, 255, 0)
    
    cv.imshow("Canny", canny)
    
    conts, hir = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    contsSorted = sorted(conts, key=lambda x: cv.contourArea(x))
    blank = img.copy()
    
    cx=0
    cy=0
    for c in contsSorted:
        area = cv.contourArea(c)
        if area > 1000:
            print(area)
        if 1300 < area < 2000:
            print(area)
            cv.drawContours(blank, [c], -1, (255, 0, 0), 2)
            M = cv.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv.circle(blank, (cx, cy), 5, (0, 0, 255), 2)
        
    cv.imshow("Contours", blank)
    return (cx, cy)


# In[10]:


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


# In[11]:


def main():
    
    cap = setCamera(2)
    cap.set(cv.CAP_PROP_SETTINGS, 1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    #Capture 300 frames for adjusting focus
    for i in range(300):
        ret, image = cap.read()
        cv.imshow("Pre-trial", image)
        cv.waitKey(1)

    # Capture processing frame
    ret, image = cap.read()
    calibrated_image = getCalibratedImage(image)
    
    # Calculate center and rotation angle in image frame
    rotation_angle = findRotationAngle(calibrated_image)
    center = findCenterCoord(calibrated_image)
    print("Center - ", center)
    
    # Convert center coordinates to world plane
    world_center = getWorldCoordinates(center)
    
    # Add robot plane offsets to the world coordinates
    robot_coordinates_raw = addRobotOffset(world_center)
    
    # Apply clockwise rotation to Y by 4 degree
    robot_coordinates_final = rotateYBy4(robot_coordinates_raw)
    
#     Configure modbus and send values
#     client = configureModbus("194.94.86.6", 502)

#     address = 24576
#     print("Sending angle value - ", rotation_angle, " at address ", address)
#     sendValue(client, address, rotation_angle)

#     address = 24637
#     print("Sending x - ", robot_coordinates_final[0], " at address ", address)
#     sendValue(client, address, robot_coordinates_final[0])

#     address = address + 1
#     print("Sending y - ", robot_coordinates_final[1], " at address ", address)
#     sendValue(client, address, robot_coordinates_final[1])

#     address = address + 1
#     print("Sending z - ", 130, " at address ", address)
#     sendValue(client, address, 130)

    k = cv.waitKey(0)
    if k==ord("q"):
        cv.destroyAllWindows()


# In[12]:


main()


# In[ ]:





# In[ ]:




