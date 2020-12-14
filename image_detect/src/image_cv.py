#!/usr/bin/env python
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
import sys
import math
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
# ROS Image message
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
import numpy as np
threshold = 120
pic_resize = 400

# Instantiate CvBridge
bridge = CvBridge()
# given PATTERN
path = r'/home/joseph/Programming/ELEC3210/ELEC3210-Project/src/image_detect/picture/'
name = ["Obama", "Avril", "Levi", "Bloom", "chinese", "No picture"] #keep no picture at least
pic = []
pic.append(cv2.imread(path+'obama.jpg'))
pic.append(cv2.imread(path+'avril.jpg'))
pic.append(cv2.imread(path+'levi.jpg'))
pic.append(cv2.imread(path+'bloom.jpg'))
pic.append(cv2.imread(path+'chinese.jpg'))

pic_f = []
pic_f.append(cv2.imread(path+'obama.jpg'))
pic_f.append(cv2.imread(path+'avril.jpg'))
pic_f.append(cv2.imread(path+'levi.jpg'))
pic_f.append(cv2.imread(path+'bloom.jpg'))
pic_f.append(cv2.imread(path+'chinese.jpg'))
voting =[] #voting check for sift
voting2=[] #voting check for sqare
published =[]
pic_len = len(pic)
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
des= [None] * pic_len
kp = [None] * pic_len
des_f= [None] * pic_len
kp_f = [None] * pic_len
for i in range(0, pic_len):
    pic[i] = cv2.resize(pic[i], (pic_resize,pic_resize), interpolation = cv2.INTER_AREA)
    #pic_f[i]=cv2.flip(pic[i], 1)
    pic[i]=cv2.flip(pic[i], 1)
    kp[i], des[i] = sift.detectAndCompute(pic[i],None)
    kp_f[i], des_f[i] = sift.detectAndCompute(pic_f[i],None)
    voting.append(0)
    voting2.append(0)
    published.append(False)
# BFMatcher with default params
bf = cv2.BFMatcher()

pub = rospy.Publisher('visualization_marker', Marker, queue_size=100)
marker_arr = []
for i in range(0,pic_len):
    marker = Marker()
    marker.header.frame_id = "/camera_link"
    marker.type = marker.TEXT_VIEW_FACING
    marker.action = marker.ADD
    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.pose.orientation.w = 1.0
    marker.text = name[i]
    marker.id = i
    marker_arr.append(marker)

def pub_marking(img,name_id): #Camera resectioning & make marker
    #conver to bw
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _,contours,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # find the biggest area
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    # draw the book contour (in green)
    ratio = float(w)/h
    #print("ratio:", ratio, end ="")
    if(ratio>1.1 and ratio<0.5):
        #cv2.rectangle(gray,(x,y),(x+w,y+h),(0,0,255),2)
        #cv2.imshow('squre', gray)
        #cv2.waitKey(1)
        voting2[name_id]=0;
        return
    else:
        voting2[name_id]=voting2[name_id]+1;
        #cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.imshow('squre', gray)
        #cv2.waitKey(1)
    if(voting2[name_id]>15):
        print name[name_id], "Pass ", ratio
        im_pix_h = h / 2
        im_theta = ((math.pi / 8) * h) / (img.shape[1])
        dist_pix = (im_pix_h) / math.tan(im_theta)
        dist_x_real = ((dist_pix * 0.5) / im_pix_h)

        marker_arr[name_id].pose.position.x = dist_x_real
        marker_arr[name_id].pose.position.y = ((x + w / 2) / img.shape[1])
        marker_arr[name_id].pose.position.z = 0

        #print(name[name_id] + " " + str(dist_x_real))
        pub.publish(marker_arr[name_id])


def compare(i, cam_img): #i = id of given picture, cam_img = camera
    # find the keypoints and descriptors with SIFT
    try:
        kp_cam, des_cam = sift.detectAndCompute(cam_img,None)
        matches = bf.knnMatch(des[i],des_cam,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

         #cv.drawMatchesKnn expects list of lists as matches.
        #img3 = cv2.drawMatchesKnn(pic[i],kp_f[i],cv2_img,kp_cam,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #if(len(good)>threshold):
        #    cv2.imshow('Detection', img3)    #real time camera
        #    cv2.waitKey(1)
        #Flipped
        matches = bf.knnMatch(des_f[i],des_cam,k=2)
        # Apply ratio test
        good_f = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

         #cv.drawMatchesKnn expects list of lists as matches.
        #img3 = cv2.drawMatchesKnn(pic_f[i],kp_f[i],cv2_img,kp_cam,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #if(len(good_f)>threshold):
        #    cv2.imshow('Detection', img3)    #real time camera
        #    cv2.waitKey(1)
        return len(good)+len(good_f) #how similar
    except:
        return -1

def image_callback(msg):
    global val_sim, cv2_img
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        #cv2_img = cv2.Canny(cv2_img,100,200)
        max_id = 0
        max = compare(0, cv2_img) #+ compare(0, cv2.flip(cv2_img, 1))    #marks of normal + flip
        for i in range(0, pic_len):
            this = compare(i, cv2_img) #+ compare(i, cv2.flip(cv2_img, 1))
            if(this>max):
                max_id=i
                max=this
        if(max<threshold): #thresholding for no picture detected
            voting[max_id]=0    #it is just a noise, reset vote counter
            max_id=len(name)-1
        #print(name[max_id], max)
        else:    #not the last one-> not
            voting[max_id] = voting[max_id]+1
            if(voting>10 and ~(published[max_id])):  #continously 10 correct sample
                pub_marking(cv2_img, max_id)
                published[max_id] = True
    except CvBridgeError, e:
        #print(e)
        i=1

def main():
    rospy.init_node('image_detect')
    # Define your image topic
    image_topic = "/vrep/image"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
