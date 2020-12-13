#!/usr/bin/env python2.7

# Python dependencies
import numpy as np
import cv2
#from simple_pid import PID
from cv_bridge import CvBridge, CvBridgeError

# ROS dependencies
import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# Instantiate CvBridge
bridge = CvBridge()

def image_callback(msg):
    global cv2_img
    try:
        # Convert ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2_img = cv2.flip(cv2_img, 1)
        #mask ball image
        hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        #find ball contour
        (_,contours,_) = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            return

        #locate centroid of ball
        for c in contours:
            #print(cv2.arcLength(c, True))
            M = cv2.moments(c)
            cX = float(M["m10"]/M["m00"])
            cY = float(M["m01"]/M["m00"])
            rX = int(M["m10"]/M["m00"])
            rY = int(M["m01"]/M["m00"])


        #find distance
        h,w = cv2_img.shape[:2]
        (ideal_X, ideal_Y) = (w/2, 350)
        verticle_diff = cY-ideal_Y
        angle_diff = cX-ideal_X
        
        pub = rospy.Publisher('/vrep/cmd_vel', Twist, queue_size=10)
        twist = Twist()
        #linear
        if verticle_diff <= -50:
            #print("F1")
            twist.linear.x = 1.1
        elif (verticle_diff > -50) & (verticle_diff < 0):
            #print("F2")
            twist.linear.x = 0.5
        elif verticle_diff >= 20:
            #print("B1")
            twist.linear.x = -0.6
        elif (verticle_diff <20) & (verticle_diff > 5):
            #print("B2")
            twist.linear.x = -0.3
        else:
            #print("stay")
            twist.linear.x = 0    
        #angular
        if angle_diff >= 30:
            #print("R1")
            twist.angular.z = -1
        elif (angle_diff < 30) & (angle_diff > 10):
            #print("R2")
            twist.angular.z = -0.5
        elif angle_diff <= -30:
            #print("L1")
            twist.angular.z = 1
        elif (angle_diff > -30) & (angle_diff < -10):
            #print("L2")
            twist.angular.z = 0.5
        else:
            #print("center")
            twist.angular.z = 0
        
        pub.publish(twist)
        
        #Debug & window
        print("center:",cX,cY)
        copy_img = cv2_img.copy()
        cv2.drawContours(copy_img, contours, -1, (0, 0, 255), 2)
        cv2.circle(copy_img, (rX, rY), 3, (255, 0, 0), -1)
        cv2.putText(copy_img, "centroid", (rX - 25, rY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Image", copy_img)
        #cv2.imshow("filtered", mask)
        cv2.waitKey(1)


    except CvBridgeError, e:
        print(e)

def main():
    print("Start Ball Trackering")
    rospy.init_node('ball_track')
    rospy.Subscriber('/vrep/image', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()