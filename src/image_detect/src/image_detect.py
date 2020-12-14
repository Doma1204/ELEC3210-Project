#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge

import math
import cv2

class ImageDetection:
    def __init__(self):
        # ROS Stuff
        self.subscriber = rospy.Subscriber("/vrep/image", Image, self.callback)
        self.publisher = rospy.Publisher("image_marker", Marker)

        # ROS to CV imgage convert
        self.bridge = CvBridge()

        # Load the image
        path = "/home/joseph/Programming/ELEC3210/ELEC3210-Project/src/image_detect/picture/"
        pictures_name = ["Obama", "Avril", "Levi", "Bloom", "Chinese"]
        self.pictures = [cv2.imread(path+name+".jpg") for name in pictures_name]

        # image detection init
        self.vote_sift = [0] * len(self.pictures)
        self.vote_square = [0] * len(self.pictures)
        self.marked = [False] * len(self.pictures)

        self.bf = cv2.BFMatcher()
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.keypoints, self.descriptors = [], []
        for i, picture in enumerate(self.pictures):
            picture = cv2.resize(picture, (400, 400), interpolation=cv2.INTER_AREA)
            k, d = self.sift.detectAndCompute(picture, None)
            picture = cv2.flip(picture, 1)
            kf, df = self.sift.detectAndCompute(picture, None)
            self.keypoints.append((k, kf))
            self.descriptors.append((d, df))

        # Create standard marker
        markers = [Marker() for i in range(len(pictures_name))]
        for marker in markers:
            marker.id = i
            marker.header.frame_id = "/camera_link"
            marker.type = marker.TEXT_VIEW_FACING
            marker.action = marker.ADD
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.text = pictures_name[i]

    def _mark(self, id, img):
        # Find the contours of the image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_BW = cv2.threshold(img_gray, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, contours, _ = cv2.findContours(img_BW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        ratio = float(w)/h

        if ratio > 1.1 and ratio < 0.5:
            self.vote_square[id] = 0
        else:
            self.vote_square[id] += 1

        if (self.vote_square[id] >= 15):
            theta = (math.pi / 8 * h) / img.shape[1]
            # h /= 2
            # self.markers[id].pose.position.x = (h / math.tan(theta) * 0.5 / h)
            self.markers[id].pose.position.x = 0.5 * math.tan(theta)
            self.markers[id].pose.position.y = (x + w/2) / img.shape[1]
            self.markers[id].pose.position.z = 0

            self.publisher.publish(self.markers[id])
            self.marked[id] = True

    def _best_fit(self, img):
        try:
            best_id = 0
            best = 0
            keypoint, descriptor = self.sift.detectAndCompute(img, None)

            for i, d in enumerate(self.descriptors):
                cur_cnt = 0

                for dd in d:
                    result = self.bf.knnMatch(dd, descriptor, k=2)
                    for m, n in result:
                        if m.distance < 0.75 *n.distance:
                            cur_cnt += 1

                if cur_cnt > best:
                    best = cur_cnt
                    best_id = i

            return best_id, best
        except Exception as e:
            return 0, 0

    def callback(self, img):
        try:
            # convert ROS image to CV image
            img = bridge.imgmsg_to_cv2(msg, "bgr8")
            id, cnt = self._best_fit(img)

            if (cnt > 120):
                self.vote_sift[id] = 0
            else:
                self.vote_sift[id] += 1

            if(self.vote_sift[id] >= 10 and not self.marked[id]):
                self._mark(id, img)
        except Exception:
            pass

def main():
    rospy.init_node("image_detect")
    node = ImageDetection()
    rospy.spin()

if __name__ == "__main__":
    main()