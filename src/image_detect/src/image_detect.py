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
        self.publisher = rospy.Publisher("image_marker", Marker, queue_size=100)

        # ROS to CV imgage convert
        self.bridge = CvBridge()

        # Load the image
        path = "/home/joseph/Programming/ELEC3210/ELEC3210-Project/src/image_detect/picture/"
        pictures_name = ["pic001", "pic002", "pic003", "pic004", "pic005"]
        self.pictures = [cv2.imread(path+name+".jpg") for name in pictures_name]

        # image detection init
        self.surf_cnt = [0] * len(self.pictures)
        self.square_cnt = [0] * len(self.pictures)
        self.marked = [False] * len(self.pictures)

        self.bf = cv2.BFMatcher()
        self.surf = cv2.xfeatures2d.SURF_create(400)
        self.keypoints, self.descriptors = [], []
        for i, picture in enumerate(self.pictures):
            picture = cv2.resize(picture, (400, 400), interpolation=cv2.INTER_AREA)
            k, d = self.surf.detectAndCompute(picture, None)
            picture = cv2.flip(picture, 1)
            kf, df = self.surf.detectAndCompute(picture, None)
            self.keypoints.append((k, kf))
            self.descriptors.append((d, df))

        # rospy.loginfo("init complete")

        # Create standard marker
        # self.markers = [Marker() for i in range(len(pictures_name))]
        # for marker in self.markers:
        self.markers = []
        for i in range(len(pictures_name)):
            marker = Marker()
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
            self.markers.append(marker)

    def _mark(self, id, img):
        # rospy.loginfo("mark")
        # Find the contours of the image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_BW = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, contours, _ = cv2.findContours(img_BW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        ratio = float(w)/h

        if ratio > 1.1 and ratio < 0.5:
            self.square_cnt[id] = 0
        else:
            self.square_cnt[id] += 1

        if (self.square_cnt[id] >= 15):
            self.markers[id].pose.position.x = 1 / math.tan(math.pi / 8 * h / img.shape[1]) * 0.5
            self.markers[id].pose.position.y = (x + w/2) / img.shape[1]
            self.markers[id].pose.position.z = 0

            self.publisher.publish(self.markers[id])
            self.marked[id] = True

            # rospy.loginfo('marked')

    def _best_fit(self, img):
        try:
            best_id = -1
            best = -1
            keypoint, descriptor = self.surf.detectAndCompute(img, None)

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
            return -1, -1

    def callback(self, img):
        # try:
        # rospy.loginfo("Incoming Picture")
        # convert ROS image to CV image
        img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        id, cnt = self._best_fit(img)

        if id == -1:
            return

        if (cnt < 120):
            self.surf_cnt[id] = 0
        else:
            # rospy.loginfo(str(id))
            self.surf_cnt[id] += 1

        if(self.surf_cnt[id] >= 10 and not self.marked[id]):
            self._mark(id, img)
        # except Exception:
        #     rospy.loginfo("Fail 1")
        #     pass

def main():
    rospy.init_node("image_detect")
    node = ImageDetection()
    rospy.spin()

if __name__ == "__main__":
    main()