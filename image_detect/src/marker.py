#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker

name = ["Obama", "Avril", "Levi", "Bloom", "chinese", "No picture"]
marker_arr = []
for i in range(0,4):
    marker = Marker()
    marker.header.frame_id = "camera_link"
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

def main():
    marker_arr[0].pose.position.x = 0
    marker_arr[0].pose.position.y = 0
    marker_arr[0].pose.position.z = 0
    print "Hard code name marker"
    rospy.init_node('vmker')
    pub = rospy.Publisher('visualization_marker', Marker, queue_size=100)
    rate = rospy.Rate(1) # 1hz
    pub.publish(marker_arr[0])
    while not rospy.is_shutdown():
        pub.publish(marker_arr[0])
        rospy.spin()
        #rate.sleep()

if __name__ == '__main__':
    main()
