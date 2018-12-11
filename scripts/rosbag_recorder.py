import rospy
from msg import DubinsState
from geometry_msgs import Twist




# Define our autonomous robot navigator class
class RosbagRecorder():

    def __init__(self):
        rospy.init_node('rosbag_recorder')

        #### SUBSCRIBERS ####
        rospy.Subscriber('/vicon_robot', PoseStamped, self.vicon_robot_callback)
        rospy.Subscriber('/vicon_human', PoseStamped, self.vicon_human_callback)


        #### PUBLISHERS ####
        self.robot_pub = rospy.Publisher('/robot_state', DubinsState, queue_size=5)
        self.human_pub = rospy.Publisher('/human_state', DubinsState, queue_size=5)

    #### CALLBACKS ####
    def vicon_robot_callback(self, msg):
    	# Publish 
        state = state_from_pose_stamped(msg)

        bag_msg = DubinsState
        bag_msg.x = state[0]
        bag_msg.y = state[1]
        bag_msg.th = state[2]
        self.human_pub.publish(bag_msg)


    def vicon_human_callback(self, msg):
    	# Update human adversary's pose
        state = state_from_pose_stamped(msg)

        bag_msg = DubinsState
        bag_msg.x = state[0]
        bag_msg.y = state[1]
        bag_msg.th = state[2]
        self.human_pub.publish(bag_msg)
