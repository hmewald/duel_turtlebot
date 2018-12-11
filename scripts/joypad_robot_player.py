#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist, TwistStamped
from duel_turtlebot.msg import DubinsState
import utils.xbox as xbox
from utils.config import *
from utils.utils import *



class RobotPlayer():

    def __init__(self):

        #### ROS NODE & PUBLISHER
        rospy.init_node('joypad_robot_player')
        rospy.on_shutdown(self.control_stop)

        # For use by control_update
        if environment == 'real':
            pub_topic_str = '/cmd_vel'
        elif environment == 'sim':
            pub_topic_str = '/robot/cmd_vel'
        self.rosbag_pub = rospy.Publisher('/robot_ctrl', TwistStamped, queue_size=5)
        self.cmd_pub = rospy.Publisher(pub_topic_str, Twist, queue_size=5)

        #### SUBSCRIBERS ####
        if environment == 'real':
            sub_topic_str_hum = human_pose_topic
            sub_topic_str_rob = robot_pose_topic
            rospy.Subscriber(human_pose_topic, TransformStamped, self.vicon_human_real_callback)
            rospy.Subscriber(robot_pose_topic, TransformStamped, self.vicon_robot_real_callback)
        elif environment == 'sim':
            rospy.Subscriber('/vicon_human', DubinsState, self.vicon_human_callback)
            rospy.Subscriber('/vicon_robot', DubinsState, self.vicon_robot_callback)

        #### ATTRIBUTES ####
        # Init joystick class
        self.human_z = None
        self.robot_z = None
        self.rel_z = None
        self.joy = xbox.Joystick()


    def vicon_robot_callback(self, msg):
    	# Update own pose
        state = state_from_dubins_state(msg)
        self.robot_z = state
        if self.human_z is not None:
            self.rel_z = compute_rel_state(self.robot_z, self.human_z)


    def vicon_human_callback(self, msg):
    	# Update human adversary's pose
        state = state_from_dubins_state(msg)
        self.human_z = state
        if self.robot_z is not None:
            self.rel_z = compute_rel_state(self.robot_z, self.human_z)


    def vicon_robot_real_callback(self, msg):
    	# Update own pose
        state = state_from_transform_stamped(msg)
        self.robot_z = state
        if self.human_z is not None:
            self.rel_z = compute_rel_state(self.robot_z, self.human_z)


    def vicon_human_real_callback(self, msg):
    	# Update human adversary's pose
        state = state_from_transform_stamped(msg)
        self.human_z = state
        if self.robot_z is not None:
            self.rel_z = compute_rel_state(self.robot_z, self.human_z)


    def control_update(self):
        # Read user input from controller
        stick_x = self.joy.rightX()
        stick_y = self.joy.leftY()
        if stick_y >= 0:
            v_cmd = stick_y * V_MAX
            w_max_v = W_MAX1 + stick_y * (W_MAX2 - W_MAX1)
        else:
            v_cmd = stick_y * R_MAX
            w_max_v = W_MAX1

        w_cmd = -stick_x * w_max_v

        if self.rel_z is not None:
            v_cmd += compute_reciprocal_avoidance_vel(self.rel_z, 'robot')

        if environment == 'real':
            v_cmd *= REAL_WORLD_FACTOR
            w_cmd *= REAL_WORLD_FACTOR

        cmd_msg = Twist()
        cmd_msg.linear.x = v_cmd
        cmd_msg.angular.z = w_cmd
        self.cmd_pub.publish(cmd_msg)
        
        # if environment == 'sim':
        rosbag_msg = TwistStamped()
        rosbag_msg.twist = cmd_msg
        rosbag_msg.header.stamp = rospy.Time.now()
        self.rosbag_pub.publish(rosbag_msg)


    def control_stop(self):
        # Set zero speed and turn rate
        control_msg = Twist()
    	self.cmd_pub.publish(control_msg)


if __name__=="__main__":
    rob = RobotPlayer()

    r = rospy.Rate(50)
    while not rospy.is_shutdown():
        rob.control_update()
        r.sleep()
