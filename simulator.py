#!/usr/bin/env python

# Simulator Node for the Turtlebot Duel Game
# Author: Hans Magnus Ewald, October 2018
# Version: 0.1

# Node Description:
# Simple Node that keeps track of system dynamics of two turtlebot players by subscribing
# to control updates and publishing system state messages. Use rViz to visualize simulation
# live.


import rospy, tf
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from duel_turtlebot.msg import DubinsState
import sys, os, select, termios, tty
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("../utils.utils.py"))))
from utils.utils import *



#### NON CLASS FUNCTIONS ####

# Generate Geometry Msg Pose from state 3-array
def pose_msg_from_state(z_arr):
    msg = PoseStamped()

    # Enter cartesian position
    msg.pose.position.x = z_arr[0]
    msg.pose.position.y = z_arr[1]

    # Transform yaw to quaternion
    quat = tf.transformations.quaternion_from_euler(0.0, 0.0, z_arr[2])
    msg.pose.orientation.x = quat[0]
    msg.pose.orientation.y = quat[1]
    msg.pose.orientation.z = quat[2]
    msg.pose.orientation.w = quat[3]

    # Fill in header
    msg.header.frame_id = 'map'

    return msg


def compute_rel_state(rob_z, hum_z):
    # Convert two turtlebot poses into relative state 3-vector (rho, th_h, th_r)
    rel_vec = rob_z[0:2] - hum_z[0:2]
    rho = np.linalg.norm(rel_vec)
    rel_angle = np.arctan2(rel_vec[1],rel_vec[0])
    theta_h = wrap_to_pi(hum_z[2] - rel_angle)
    theta_r = wrap_to_pi(rob_z[2] - (rel_angle + np.pi))
    return np.array([rho,theta_h,theta_r])



class Simulator():

    def __init__(self):

        rospy.init_node('simulator')


        #### ATTRIBUTES ####

        # State and control for robot player and human player respectively
        self.robot_z = np.array([0.0,-1.0, np.pi/2])
        self.robot_u = np.array([0.0,0.0])

        self.human_z = np.array([0.0,1.0, -np.pi/2])
        self.human_u = np.array([0.0,0.0])

        # Time discretization interval in seconds
        self.dt = 0.02


        #### PUBLISHERS ####

        # For use by control_update
        self.robot_pub = rospy.Publisher('/vicon_robot', DubinsState, queue_size=5)
        self.human_pub = rospy.Publisher('/vicon_human', DubinsState, queue_size=5)
        self.vis_pub = rospy.Publisher('/visualization_sim', Marker, queue_size=5)


        #### SUBSCRIBERS ####

        # Maintain updated knowledge of control commands
        rospy.Subscriber('/robot/cmd_vel', Twist, self.robot_cmd_callback)
        rospy.Subscriber('/human/cmd_vel', Twist, self.human_cmd_callback)


    #### CALLBACKS ####

    # Control updates
    def robot_cmd_callback(self, msg):
        data = msg
        self.robot_u[0] = data.linear.x
        self.robot_u[1] = data.angular.z

    def human_cmd_callback(self, msg):
        data = msg
        self.human_u[0] = data.linear.x
        self.human_u[1] = data.angular.z


    #### METHODS ####

    # Forward integration state update
    def state_update(self):
        # Update robot state
        x_r = self.robot_z
        u_r = self.robot_u
        dx_r = self.dt * np.array([u_r[0] * np.cos(x_r[2]), u_r[0] * np.sin(x_r[2]), u_r[1]])
        self.robot_z = x_r + dx_r

        # Update human state
        x_h = self.human_z
        u_h = self.human_u
        dx_h = self.dt * np.array([u_h[0] * np.cos(x_h[2]), u_h[0] * np.sin(x_h[2]), u_h[1]])
        self.human_z = x_h + dx_h

        t_now = rospy.Time.now()

        # Publish new states
        rob_msg = DubinsState() # pose_msg_from_state(self.robot_z)
        rob_msg.x = self.robot_z[0]
        rob_msg.y = self.robot_z[1]
        rob_msg.th = self.robot_z[2]
        rob_msg.header.stamp = t_now
        rob_msg.header.frame_id = 'map'
        self.robot_pub.publish(rob_msg)
        hum_msg = DubinsState() # pose_msg_from_state(self.human_z)
        hum_msg.x = self.human_z[0]
        hum_msg.y = self.human_z[1]
        hum_msg.th = self.human_z[2]
        hum_msg.header.stamp = t_now
        rob_msg.header.frame_id = 'map'
        self.human_pub.publish(hum_msg)

    #### VISUALIZATION ####
    def draw_turtles(self):
        self.draw_turtle(self.human_z, 'human_turtle', 'green')
        self.draw_turtle(self.robot_z, 'robot_turtle', 'red')


    def draw_turtle(self, state, name, color):
        m = colored_marker(color, 1.0)
        m.header.frame_id = "map"
        m.ns = name
        m.id = 0
        m.type = Marker.CYLINDER
        m.pose.position.x = state[0]
        m.pose.position.y = state[1]
        m.pose.position.z = 0.0
        m.scale.x = 0.15
        m.scale.y = 0.15
        m.scale.z = 0.15
        m.frame_locked = True
        self.vis_pub.publish(m)


# Run node loop
if __name__=="__main__":
    sim = Simulator()
    r = rospy.Rate(1/sim.dt)

    while not rospy.is_shutdown():
        sim.state_update()
        sim.draw_turtles()
        r.sleep()
