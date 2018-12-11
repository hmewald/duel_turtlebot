#!/usr/bin/env python

# Games Master Node for the Turtlebot Duel Game
# Author: Hans Magnus Ewald, October 2018
# Version: 0.1

# Node Description:
# Top of the hierarchy, manages the human_player and robot_player nodes,
# as well as autonomous navigators for each player's bot. The master node
# subscribes to the Vicon tracking system and the player's attack messages.
# It registers hits when attacks connect, then stops both players and
# returns their bots to starting positions. (Alternatively this could just be
# a short time interval where no points can be scored and the robot is disabled)


import rospy

import numpy as np
from std_msgs.msg import Bool, Int32
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from duel_turtlebot.msg import DubinsState
from visualization_msgs.msg import Marker

import sys, select, termios, tty
from utils.utils import *
from utils.config import *


#### GAME RULE PARAMS ####





class GamesMaster():

    def __init__(self):
        rospy.init_node('games_master')

        #### ATTRIBUTES ####

        # Robot can only move when this is
        self.player_active_bool = False

        # Store playing field params
        self.playing_field = PLAYING_FIELD_RECT

        # Robot state, human state
        self.robot_z = None
        self.human_z = None
        self.rel_z = None

        self.dt = 0.05

        self.robot_score = 0.0
        self.human_score = 0.0


        #### PUBLISHERS ####

        # For use by control_update
        self.status_pub = rospy.Publisher('/on_off_game', Bool, queue_size=5)
        self.hit_pub = rospy.Publisher('/player_hit', Int32, queue_size=5)
        self.vis_pub = rospy.Publisher('/visualization_gm', Marker, queue_size=5)


        #### SUBSCRIBERS ####

        # Maintain updated pose knowledge of self and human adversary
        # CHECK AND CORRECT TOPIC NAME AND MESSAGE TYPE BEFORE USE!!!
        if environment == 'real':
            rospy.Subscriber(robot_pose_topic, TransformStamped, self.vicon_robot_real_callback)
            rospy.Subscriber(human_pose_topic, TransformStamped, self.vicon_human_real_callback)
        else:
            rospy.Subscriber('/vicon_robot', DubinsState, self.vicon_robot_callback)
            rospy.Subscriber('/vicon_human', DubinsState, self.vicon_human_callback)



    #### CALLBACKS ####

    def vicon_robot_callback(self, msg):
    	# Update own pose
        # state = state_from_pose_stamped(msg)
        state = np.array([msg.x, msg.y, msg.th])
        # if self.robot_z is None:
        #     print('Initial robot state received!')
        self.robot_z = state
        if self.human_z is not None:
            # if self.rel_z is None:
            #     print('Computing initial relative state!')
            self.rel_z = compute_rel_state(self.robot_z, self.human_z)


    def vicon_human_callback(self, msg):
    	# Update human adversary's pose
        # state = state_from_pose_stamped(msg)
        state = np.array([msg.x, msg.y, msg.th])
        # if self.human_z is None:
        #     print('Initial human state received!')
        self.human_z = state
        if self.robot_z is not None:
            # if self.rel_z is None:
            #     print('Computing initial relative state!')
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


    #### METHODS ####
    # Detect attacks made by both players and resolve them correctly
    def check_hits(self):

        if self.rel_z is not None:
            if not self.check_playing_field(self.robot_z) and self.check_playing_field(self.human_z):
                self.human_score += self.dt
                print("Robot player is out of bounds!            "),
                print(" R: " + str(int(self.robot_score))),
                print(" H: " + str(int(self.human_score))),
                print(chr(13)),
                return None
            elif self.check_playing_field(self.robot_z) and not self.check_playing_field(self.human_z):
                self.robot_score += self.dt
                print("Human player is out of bounds!            "),
                print(" R: " + str(int(self.robot_score))),
                print(" H: " + str(int(self.human_score))),
                print(chr(13)),
                return None
            elif not self.check_playing_field(self.robot_z) and not self.check_playing_field(self.human_z):
                print("Both players are out of bounds!           "),
                print(" R: " + str(int(self.robot_score))),
                print(" H: " + str(int(self.human_score))),
                print(chr(13)),
                return None

            if (self.rel_z[0] >= RHO_G1) and (self.rel_z[0] <= RHO_G2):
                robot_hits = (self.rel_z[2] >= -PHI_G) and (self.rel_z[2] <= PHI_G)
                human_hits = (self.rel_z[1] >= -PHI_G) and (self.rel_z[1] <= PHI_G)

                if robot_hits and not human_hits:
                    # Robot hits human
                    hit_msg = Int32()
                    hit_msg.data = 1
                    self.hit_pub.publish(hit_msg)

                    self.robot_score += self.dt

                    print("Robot player is hitting the human player! "),
                    print(" R: " + str(int(self.robot_score))),
                    print(" H: " + str(int(self.human_score))),
                    print(chr(13)),
                    return None

                elif human_hits and not robot_hits:
                    # Human hits robot
                    hit_msg = Int32()
                    hit_msg.data = 2
                    self.hit_pub.publish(hit_msg)

                    self.human_score += self.dt

                    print("Human player is hitting the robot player! "),
                    print(" R: " + str(int(self.robot_score))),
                    print(" H: " + str(int(self.human_score))),
                    print(chr(13)),
                    return None

                elif human_hits and robot_hits:
                    # Draw scenario - both hit each other
                    hit_msg = Int32()
                    hit_msg.data = 3
                    self.hit_pub.publish(hit_msg)
                    print("Draw! Both players are hitting each other!"),
                    print(" R: " + str(int(self.robot_score))),
                    print(" H: " + str(int(self.human_score))),
                    print(chr(13)),
                    return None


            print("No player is hitting the other!           "),
            print(" R: " + str(int(self.robot_score))),
            print(" H: " + str(int(self.human_score))),
            print(chr(13)),


    def check_playing_field(self, state):
        pf = self.playing_field
        if (state[0] >= pf[0]) and (state[1] >= pf[1]) and (state[0] <= pf[2]) and (state[1] <= pf[3]):
            return True
        else:
            return False


    #### VISUALIZATION ####
    def draw_playing_field(self):
        color = "blue"
        alpha = 1.0
        name = 'playing_field'
        m = colored_marker(color, alpha)
        m.header.frame_id = "map"
        m.ns = name
        m.type = Marker.LINE_STRIP
        m.scale.x = 0.1
        m.points = [Point(self.playing_field[0], self.playing_field[1], 0.), \
                    Point(self.playing_field[2], self.playing_field[1], 0.), \
                    Point(self.playing_field[2], self.playing_field[3], 0.), \
                    Point(self.playing_field[0], self.playing_field[3], 0.), \
                    Point(self.playing_field[0], self.playing_field[1], 0.)]
        m.frame_locked = True
        self.vis_pub.publish(m)


    def draw_score(self, color="white", alpha=1.0):
        m = colored_marker(color, alpha)
        m.header.frame_id = "map"
        m.ns = "score"
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.pose.position.x = self.playing_field[0] + 0.7
        m.pose.position.y = self.playing_field[2] + 0.2
        m.pose.position.z = 1.0
        m.scale.z = 0.2
        m.text = "R: " + format(int(self.robot_score), '02d') + "  H: " + format(int(self.human_score), '02d')
        m.frame_locked = True
        self.vis_pub.publish(m)


    def draw_hit_zones(self):
        if self.robot_z is not None:
            self.draw_ring_segment(self.robot_z, 'robot_hitzone', 'red')
        if self.human_z is not None:
            self.draw_ring_segment(self.human_z, 'human_hitzone', 'green')

    def draw_ring_segment(self, state, name, color):
        phi_list = np.arange(-PHI_G,PHI_G,PHI_G/10)

        m = colored_marker(color, 1.0)
        m.header.frame_id = 'map'
        m.ns = name
        m.type = Marker.LINE_STRIP
        m.scale.x = 0.01
        m.points = [Point(state[0] + RHO_G2 * np.cos(state[2] + phi), state[1] + RHO_G2 * np.sin(state[2] + phi), 0.) for phi in phi_list]
        m.points = m.points + [Point(state[0] + RHO_G1 * np.cos(state[2] - phi), state[1] + RHO_G1 * np.sin(state[2] - phi), 0.) for phi in phi_list]
        m.points.append(Point(state[0] + RHO_G2 * np.cos(state[2] - PHI_G), state[1] + RHO_G2 * np.sin(state[2] - PHI_G), 0.))
        self.vis_pub.publish(m)



if __name__=="__main__":
    gm = GamesMaster()
    r = rospy.Rate(1/gm.dt)

    while not rospy.is_shutdown():
        gm.check_hits()
        if environment == 'sim':
            gm.draw_playing_field()
            gm.draw_score()
            gm.draw_hit_zones()
        r.sleep()
