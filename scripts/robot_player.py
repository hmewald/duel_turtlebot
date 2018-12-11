#!/usr/bin/env python

# Games Master Node for the Turtlebot Duel Game
# Author: Hans Magnus Ewald, October 2018
# Version: 0.1

# Node Description:



import rospy
import numpy as np
import tensorflow as tf
from math import log
from scipy.io import loadmat
# from scipy.interpolate import RegularGridInterpolator
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Twist, TwistStamped, Point, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker
from duel_turtlebot.msg import DubinsState
import sys, select, termios, tty
from utils.utils import *
from utils.config import *
from utils.generator import *
from utils.scorer import *



# Define our autonomous robot navigator class
class RobotPlayer():

    def __init__(self):
        rospy.init_node('robot_player')
        data_path_str = '/home/hansmagnus/catkin_ws/src/duel_turtlebot/data' # '../data'


        #### ATTRIBUTES ####

        # Grid on wchich value function and its derivatives are defined
        self.hj_grid = {}
        grid_obj = loadmat(data_path_str + '/grid.mat')['g'][0][0]
        grid_llims = grid_obj[1]
        grid_ulims = grid_obj[2]
        grid = grid_obj[7]
        self.hj_grid['llims'] = grid_llims # Shape is (3,1)
        self.hj_grid['ulims'] = grid_ulims # Shape is (3,1)
        self.hj_grid['rho'] = grid[0,0]
        self.hj_grid['phi_h'] = grid[1,0]
        self.hj_grid['phi_r'] = grid[2,0]
        self.hj_grid['shape'] = np.shape(grid[0,0])
        del grid_obj, grid_llims, grid_ulims, grid

        # Time horizon for the avoid value function is T = t * 0.05s
        t_ind = -1

        # Load value function and gradient for avoid set
        avoid_val = loadmat(data_path_str + '/avoid_values.mat')['data']
        self.avoid_value_fnc = avoid_val[:,:,:,t_ind]
        self.avoid_grad = {}
        avoid_prime = loadmat(data_path_str + '/avoid_values_grad.mat')['data_prime']
        self.avoid_grad['dv_drho'] = avoid_prime[0,0][:,:,:,t_ind]
        self.avoid_grad['dv_dphi_h'] = avoid_prime[1,0][:,:,:,t_ind]
        self.avoid_grad['dv_dphi_r'] = avoid_prime[2,0][:,:,:,t_ind]
        del avoid_prime, avoid_val

        # Load value function and gradient for reach set
        # self.reach_value_fnc = loadmat(data_path_str + '/avoid_values.mat')['goal']
        # self.reach_grad = {}
        # reach_prime = loadmat(data_path_str + '/reach_values_grad.mat')['data_prime']
        # self.reach_grad['dv_drho'] = reach_prime[0,0][:,:,:,t_ind]
        # self.reach_grad['dv_dphi_h'] = reach_prime[1,0][:,:,:,t_ind]
        # self.reach_grad['dv_dphi_r'] = reach_prime[2,0][:,:,:,t_ind]
        # del reach_prime

        # Time horizons for different reach value functions are T_i = t_i * 0.05s
        t_ind_short = 1
        t_ind_mid = 21
        t_ind_long = -1
        self.time_hor_rho_bins = (0.85, 1.6)
        self.time_hor_phi_h_bins = (np.pi/4,np.pi/8)

        # Load value function and gradient for combined reachavoid policy
        reachavoid_val = loadmat(data_path_str + '/reachavoid_values.mat')['data']
        self.reachavoid_value_fnc = []
        self.reachavoid_value_fnc.append(reachavoid_val[:,:,:,t_ind_short])
        self.reachavoid_value_fnc.append(reachavoid_val[:,:,:,t_ind_mid])
        self.reachavoid_value_fnc.append(reachavoid_val[:,:,:,t_ind_long])
        reachavoid_prime = loadmat(data_path_str + '/reachavoid_values_grad.mat')['data_prime']
        self.reachavoid_grad = []
        reachavoid_grad_short = {}
        reachavoid_grad_short['dv_drho'] = reachavoid_prime[0,0][:,:,:,t_ind_short]
        reachavoid_grad_short['dv_dphi_h'] = reachavoid_prime[1,0][:,:,:,t_ind_short]
        reachavoid_grad_short['dv_dphi_r'] = reachavoid_prime[2,0][:,:,:,t_ind_short]
        self.reachavoid_grad.append(reachavoid_grad_short)
        reachavoid_grad_mid = {}
        reachavoid_grad_mid['dv_drho'] = reachavoid_prime[0,0][:,:,:,t_ind_mid]
        reachavoid_grad_mid['dv_dphi_h'] = reachavoid_prime[1,0][:,:,:,t_ind_mid]
        reachavoid_grad_mid['dv_dphi_r'] = reachavoid_prime[2,0][:,:,:,t_ind_mid]
        self.reachavoid_grad.append(reachavoid_grad_mid)
        reachavoid_grad_long = {}
        reachavoid_grad_long['dv_drho'] = reachavoid_prime[0,0][:,:,:,t_ind_long]
        reachavoid_grad_long['dv_dphi_h'] = reachavoid_prime[1,0][:,:,:,t_ind_long]
        reachavoid_grad_long['dv_dphi_r'] = reachavoid_prime[2,0][:,:,:,t_ind_long]
        self.reachavoid_grad.append(reachavoid_grad_long)
        del reachavoid_prime, reachavoid_val

        # Interpolator, define with avoid value function
        # test_x = np.linspace(self.hj_grid['llims'][0], self.hj_grid['ulims'][0], self.hj_grid['shape'][0])
        # test_y = np.linspace(self.hj_grid['llims'][1], self.hj_grid['ulims'][1], self.hj_grid['shape'][1])
        # test_z = np.linspace(self.hj_grid['llims'][2], self.hj_grid['ulims'][2], self.hj_grid['shape'][2])
        # test_grid = np.meshgrid(test_x, test_y, test_z, indexing='ij', sparse=True)
        # print self.hj_grid['phi_r'][:,0,0]
        # print self.hj_grid['phi_r'][0,:,0]
        # print self.hj_grid['phi_r'][0,0,:]
        # self.grid_interp = RegularGridInterpolator(test_grid, self.avoid_value_fnc)

        # Proposed robot future generator
        self.future_generator = Generator()
        self.future_scorer = Scorer()

        # Robot can only move when this is true
        self.player_active_bool = True

        # Robot state, robot action, human state, human action, rel state
        self.robot_z = None
        self.robot_u = None
        self.human_z = None
        self.human_d = None
        self.rel_z = None

        self.max_v = V_MAX
        self.min_v = R_MAX
        self.max_w1 = W_MAX1
        self.max_w2 = W_MAX2

        # self.control_region = 0
        self.region_map = {0:'inactive    ', 1:'reachability', 2:'forward     ', 3:'right       ', 4:'left        ', 5:'reverse     '}

        self.phi_f = np.pi
        self.rho_c = 0.15

        # Reach Avoid merging params
        self.avoid_val_hi = 0.25
        self.avoid_val_lo = 0.0

        # Playing field limits in rectangular form [x_min, y_min, x_max, y_max]
        self.playing_field = PLAYING_FIELD_RECT
        self.edge_margin = 0.5
        self.edge_force = 1.0

        rospy.on_shutdown(self.control_stop)


        #### PUBLISHERS ####

        # For use by control_update
        if environment == 'real':
            pub_topic_str = '/cmd_vel'
        elif environment == 'sim':
            pub_topic_str = '/robot/cmd_vel'
        self.rosbag_pub = rospy.Publisher('/robot_ctrl', TwistStamped, queue_size=5)
        self.control_pub = rospy.Publisher(pub_topic_str, Twist, queue_size=5)
        self.vis_pub = rospy.Publisher('/visualization_rob', Marker, queue_size=5)


        #### SUBSCRIBERS ####

        # Maintain updated pose knowledge of self and human adversary
        if environment == 'real':
            rospy.Subscriber('/human_ctrl', TwistStamped, self.human_cmd_callback)
            rospy.Subscriber(robot_pose_topic, TransformStamped, self.vicon_robot_real_callback)
            rospy.Subscriber(human_pose_topic, TransformStamped, self.vicon_human_real_callback)
        else:
            rospy.Subscriber('/human_ctrl', TwistStamped, self.human_cmd_callback)
            rospy.Subscriber('/vicon_robot', DubinsState, self.vicon_robot_callback)
            rospy.Subscriber('/vicon_human', DubinsState, self.vicon_human_callback)


    #### METHODS ####

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


    def human_cmd_callback(self, msg):
        self.human_d = np.array([msg.twist.linear.x, msg.twist.angular.z])


    def control_update(self):
        control_region = self.valid_state()

        if control_region == 1:
            v_cmd, w_cmd = self.reachability_control()
        elif control_region == 2:
            v_cmd, w_cmd = self.forward_control()
        elif control_region == 3:
            v_cmd, w_cmd = self.right_control()
        elif control_region == 4:
            v_cmd, w_cmd = self.left_control()
        elif control_region == 5:
            v_cmd, w_cmd = self.reverse_control()
        else:
            v_cmd = 0.0
            w_cmd = 0.0

        v_cmd += compute_reciprocal_avoidance_vel(self.rel_z, 'robot')

        self.robot_u = np.array([v_cmd, w_cmd])
        control_msg = Twist()
        control_msg.linear.x = v_cmd
        control_msg.angular.z = w_cmd
    	self.control_pub.publish(control_msg)

        # print current controller region
        # print('Current controller region: ' + self.region_map[control_region]),
        # print(chr(13)),

        # if environment == 'sim':
        rosbag_msg = TwistStamped()
        rosbag_msg.twist = control_msg
        rosbag_msg.header.stamp = rospy.Time.now()
        self.rosbag_pub.publish(rosbag_msg)


    def valid_state(self):
        # Check if relative state is up to date and within limits, decides which controller will be used
        if (self.rel_z[1] >= -self.phi_f) and (self.rel_z[1] <= self.phi_f):
            if (self.rel_z[0] >= max(self.rho_c, self.hj_grid['llims'][0])) and (self.rel_z[0] <= self.hj_grid['ulims'][0]):
                if (self.rel_z[2] >= -self.phi_f) and (self.rel_z[2] <= self.phi_f):
                    return 1
                elif self.rel_z[2] > self.phi_f: #self.hj_grid['ulims'][2]:
                    # Robot facing too far left -> turn right controller
                    return 3
                elif self.rel_z[2] < -self.phi_f:#self.hj_grid['llims'][2]:
                    # Robot facing too far right -> turn left controller
                    return 4
            elif self.rel_z[0] > self.hj_grid['ulims'][0]:
                # Robot too far away from opponent -> drive closer controller
                return 2
            elif self.rel_z[0] < max(self.rho_c, self.hj_grid['llims'][0]):
                # Robot too close to opponent -> reverse controller
                return 5
        # Opponent facing the wrong way -> do nothing
        return 0


    def reachability_control(self):
    	# Use Value Function to get optimal control as per reachability analysis

        # DEBUG: Print current state as verification
        # print(self.rel_z)

        # Find indices of current position on the discrete grid
        grid_inds = tuple([int((self.hj_grid['shape'][i]-1) * (self.rel_z[i] - self.hj_grid['llims'][i])/(self.hj_grid['ulims'][i] - self.hj_grid['llims'][i])) \
                            for i in [0,1,2]])

        # Get avoid value
        va_val = self.avoid_value_fnc[grid_inds]
        # va_val = self.interpolate_hj_grid(self.avoid_value_fn, self.rel_z)
        # Compute avoid value function gradient over command velocity
        dva_dv = - self.avoid_grad['dv_drho'][grid_inds] * np.cos(self.rel_z[2]) + \
        self.avoid_grad['dv_dphi_h'][grid_inds] * np.sin(self.rel_z[2]) / self.rel_z[0] + \
        self.avoid_grad['dv_dphi_r'][grid_inds] * np.sin(self.rel_z[2]) / self.rel_z[0]
        dva_dw = self.avoid_grad['dv_dphi_r'][grid_inds]

        # Compute reach value function gradient over command velocity
        # dvr_dv = - self.reach_grad['dv_drho'][grid_inds] * np.cos(self.rel_z[2]) + \
        # self.reach_grad['dv_dphi_h'][grid_inds] * np.sin(self.rel_z[2]) / self.rel_z[0] + \
        # self.reach_grad['dv_dphi_r'][grid_inds] * np.sin(self.rel_z[2]) / self.rel_z[0]
        # dvr_dw = self.reach_grad['dv_dphi_r'][grid_inds]

        # Get reachavoid value
        if (self.rel_z[0] < self.time_hor_rho_bins[0]) and (np.fabs(self.rel_z[1]) > self.time_hor_phi_h_bins[0]):
            T_h = 0
        elif (self.rel_z[0] >= self.time_hor_rho_bins[1]) or (np.fabs(self.rel_z[1]) < self.time_hor_phi_h_bins[1]):
            T_h = 2
        else:
            T_h = 1

        vra_val = self.reachavoid_value_fnc[T_h][grid_inds]
        # Compute reachavoid value function gradient over command velocity
        dvra_dv = - self.reachavoid_grad[T_h]['dv_drho'][grid_inds] * np.cos(self.rel_z[2]) + \
        self.reachavoid_grad[T_h]['dv_dphi_h'][grid_inds] * np.sin(self.rel_z[2]) / self.rel_z[0] + \
        self.reachavoid_grad[T_h]['dv_dphi_r'][grid_inds] * np.sin(self.rel_z[2]) / self.rel_z[0]
        dvra_dw = self.reachavoid_grad[T_h]['dv_dphi_r'][grid_inds]

        # Print current value functions
        # print('Avoid value: ' + format(va_val, '05f') + ' Reach value: ' + format(vra_val, '05f')),
        # print(chr(13)),

        # Combine into total gradients over v and w
        if(va_val > self.avoid_val_hi):
            dv_dv = -dvra_dv
            dv_dw = -dvra_dw
        elif (va_val <= self.avoid_val_hi) and (va_val > self.avoid_val_lo):
            theta = (va_val - self.avoid_val_lo)/(self.avoid_val_hi - self.avoid_val_lo)
            dv_dv = (1-theta) * dva_dv + theta * (-dvra_dv)
            dv_dw = (1-theta) * dva_dw + theta * (-dvra_dw)
        else:
            dv_dv = dva_dv
            dv_dw = dva_dw

        # Factor in playing field constraint
        edge_dist1 = self.robot_z[0] - self.playing_field[0]
        edge_dist2 = self.robot_z[1] - self.playing_field[1]
        edge_dist3 = self.playing_field[2] - self.robot_z[0]
        edge_dist4 = self.playing_field[3] - self.robot_z[1]
        if edge_dist1 <= min([edge_dist2, edge_dist3, edge_dist4]):
            edge_d = edge_dist1
            edge_th = 0
        elif edge_dist2 <= min([edge_dist1, edge_dist3, edge_dist4]):
            edge_d = edge_dist2
            edge_th = np.pi/2
        elif edge_dist3 <= min([edge_dist1, edge_dist2, edge_dist4]):
            edge_d = edge_dist3
            edge_th = np.pi
        elif edge_dist4 <= min([edge_dist1, edge_dist2, edge_dist3]):
            edge_d = edge_dist4
            edge_th = -np.pi/2

        if edge_d < self.edge_margin:
            # Add -log potential barrier
            dve_dv = np.cos(self.robot_z[2] - edge_th) * self.edge_force * (-1) * log(max(edge_d,0.001)/self.edge_margin)
        else:
            dve_dv = 0
        dv_dv += dve_dv

        # print(str(edge_d) + ' ' + str(edge_th) + ' ' + str(dve_dv)),
        # print(chr(13)),

        # Set v_cmd accordingly
        if dv_dv > 0.0:
            v_cmd = self.max_v
            w_max_v = self.max_w2
        elif dv_dv < 0.0:
            v_cmd = -self.min_v
            w_max_v = self.max_w1
        else:
            v_cmd = 0.0
            w_max_v = self.max_w1

        # Set w_cmd accordingly
        if dv_dw > 0.0:
            w_cmd = w_max_v
        elif dv_dw < 0.0:
            w_cmd = -w_max_v
        else:
            w_cmd = 0.0

        if environment == 'real':
            v_cmd *= REAL_WORLD_FACTOR
            w_cmd *= REAL_WORLD_FACTOR

        return v_cmd, w_cmd


    def forward_control(self):
        # Trigonometric heuristic to close with the opponent
        v_cmd = max(np.cos(self.rel_z[2]),0) * self.max_v
        w_cmd = self.max_w2 * (-1) * np.sign(self.rel_z[2])
        if environment == 'real':
            v_cmd *= REAL_WORLD_FACTOR
            w_cmd *= REAL_WORLD_FACTOR

        return v_cmd, w_cmd


    def right_control(self):
        # Maximal right turn
        w_cmd = -self.max_w2
        if environment == 'real':
            w_cmd *= REAL_WORLD_FACTOR

        return 0.0, w_cmd


    def left_control(self):
        # Maximal left turn
        w_cmd = self.max_w2
        if environment == 'real':
            w_cmd *= REAL_WORLD_FACTOR

        return 0.0, w_cmd


    def reverse_control(self):
        # Trigonometric heuristic to close with the opponent
        v_cmd = (-1) * max(np.cos(self.rel_z[2]),0) * self.max_v
        w_cmd = self.max_w1 * (-1) * np.sign(self.rel_z[2])

        if environment == 'real':
            v_cmd *= REAL_WORLD_FACTOR
            w_cmd *= REAL_WORLD_FACTOR

        return v_cmd, w_cmd


    def control_stop(self):
        # Set zero speed and turn rate
        control_msg = Twist()
    	self.control_pub.publish(control_msg)


    # def interpolate_hj_grid(self, value_fnc, point):
    #     self.grid_interp.values = value_fnc
    #     return self.grid_interp(point)



    #### VISUALIZATION ####
    def draw_reach_values(self):
        if self.rel_z is not None:
            # phi_r_ind = int((self.hj_grid['shape'][2]-1) * (self.rel_z[2] - self.hj_grid['llims'][2])/(self.hj_grid['ulims'][2] - self.hj_grid['llims'][2]))
            phi_r_ind = int((self.hj_grid['shape'][2]-1)/2)
            vra_slice_short = self.reachavoid_value_fnc[0][:,:,phi_r_ind]
            slice_ext_short =(np.amin(vra_slice_short), np.amax(vra_slice_short))
            vra_slice_mid = self.reachavoid_value_fnc[1][:,:,phi_r_ind]
            slice_ext_mid = (np.amin(vra_slice_mid), np.amax(vra_slice_mid))
            vra_slice_long = self.reachavoid_value_fnc[2][:,:,phi_r_ind]
            slice_ext_long =(np.amin(vra_slice_long), np.amax(vra_slice_long))
            print('Min reach value: ' + format(slice_ext_mid[0], '05f') + ' Max reach value: ' + format(slice_ext_mid[1], '05f')),
            print(chr(13)),
            m = Marker()
            m.header.frame_id = "map"
            m.ns = "reach_val"
            m.type = Marker.POINTS
            m.scale.x = 0.02
            m.scale.y = 0.02
            m.points = []
            m.colors = []
            m.frame_locked = True

            for i in range(self.hj_grid['shape'][0]):
                for j in range(self.hj_grid['shape'][1]):
                    th_ij = self.human_z[2] + self.hj_grid['phi_h'][i,j,phi_r_ind]
                    x_ij = self.human_z[0] + np.cos(th_ij) * self.hj_grid['rho'][i,j,phi_r_ind]
                    y_ij = self.human_z[1] + np.sin(th_ij) * self.hj_grid['rho'][i,j,phi_r_ind]
                    r_ij = (vra_slice_short[i,j] - slice_ext_short[0])/(slice_ext_short[1] - slice_ext_short[0])
                    g_ij = (vra_slice_mid[i,j] - slice_ext_mid[0])/(slice_ext_mid[1] - slice_ext_mid[0])
                    b_ij = (vra_slice_long[i,j] - slice_ext_long[0])/(slice_ext_long[1] - slice_ext_long[0])
                    m.points.append(Point(x_ij,y_ij,0.))
                    m.colors.append(ColorRGBA(r_ij, g_ij, b_ij, 1.0))

            self.vis_pub.publish(m)


    def propose_future(self):
        future_trajs = self.future_generator.propose_constant_paths(self.robot_z)
        robot_state = np.array([self.robot_z[0], self.robot_z[1], self.robot_z[2], self.robot_u[0], self.robot_u[1]])
        human_state = np.array([self.human_z[0], self.human_z[1], self.human_z[2], self.human_d[0], self.human_d[1]])
        human_trajs = self.future_scorer.score_trajectories(robot_state, human_state, future_trajs)
        n_traj = len(future_trajs)

        col_a = (255, 255, 0) # Yellow = Red + Green
        col_b = (255, 0, 255) # Violet = Red + Blue
        color_list = [tuple([int((i*col_a[j] + (n_traj-1-i)*col_b[j])/n_traj) for j in range(3)]) for i in range(n_traj)]

        for i in range(n_traj):
            traj = future_trajs[i]
            human_traj = human_trajs[i]
            n_pts = traj.shape[1]

            m = colored_marker(color_list[i], 1.0)
            m.header.frame_id = 'map'
            m.ns = "proposed_traj_{}".format(i)
            m.type = Marker.LINE_STRIP
            m.scale.x = 0.01
            m.points = [Point(traj[0,j,0], traj[0,j,1], 0.) for j in range(n_pts)]
            self.vis_pub.publish(m)

            m = colored_marker(color_list[i], 1.0)
            m.header.frame_id = 'map'
            m.ns = "human_traj_{}".format(i)
            m.type = Marker.LINE_STRIP
            m.scale.x = 0.01
            m.points = [Point(human_traj[0,j,0], human_traj[0,j,1], 0.) for j in range(n_pts)]
            self.vis_pub.publish(m)



if __name__=="__main__":
    rob_player = RobotPlayer()
    r = rospy.Rate(ROB_F)

    while not rospy.is_shutdown():
        if (rob_player.rel_z is not None):
            rob_player.control_update()
            if (rob_player.human_d is not None):
                rob_player.propose_future()
        # rob_player.draw_reach_values()
        r.sleep()
