import rospy
import numpy as np
import tf
import math
from config import *
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from matplotlib.colors import get_named_colors_mapping, to_rgb


COLOR_MAP = get_named_colors_mapping()

def color_to_rgb_float(color):
    if isinstance(color, str):
        color = to_rgb(COLOR_MAP[color])
    if isinstance(color, tuple) and isinstance(color[0], int):
        color = (color[0]/255, color[1]/255, color[2]/255)
    return color


def sigmoid(x):
    return 1/(1+math.exp(-x))


def wrap_to_pi(angle):
    # Project angle into range (-pi, pi]
    return (angle + np.pi) % (2 * np.pi) - np.pi

def state_from_dubins_state(msg):
    return np.array([msg.x, msg.y, msg.th])

def state_from_pose_stamped(msg):
    # Extract state info from PoseStamped format
    pos = msg.pose.position
    orn = msg.pose.orientation
    quat = (orn.x, orn.y, orn.z, orn.w)
    euler = tf.transformations.euler_from_quaternion(quat) # (roll, pitch, yaw)
    return np.array([pos.x, pos.y, euler[2]])

def state_from_transform_stamped(msg):
    # Extract state info from PoseStamped format
    pos = msg.transform.translation
    orn = msg.transform.rotation
    quat = (orn.x, orn.y, orn.z, orn.w)
    euler = tf.transformations.euler_from_quaternion(quat) # (roll, pitch, yaw)
    return np.array([pos.x, pos.y, euler[2]])

def compute_rel_state(rob_z, hum_z):
    # Convert two turtlebot poses into relative state 3-vector (rho, th_h, th_r)
    rel_vec = rob_z[0:2] - hum_z[0:2]
    rho = np.linalg.norm(rel_vec)
    rel_angle = np.arctan2(rel_vec[1],rel_vec[0])
    theta_h = wrap_to_pi(hum_z[2] - rel_angle)
    theta_r = wrap_to_pi(rob_z[2] - (rel_angle + np.pi))
    return np.array([rho,theta_h,theta_r])

def colored_marker(color, alpha=1.0):
    color = color_to_rgb_float(color)
    m = Marker()
    m.color.r = color[0]
    m.color.g = color[1]
    m.color.b = color[2]
    m.color.a = alpha
    m.pose.orientation.w = 1.0
    return m

def compute_reciprocal_avoidance_vel(rel_z, player):
    if rel_z[0] > RHO_C2:
        return 0
    else:
        rho = rel_z[0]
        if player == 'human':
            phi = rel_z[1]
        elif player == 'robot':
            phi = rel_z[2]
        return math.log(max(rho - RHO_C1, 0.001)/RHO_C2) * V_MAX * np.cos(phi)
