import numpy as np

environment = 'sim'

PHI_G = np.pi/12
RHO_G1 = 0.15
RHO_G2 = 0.4

RHO_C1 = 0.11
RHO_C2 = 0.18

W_MAX1 = 0.15
W_MAX2 = 0.45
V_MAX = 0.15
R_MAX = 0.06

ROB_F = 10

REAL_WORLD_FACTOR = 1.0

PLAYING_FIELD_RECT = (-1.5, -1.5, 1.5, 1.5)

human_pose_topic = 'vicon/joe_burger/joe_burger'
robot_pose_topic = 'vicon/mo_burger/mo_burger'

MIN_HISTORY_LENGTH = 5
PREDICTION_HORIZON = 15
HISTORY_BUFFER = 1000
MODEL_DIR = '/home/hansmagnus/catkin_ws/src/ml_duel_turtlebot/models/1543969388'
