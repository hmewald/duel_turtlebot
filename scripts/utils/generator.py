# Class for generating robot action proposals
import numpy as np
from config import *

class Generator():
    def __init__(self):
        return None

    def propose_constant_paths(self, state):
        # Simple trajectory generator assuming constant v and w
        # Create numpy array for each traj, in shape [1,PREDICTION_HORIZON,5]
        dt = 1.0/ROB_F
        u_list = [[-R_MAX, -W_MAX1], [-R_MAX, 0], [-R_MAX, W_MAX1], \
                        [0, -W_MAX1], [0, 0], [0, W_MAX1], \
                        [V_MAX, -W_MAX2], [V_MAX, 0], [V_MAX, W_MAX2]]

        state_action_trajs = []
        for v,w in u_list:
            traj_vw = np.zeros([1,PREDICTION_HORIZON,5])
            traj_vw[0,:,3] = np.repeat(v, PREDICTION_HORIZON)
            traj_vw[0,:,4] = np.repeat(w, PREDICTION_HORIZON)
            prev_state = state
            for i in range(0,PREDICTION_HORIZON):
                new_state = np.array([prev_state[0] + dt*np.cos(prev_state[2])*traj_vw[0,i,3], \
                                    prev_state[1] + dt*np.sin(prev_state[2])*traj_vw[0,i,3],\
                                    prev_state[2] + dt*traj_vw[0,i,4]])
                traj_vw[0,i,0:3] = new_state
                prev_state = new_state

            state_action_trajs.append(traj_vw)

        return state_action_trajs
