# Class to use human behavior model to generate human response sequences to proposed robot sequences
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from config import *



def standardize(tensor, mean, std):
    tile_ct = int(tensor.shape[-1].value / mean.shape[-1].value)
    return (tensor  - tf.tile(mean, [tile_ct])) / tf.tile(std, [tile_ct])


def unstandardize(tensor, mean, std, include_bias=True):
    tile_ct = int(tensor.shape[-1].value / mean.shape[-1].value)
    if include_bias:
        return tensor * tf.tile(std, [tile_ct]) + tf.tile(mean, [tile_ct])
    else:
        return tensor * tf.tile(std, [tile_ct])


def forward_integrate(state, future_actions):
    # state, future_actions should be np arrays
    # state should have 5 elements
    # future_actions should have dims [1, PREDICTION_HORIZON, 2]
    traj = np.zeros([1,PREDICTION_HORIZON,5])
    traj[0,:,3:5] = future_actions
    prev_state = state[0:3]
    for i in range(0,PREDICTION_HORIZON):
        new_state = np.array([prev_state[0] + dt*np.cos(prev_state[2])*traj[0,i,3], \
                            prev_state[1] + dt*np.sin(prev_state[2])*traj[0,i,3],\
                            prev_state[2] + dt*traj[0,i,4]])
        traj[0,i,0:3] = new_state
        prev_state = new_state

    return traj




class Scorer(object):
    def __init__(self):
        self.car1_history = []
        self.car2_history = []

        with tf.Graph().as_default() as g:
            self.sess = tf.Session()
            tf.saved_model.loader.load(self.sess,[tf.saved_model.tag_constants.SERVING], MODEL_DIR)
            self.y = g.get_tensor_by_name("outputs/y:0")
            self.car1 =  g.get_tensor_by_name("car1:0")
            self.car2 =  g.get_tensor_by_name("car2:0")
            self.extras = g.get_tensor_by_name("extras:0")
            self.car1_future = g.get_tensor_by_name("car1_future:0")

    def score_trajectories(self, car1_state, car2_state, car1_future_trajs):
        # car1_state, car2_state need to be np arrays with 5 elements
        # car1_future_trajs needs to be a list of np arrays shaped [1,timesteps,5]
        print(chr(13)),
        car1_reshaped = np.reshape(car1_state, [1,1,5])
        car2_reshaped = np.reshape(car2_state, [1,1,5])
        self.car1_history.append(car1_reshaped)
        # print("History1 "),
        # print(len(self.car1_history)),
        if len(self.car1_history) > HISTORY_BUFFER:
            self.car1_history.pop(0)
        elif len(self.car1_history) < MIN_HISTORY_LENGTH:
            while len(self.car1_history) < MIN_HISTORY_LENGTH:
                self.car1_history.append(self.car1_history[-1])

        self.car2_history.append(car2_reshaped)
        # print("History2 "),
        # print(len(self.car2_history)),
        if len(self.car2_history) > HISTORY_BUFFER:
            self.car2_history.pop(0)
        elif len(self.car2_history) < MIN_HISTORY_LENGTH:
            while len(self.car2_history) < MIN_HISTORY_LENGTH:
                self.car2_history.append(self.car2_history[-1])

        n_prop = len(car1_future_trajs)

        feed_dict = {}
        feed_dict["car1:0"] = np.repeat(np.concatenate(self.car1_history, axis=1), n_prop, axis=0)
        feed_dict["car2:0"] = np.repeat(np.concatenate(self.car2_history, axis=1), n_prop, axis=0)
        feed_dict["car1_future:0"] = np.concatenate(car1_future_trajs, axis=0)
        feed_dict["extras:0"] = np.zeros([n_prop, len(self.car1_history), 0])
        feed_dict["traj_lengths:0"] = np.repeat(np.array([len(self.car1_history)]), n_prop, axis=0)
        feed_dict["sample_ct:0"] = np.repeat(np.array([1]), n_prop, axis=0)


        print('car1: '),
        print(np.shape(feed_dict["car1:0"])),
        print(' car2: '),
        print(np.shape(feed_dict["car2:0"])),
        print(' car1 future: '),
        print(np.shape(feed_dict["car1_future:0"])),
        print(' traj_lengths: '),
        print(feed_dict["traj_lengths:0"]),
        print(' sample_ct: '),
        print(feed_dict["sample_ct:0"]),


        action_pred = self.sess.run(self.y, feed_dict=feed_dict)

        car2_futures = []
        for i in range(n_prop):
            actions_i = self.y[0, 0:PREDICTION_HORIZON, 0:2]
            traj_i = forward_integrate(car2_state, actions_i)
            car2_futures.append(traj_i)

        return car2_futures
