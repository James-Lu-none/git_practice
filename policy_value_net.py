from keras.layers import *
from keras.engine.training import Model
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model
from keras.models import Sequential
from keras.utils import np_utils

import numpy as np
import pickle


class PolicyValueNet():
    """policy-value network """

    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        self.create_policy_value_net(model_file)

    def create_policy_value_net(self,model_file):
        self.model = load_model(model_file)
        def policy_value(state_input):
            state_input_union = np.array(state_input)
            results = self.model.predict_on_batch(state_input_union)
            return results
        self.policy_value = policy_value

    def policy_value_fn(self, board):
        legal_positions = board.availables
        current_state = board.current_state()
        act_probs, value = self.policy_value(
            current_state.reshape(-1, 5, self.board_width, self.board_height))
        
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        # print(list(act_probs))
        # act_probs.flatten()[legal_positions] takes only the probs for legel_positions, others are zero
        return act_probs, value[0][0]

    def save_model(self, model_file):
        self.model.save(model_file, save_format='h5')
        
        