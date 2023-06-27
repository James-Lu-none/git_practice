import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_AI import MCTSPlayer
from policy_value_net import PolicyValueNet  # Keras
import sys
import math
from pathlib import Path
import os
import time
import pickle
import string

def generate_random_string(length):
    # Define the characters to include in the random string
    characters = string.ascii_letters + string.digits

    # Generate a random string of the specified length
    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string

current_file_path = Path(get_ipython().run_line_magic('pwd', ''))
root_path = current_file_path
while not any(file.suffix == ".ipynb" for file in root_path.glob("*")):
    root_path = root_path.parent

root_path = str(root_path)
print(root_path)


self_play_data_dir=os.path.join(root_path,'self_play_data')
model_rec_dir=os.path.join(root_path,'model_record')

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

class TrainPipeline():
    
    def __init__(self, gui,init_model_dir=None):
        self.gui=gui
        self.board_width = 15
        self.board_height = 15
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 150  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        # if data_buffer appends after it reaches its maxlen then it will do
        # left pop and append the new one

        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 5
        self.game_batch_num = 500

        self.init_model_dir=init_model_dir
        self.policy_value_net = PolicyValueNet(self.board_width,
                                                self.board_height,
                                                model_file=os.path.join(self.init_model_dir,'model.h5'))
        
        self.game = Game(self.board,gui,self.policy_value_net.policy_value_fn)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      self.gui,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):

        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                         temp=self.temp)
            
            
            play_data = list(play_data)[:]
            file_name=f"PO_{self.n_playout}_{generate_random_string(15)}"
            print(f"game {i} finished")
            with open(os.path.join(self_play_data_dir,file_name), 'wb') as f:
                pickle.dump(play_data, f)
            # print(play_data)
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        return loss, entropy

    def run(self):
        """run the training pipeline"""
        try:
            
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)

                print("batch i:{}, episode_len:{}".format(
                    i+1, self.episode_len))
                
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    self_play_training_model_dir=os.path.join(self.init_model_dir,'self_play_training_model')
                    os.makedirs(self_play_training_model_dir, exist_ok = True)
                    self.policy_value_net.save_model(os.path.join(self_play_training_model_dir,f'SP_{i+1}_model.h5'))
        except KeyboardInterrupt:
            print('\n\rquit')
    def getboard(self):
        return self.board.states