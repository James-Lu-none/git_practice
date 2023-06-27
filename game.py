
from __future__ import print_function
import numpy as np
import time
import pickle


class Board(object):
    """board for the game"""
    def __init__(self, **kwargs):
        self.width = 15
        self.height = 15
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = 5
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):

        square_state = np.zeros((5, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[2][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[3][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[1]=1-square_state[2]-square_state[3]
            square_state[4][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[0][:, :] = 1.0  # indicate the colour to play
        return square_state

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move
    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1
    
    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player
    def get_board(self):
        num_board=np.full((225),-1)
        for idx,i in enumerate(self.states):
            num_board[i]=idx
        return num_board
    def get_square_board(self):
        return np.reshape(self.get_board(),(15,15))

import random
first_moves=[96, 97, 98, 111, 112, 113, 126, 127, 128]
class Game(object):
    def __init__(self, board,gui,policy,**kwargs):
        self.board = board
        self.gui=gui
        self._policy=policy

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        self.board.init_board(start_player)
        if(start_player==1):
            self.board.do_move(random.choice(first_moves))
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        
        if is_shown:
            self.gui.show_board.emit(list(self.board.get_board()))
        while True:
            action_probs,_=self._policy(self.board)
            probs=np.zeros(225)
            for act,prob in action_probs:
                probs[act]=prob
            self.gui.show_probs.emit(list(probs))
            
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            print("the move is ",move)
            self.board.do_move(move)
            if is_shown:
                self.gui.show_board.emit(list(self.board.get_board()))
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner
            
        
    def start_self_play(self, player, is_shown=0, temp=1e-3):

        print("start self playing...")
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []

        
        # action_probs,_=self._policy(self.board)
        # probs=np.zeros(225)
        # for act,prob in action_probs:
        #     probs[act]=prob
        # # print(probs)
        # self.gui.show_probs.emit(list(probs))
        
        self.board.do_move(random.choice(first_moves))
        self.gui.show_board.emit(list(self.board.get_board()))
        while True:
            action_probs,_=self._policy(self.board)
            probs=np.zeros(225)
            for act,prob in action_probs:
                probs[act]=prob
            self.gui.show_probs.emit(list(probs))
            # print(probs)
            
            
            print("playing step {}...".format(len(states)+1),end='')
            startT = time.process_time()

            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            endT=time.process_time()
            # store the data
            print("done in {} s".format(endT-startT))
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            self.gui.show_board.emit(list(self.board.get_board()))

            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)