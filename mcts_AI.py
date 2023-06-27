import random
import numpy as np
import copy
def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def move_filter(board, bounds):
    filter_move = np.zeros_like(board)

    size = board.shape[0]
    for i in range(size):
        for j in range(size):
            if board[i, j] == 1:
                for x in range(i - bounds, i + bounds + 1):
                    for y in range(j - bounds, j + bounds + 1):
                        if 0 <= x < size and 0 <= y < size:
                            filter_move[x, y] = 1

    return filter_move

class TreeNode(object):

    def __init__(self, parent, prior_p): 
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, policy_a_p):
        for action, prob in policy_a_p:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
        # for _ in range(len(policy_action)):
        #     action = policy_action[_]
        #     prob = policy_prob[_]
        #     if action not in self._children:
        #         self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        #選擇UCB最大的一個children
        return max(self._children.items(), #items輸出「索引、值」對，索引為元組action 值為TreeNode
                   key=lambda act_node: act_node[1].get_value(c_puct)) #計算目前的children的UCB(下一層的TreeNode的UCB) act_node[1]表示的是值TreeNode

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        
    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)
    def get_value(self, c_puct): 
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u
    def is_leaf(self):
        return self._children == {}
    def is_root(self):
        return self._parent is None
    
class MCTS(object):
    def __init__(self,gui, policy_value_fn, c_puct=5, n_playout=10000):
        self.gui=gui
        self._root = TreeNode(None, 1.0) #創建一個TreeNode類型的根節點(沒父節點)
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._policy=policy_value_fn
        self.timer=0
    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def get_move_probs(self, board, temp=1e-3):
        probs=np.zeros(225)
        # print("current board here\n",board)
        for n in range(self._n_playout):
            board_copy = copy.deepcopy(board)
            # print("Playout",n)
            self._playout(board_copy)
            if(n%1==0 and n!=0 and time.process_time()-self.timer>0.3):
                self.timer=time.process_time()
                probs=np.zeros(225)
                act_visits = [(act, node._n_visits)
                    for act, node in self._root._children.items()]
                acts, visits = zip(*act_visits)
                act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
                probs[list(acts)] = act_probs
                self.gui.show_probs.emit(list(probs))
            else:
                # print("UI not ready, time left:",time.process_time()-self.timer)
                pass

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)  
                    for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        probs[list(acts)] = act_probs
        # self.gui.show_probs.emit(list(probs))

        return acts, act_probs

    def _playout(self, board):
        # print(node._n_visits)
        # print("node_head_step_cnt:",step_cnt)
        node = self._root #複製一個根節點，提供之後的mcts
        while(1): #開始selection
            if node.is_leaf(): #是否到了葉節點(沒children)
                # print("leaf reached")
                break 
            action, node = node.select(self._c_puct) #select一個children，並更新node為選中的children
            board.do_move(action) #更新node的棋盤狀態
        # print("selection done")
        # print("selection board here\n",self.board)
        policy_a_p, leaf_value = self._policy(board) #到了葉節點，通過策略網路開始尋找下層要expand的5個點的位置和勝率和估值。
        policy_a_p = sorted(list(policy_a_p), key=lambda x: x[1],reverse=True)
        f_policy_a_p=policy_a_p[:15]
        # print(policy_a_p)
        # if node == self._root:
        #     for a,p in policy_a_p:
        #         print(a//15, a%15)
        #         print(p)
        #     input()
        # print(node._n_visits)
        if node._n_visits == 0 and node != self._root:
            leaf_value = self.roll_out(board) #將估值改為roll_out出來的值
        else: 
            node.expand(f_policy_a_p) #expand出5個策略網路算出來最好的點
        #backpropagation 更新自己和列祖列宗的node_value
    
        node.update_recursive(-leaf_value)
        
    def roll_out(self, org_board):
        board=copy.deepcopy(org_board)
        player = board.get_current_player()
        cnt=0
        root_board_step=len(board.states)

        # print()
        # print(board.get_square_board())
        while(len(board.availables)>0):
            end, winner = board.game_end()
            
            if end:
                break
            policy_a_p, _ = self._policy(board)
            
            played_move_board=board.get_square_board()
            filter=move_filter(np.where(played_move_board<0,0,1),2)
            # print(filtered_move)
            prob_acts=[]
            for act,prob in policy_a_p:
                if(filter[act//15][act%15]==1):
                    prob_acts.append((act,prob))
            probs = [prob for _, prob in prob_acts]
            
            move = np.random.choice([act for act, _ in prob_acts], p=probs/sum(probs))
            board.do_move(move)
            
            # time.sleep(0.001)
            # self.gui.show_probs.emit(list(probs))
            # print(prob_acts)
            self.gui.show_playout_board.emit([list(board.get_square_board()),root_board_step])
            cnt+=1

        # print("player:",player)
        # print("winner:",winner)
        # a=input()
        
        # print(f"playout done in {cnt} moves\nwinner: {winner}")
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1


import time
class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function, gui,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.gui=gui #send the worker's self instance to here and call set_state in MainWindow
        # self.mcts = MCTS(self.gui ,policy_value_function, c_puct, n_playout,)
        self._is_selfplay = is_selfplay
        self._policy=policy_value_function
        self.mcts = MCTS(policy_value_fn=policy_value_function,
                         gui=self.gui,
                         c_puct=c_puct, n_playout=n_playout)
    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.argmax(move_probs)
                
                print(move)
                # move = np.random.choice(
                #     acts,
                #     p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                # )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.argmax(move_probs)
                
                print(move)
                # reset the root node
                self.mcts.update_with_move(-1)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")
    def __str__(self):
        return "MCTS {}".format(self.player)
