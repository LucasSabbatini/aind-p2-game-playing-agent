import numpy as np 
from isolation_RL import Baord
from game_agent_RL import *


def evaluate(game, player, max_actions=8):
    """
    Returns the cross product of weights and evaluation values
    """
    eval_vec = np.zeros((.num_functions))
    eval_vec[0] = actionMobility(game, player, max_actions)
    eval_vec[1] = my_moves_op(game, player)
    eval_vec[2] = my_moves_2_op(game, player)
    eval_evc[3] = distance_from_center(game, player)
    eval_vec[4] = actionFocus(game, player, max_actions)
    return eval_vec

def actionMobility(game, player, max_actions=8):
    """
    Returns number of possible moves
    """
    return (len(game.get_legal_moves(player))*100.0/float(max_actions))

def my_moves_op(game, player):
    """
    Returns (#my_moves-#op_moves)
    """
    return (len(game.get_legal_moves(player))-len(game.get_legal_moves(game.get_opponent(player))))

def my_moves_2_op(game, player):
    """
    Returns (#my_moves-2*#op_moves)/
    """
    return (len(game.get_legal_moves(player))-2*len(game.get_legal_moves(game.get_opponent(player))))

def distance_from_center(gamme, player):
    """
    Returns distance from center / max_dist
    """
    max_dist = np.sqrt(2*((game.height//2)**2))
    center = height//2
    current_position = game.get_player_location(player)
    distance = np.sqrt((abs(current_position[0]-center)**2)+(abs(current_possition[1]-center))**2)
    return distance * 100.0/max_dist

def action_focus(game, player, max_actions=8):
    """

    """
    return 100.0-actionMobility(game, player, max_actions)


def get_eval_vec(game, play):
    """
    Returns the cross product of weights and evaluation values
    """
    eval_vec = np.zeros((num_functions))
    eval_vec[0] = actionMobility(game, player, max_actions)
    eval_vec[1] = my_moves_op(game, player)
    eval_vec[2] = my_moves_2_op(game, player)
    eval_vec[3] = distance_from_center(game, player)
    eval_vec[4] = actionFocus(game, player, max_actions)
    return eval_vec 



def build_model(hidden_layer=2, n_nodes=10, num_functions):
    """
    This function build a simple feed_forward neural network, with two hidden_layers
    """
    inputs_ = tf.placeholder(tf.float32, [None ,num_functions], )


class QNetwork:
    def __init__(self, 
                 learning_rate=0.01, 
                 num_functions=4, 
                 hidden_size=10,
                 name="QNetwork"):
        with tf.variable_scope(name):

            # inputs will be the evaluation vactor, features extracted from the board
            self.inputs_ = tf.placeholder(tf.float32, [None, num_functions], name='inputs')
            # targets
            self.targets_ = tf.placeholder(tf.float32, [None, 1])

            # hidden layer
            self.fc = tf.layers.dense(self.inputs_, hidden_size, activation=tf.sigmoid)

            # output
            self.logit = tf.layers.dense(self.fc, 1)

            # loss
            self.loss = tf.reduce_mean(tf.softmax_cross_entropy_with_logits(logits=self.logits, labels=))

            # optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


state_features = []
total_time_limit = 150.0
time_threshold = 10.0
total_rewards = []
num_functions = 5

def train(player1, 
        player2, 
        total_rewards, 
        state_features, 
        learning_rate, 
        num_functions, 
        name_model, 
        num_games,
        hidden_size):

    model = QNetwork(learning_rate, num_functions, hidden_size, name_model)
    board = Board(player1, player2)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(num_games):

            j=0
            state_features_batch = []
            rewards = []
            values = []

            # we make one batch in one game
            while True:

                j+=1
                time_left = lambda : time_limit - (time_millis() - move_start)

                if board._active_player == player2
                    legal_player_moves = board.get_legal_moves()
                    game_copy = board.copy()    
                    move_start = time_millis()
                    curr_move = board._active_player.get_move(game_copy, time_left)         
                    move_end = time_left()

                    if curr_move is None:
                        curr_move = Board.NOT_MOVED

                    if curr_move not in legal_player_moves or move_end < 0:
                        # if player1 loses, append positive reward to rewards list,
                        # [0.0*len(num_functions)]
                        reward.append(1.0)
                        state_features_batch.append(np.zeros(len(num_functions)))
                        values.append(0.0)
                        break

                    move_history.append(list(curr_move))
                    board.apply_move(curr_move)

                else:

                    legal_player_moves = board.get_legal_moves()
                    board_copy = board.copy()
                    move_start = time_millis()
                    curr_move, val, sf = board._active_player.get_move(board.copy)
                    move_end = time_left()

                    if curr_move is None:
                        curr_move = Board.NOT_MOVED

                    if curr_move not in legal_player_moves or move_end < 0:
                        rewards.append(-1.0)
                        state_features_batch.append(np.zeros(len(num_functions)))
                        values.append(val)
                        break

                    values.append(val)
                    state_features_batch.append(sf)
                    rewards.append(0.0)

                    move_history.append(list(curr_move))

                    board.apply_move(curr_move)






















