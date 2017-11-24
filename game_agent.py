"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import numpy as np 

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def evaluate(game, player, index_list):
    """
    Returns the cross product of weights and evaluation values

    This function now only evaluate functions with its index in index_list

    Parameters
    ----------
    game : isolation.Board
    player : game-playing agent
    index_list : list
        indexes for functions to be evaluated

    Returns
    -------
    list with values returned from heuristic functions
    """
    try:
        eval_functions = [actionMobility, 
                            my_moves_op,
                            my_moves_2_op,
                            distance_from_center,
                            actionFocus]
        vec = []
        for idx in index_list:
            vec.append(eval_functions[idx](game, player))
            if player.time_left() < player.TIMER_THRESHOLD:
                raise SearchTimeout()
    except:
        print("EvaluateFunctionEror")
    return vec

def actionMobility(game, player):
    """
    Parameters
    ----------
    game : isolation_RL.Board
    player : player object
    max_acitons : int

    Reutrns
    -------
    number of possible moves/max_actions
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    #opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves)

def my_moves_op(game, player):
    """
    Parameters
    ----------
    game : isolation_RL.Baord
    player : player object
    
    Returns
    -------
    #my_moves-#op_moves
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)

def my_moves_2_op(game, player):
    """
    Parameters
    ----------
    game : isolation_RL.Baord
    player : player object
    
    Returns
    -------
    #my_moves-2*#op_moves
    """

    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    return float(len(game.get_legal_moves(player))-2*len(game.get_legal_moves(game.get_opponent(player))))

def distance_from_center(game, player):
    """
    Parameters
    ----------
    game : isolation_RL.Baord
    player : player object
    Returns 
    -------
    distance from center / max_dist
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")


    max_dist = np.sqrt(2*((game.height//2)**2))
    center = game.height//2
    current_position = game.get_player_location(player)
    distance = np.sqrt((abs(current_position[0]-center)**2)+(abs(current_position[1]-center))**2)
    return distance * 100.0/max_dist

def actionFocus(game, player, max_actions=8):

    """

    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    return 100.0-actionMobility(game, player)


def custom_score(game, player):
    """
    Evaluate current state on player's view, using the evaluate function to obtain 
    state features and a neural network forward pass to get the associated value

    Parameters
    ----------
    game : isolation_RL.Board
    player : player that the state should be evaluated for
    model : isolation_nn.Network

    Returns
    -------
    value : float

    Notes:
    - If weights are all set to 1.0 or 0.0, than this code is inefficient. It is 
    only worth if positive weighst are different than 1.0
    """

    try:
        index_list = []
        weights = [1.0, 0.25, 0.25, 0.0, 0.5]
        for i in range(len(weights)):
            if weights[i] != 0.0:
                index_list.append(i)
        # at the end, weights and index_list

        eval_vec = evaluate(game, player, index_list)
        value = 0
        for i in range(len(index_list)):
            value += weights[index_list[i]]*eval_vec[i]

    except:
        print("Custom_scoreFuncitonError")
    return value

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    try:
        index_list = []
        weights = [1.0, 0.0, 0.25, 0.0, 0.75]
        for i in range(len(weights)):
            if weights[i] != 0.0:
                index_list.append(i)
        # at the end, weights and index_list

        eval_vec = evaluate(game, player, index_list)
        value = 0
        for i in range(len(index_list)):
            value += weights[index_list[i]]*eval_vec[i]
    except:
        print("Custom_scoreFuncitonError")
    return value


def custom_score_3(game, player):
    """

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.

    
    """
    
    try:
        index_list = []
        weights = [0.0, 1.0, 0.0, 0.0, 0.0] # CURRENTLY ONLY USING MY_MOVES_OP
        for i in range(len(weights)):
            if weights[i] != 0.0:
                index_list.append(i)
        # at the end, weights and index_list

        eval_vec = evaluate(game, player, index_list)
        value = 0
        for i in range(len(index_list)):
            value += weights[index_list[i]]*eval_vec[i]
    except:
        print("Custom_scoreFuncitonError")
    return value


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        possible_actions = game.get_legal_moves()
        values_for_actions = np.zeros(len(possible_actions))
        for i in range(len(possible_actions)):
            values_for_actions[i] = self.min_value(game.forecast_move(possible_actions[i]), depth-1)
        try: 
            return possible_actions[np.argmax(values_for_actions)]
        except:

            return (-1, -1)

    def max_value(self, game, depth):
        """Max player in the minimax method. Look for the following move
        that will maximize the expected evaluation

        Parameters
        ----------
        game : Board object
            Board objest representing a state of the game. It is a forecast state
        following the last min action in the search tree

        depth : int
            remaining steps to reach maximum depth specified

        Returns
        -------
        val : int
            Utility value for current state

        """
        # timer check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # checking if limit depth or terminal test
        if depth == 0:
            return self.score(game, self)
        v = float("-inf")
        for action in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(action), depth-1))
        return v

    def min_value(self, game, depth):
        """Min player in the minimax method. Look for the following move that will
        minimize the expected evaluation

        Parameters
        ----------
        game : Board object
            Board objest representing a state of the game. It is a forecast state
        following the last min action in the search tree

        depth : int
            remaining steps to reach maximum depth specified

        Returns
        -------
        val : int
            Mimimum expected value associated with possible actions
    
        """
        # timer chack
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # checking if limit depth or terminal test
        if depth == 0:
            return self.score(game, self)
        v = float("inf")
        for action in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(action), depth-1))
        return v


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        best_move = (-1, -1)

        depth = 1
        while self.time_left() > self.TIMER_THRESHOLD:
            try:
                best_move = self.alphabeta(game, depth)
                depth += 1

            except SearchTimeout:
                break

        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        possible_actions = game.get_legal_moves()
        n_moves = len(possible_actions)
        values_for_actions = np.zeros(n_moves)
        for i in range(n_moves):
            values_for_actions[i] = self.min_alpha_beta(game.forecast_move(possible_actions[i]), depth-1, alpha, beta)
            alpha = max(values_for_actions[i], alpha)
        try: 
            return possible_actions[np.argmax(values_for_actions)]
        except:
            return (-1, -1)

    def min_alpha_beta(self, game, depth, alpha, beta):
        """Min player in the alpha beta search

        Parameter
        ---------
        game : isolation.Board
        depth : int
        alpha : int
            Since this is a min level, alpha represents the minimum value that can be found
            in this branch, because it is the lower bound of the parent of this node (it 
            will not choose a node with a value lower than alpha)
        beta :

        Returns
        -------
        v : int
            minimum evaluation found in its children
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self)

        v = float("inf")
        for action in game.get_legal_moves():
            v = min(v, self.max_alpha_beta(game.forecast_move(action), depth-1, alpha, beta))
            if v <= alpha: 
                return v
            beta = min(beta, v)
        return v

    def max_alpha_beta(self, game, depth, alpha, beta):
        """Max player in the alpha beta search

        Parameter
        ---------
        game : isolation.Board
        depth : int
        alpha : int
        beta : int
            Since this is a max level, beta represents the maximum value that can be found
            in this branch, because it is the upper bound of the parent of this node (it
            will not choose a node with a value higher than beta).

        Returns
        -------
        v : int
            maximum evaluation found in its children
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self)

        v = float("-inf")
        for action in game.get_legal_moves():
            v = max(v, self.min_alpha_beta(game.forecast_move(action), depth-1, alpha, beta)) 
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

