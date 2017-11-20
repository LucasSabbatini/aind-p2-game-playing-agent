
import random
import numpy as np 

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

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
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return float(len(game.get_legal_moves(player)))

class IsolationPlayer:

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score_fn = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """
    Game agent using only minimax method.


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
        """
        This function will perform a depth-limited searh to find the best move.
        It'll act like the minimax-decision funciton previously implemented, so 
        it'll call a max_value and a min_value methods, which will be implemented
        within this class.

        	This method is the starting (root) node of a search tree, and what follows
        	is a min node.


        Assumptions:
            1. The minimax algorithm finds the path to the best game for max (it 
        searches the entire tree to find the answer).
            refutation: This code will not search the whole tree for the best move, 
        since it is a depth limited search. It'll keep opening branches till it reaches
        the defined depth, and than apply the evaluation function in that state and
        return the value found.
            
            2. Since this is a depth limited search, this will actually be a quiescent
        search, which means will iteratively go down tree, till we find a depth 
        where the eval
            refutation: Will not be a quiescent search. The depth is well defined, so
        we don't have to find a quiescence depth.


         Arguments:
         - game: Board object representing the current game state
         - depth: depth which our code should look for the answer

         Returns:
         - move: a tuple of the form (int, int) representing the position on the board
         which the MinimaxPlayer  should move
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

            print(type(possible_actions))
            print(possible_actions)  
            pass

    def max_value(self, game, depth):
        """Max player in the minimax method. Look for the following move
        that will maximize the expected evaluation

        Parameters
        ----------
        game : isolation.Board
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
        game : isolation.Board
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

    

