from isolation import Board
from sample_players import RandomPlayer
from sample_players import GreedyPlayer
from game_agent import MinimaxPlayer
import timeit

# create an isolation board (by default 7x7)
player1 = MinimaxPlayer()
player2 = GreedyPlayer()
game = Board(player1, player2)
print("Player1: ", player1)
print("Player2: ", player2)


# place player 1 on the board at row 2, column 3, then place player 2 on
# the board at row 0, column 5; display the resulting board state.  Note
# that the .apply_move() method changes the calling object in-place.
game.apply_move((2, 3))
game.apply_move((1, 5))
print(game.to_string())

# players take turns moving on the board, so player1 should be next to move
assert(player1 == game.active_player)

# get a move from MinimaxPlayer
time_millis = lambda: 1000*timeit.default_timer()
start = time_millis()
time_left = lambda: 150 - (time_millis() - start)
minimax_move = game._active_player.get_move(game.copy(), time_left)
print(minimax_move)

# get a list of the legal moves available to the active player
legal_moves = game.get_legal_moves()


# get a successor of the current state by making a copy of the board and
# applying a move. Notice that this does NOT change the calling object
# (unlike .apply_move()).
print("current player: ", game._active_player)
new_game = game.forecast_move(minimax_move)
print("player after move ", minimax_move, ":", new_game._active_player)
assert(new_game.to_string() != game.to_string())
print("\nOld state:\n{}".format(game.to_string()))
print("\nNew state:\n{}".format(new_game.to_string()))

# play the remainder of the game automatically -- outcome can be "illegal
# move", "timeout", or "forfeit"
winner, history, outcome = game.play()
print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
print(game.to_string())
print("Move history:\n{!s}".format(history))

