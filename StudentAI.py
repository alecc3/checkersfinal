
import copy
from math import log, sqrt
from random import randint, choice
from BoardClasses import Move, Board




class StudentAI():

    def __init__(self,col,row,p):
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col,row,p)
        self.board.initialize_game()
        self.color = ''
        self.opponent = {1:2,2:1}
        self.color = 2


    def get_move(self,move):

        if len(move) != 0:
            self.board.make_move(move, self.opponent[self.color])
        else:
            self.color = 1

        MCTS = MonteCarlo(self.board, self.color)
        move = MCTS.get_move()
        self.board.make_move(move, self.color)

        return move



class MonteCarlo:

    def __init__(self, myBoard, myPlayer, i=120, c=1):
        self.board = myBoard
        self.player = myPlayer
        self.opponent = {1: 2, 2: 1}
        self.wins = {}              # key:move, value:number of wins for the move
        self.plays = {}             # key:move, value:number of simulations for the move
        self.iterations = i         # number of total simulations
        self.c = c                  # exploration parameter


    def get_move(self):

        all_moves = []
        moves = self.board.get_all_possible_moves(self.player)
        for i in moves:
            for j in i:
                all_moves.append(j)

        if len(all_moves) == 1:
            return all_moves[0]

        # run a number of simulations, calculate the probability of wins/plays,
        # and return a move with the highest win ratio
        for _ in range(self.iterations):
            self.rollout()

        # create a dictionary of all the original/available moves from the initial state
        # the value of each move should be its win ratio (initialized to 0)
        valid_moves = {}
        for m in all_moves:
            valid_moves[m] = 0

        # for all the moves in valid_moves, calculate the probabilities,
        # and put the result in the valid_moves dictionary
        for move in valid_moves:
            for player, play in self.plays:
                if self.player == player and move[0] == play[0] and move[1] == play[1]:
                    valid_moves[move] = self.wins[(player, play)]/self.plays[(player, play)]

        # choose the move with the highest win rate from valid_moves
        best_move = max(valid_moves, key=valid_moves.get)
        return best_move


    def rollout(self):
        def will_promote(move, player):
            x, y = move[1][0], move[1][1]
            if player == 1: # black
                return x == self.board.row-1 # move to last row
            else:
                return x == 0


        def distance_from_center(move):
            # avoid expensive sqrt()
            x,y = move[1][0], move[1][1]
            center_r, center_c = self.board.row//2, self.board.col//2
            sq = lambda x:x*x
            # distance sqrt ((y2-y1)^2+(x2-x1)^2)
            euclidian = sq(y-center_c)+sq(x-center_r)
            return sqrt(euclidian)
        visited_boards = set()
        board_copy = copy.deepcopy(self.board)

        expand = True
        winner = 0
        while True:

            # pick a move for ourself from all possible moves
            # flatten the 2D list in to a 1D list of moves
            my_moves = []
            chosen_move = None
            all_moves = board_copy.get_all_possible_moves(self.player)
            for i in all_moves:
                for j in i:
                    my_moves.append(j)

            # check if all the possibles moves are in the "plays" dictionary
            all_in = True
            for m in my_moves:
                one_in = False
                for player, play in self.plays:
                    if self.player == player and m[0] == play[0] and m[1] == play[1]:
                        one_in = True
                # if a move is not in the "plays" dictionary, break out
                if not one_in:
                    all_in = False
                    break

            # skip move selection if there are only one possible moves
            if len(my_moves) == 1:
                chosen_move = my_moves[0]
                board_copy.make_move(chosen_move, self.player)

            # if the "plays" dictionary has all the moves in the list of possible moves
            # calculate the UCT of each move and choose the best move
            elif all_in:

                # create another dictionary to store each possible move's UCT value
                UCT_moves= {}
                for m in my_moves:
                    UCT_moves[m] = 0

                # first calculate the parents node's total number of simulations
                sp = 0
                for m in my_moves:
                    for player, play in self.plays:
                        if self.player == player and m[0] == play[0] and m[1] == play[1]:
                            sp += self.plays[(player, play)]

                for m in my_moves:
                    # then calculate the number of wins and simulations for each child node
                    wi = 0
                    si = 0

                    ## CALCULATE DISTANCE
                    dist = 0
                    for player, play in self.plays:
                        if self.player == player and m[0] == play[0] and m[1] == play[1]:
                            wi = self.wins[(player, play)]
                            si = self.plays[(player, play)]
                            dist = distance_from_center(m)

                    # after we got all the variables, calculate the UCT for each child node
                    for p in UCT_moves:
                        if m[0] == p[0] and m[1] == p[1]:
                            # calibrate this heuristic
                            dist_heuristic = -dist/si
                            uct = (wi/si) + (self.c * sqrt(log(sp)/si)) + dist_heuristic
                            UCT_moves[p] = uct

                # after we calculated the UCT value for each move, pick the move with the max UCT
                chosen_move = max(UCT_moves, key=UCT_moves.get)
                board_copy.make_move(chosen_move, self.player)
            

            # else choose a random move
            elif my_moves:
                chosen_move = choice(my_moves)
                board_copy.make_move(chosen_move, self.player)
            #board_copy.show_board()

            # check whether the move is already in the "plays" dictionary
            is_in = False
            for player, play in self.plays:
                if self.player == player and chosen_move[0] == play[0] and chosen_move[1] == play[1]:
                    is_in = True

            # if we have not expanded in this simulation and the move is not in the dictionary
            # add the move in the dictionary (create a node)
            if expand and not is_in:
                expand = False
                self.plays[(self.player, chosen_move)] = 1
                self.wins[(self.player, chosen_move)] = 1
            visited_boards.add((self.player, chosen_move))

            # check if winner after making a move
            winner = board_copy.is_win(self.player)
            if winner != 0:
                break

            ################## Now pick a move for the opponent ##################

            # pick a move for the opponent from all possible move
            opp_moves = []
            opponent_move = None
            all_moves = board_copy.get_all_possible_moves(self.opponent[self.player])
            for i in all_moves:
                for j in i:
                    opp_moves.append(j)

            # check if all the possibles moves are in the "plays" dictionary
            all_in = True
            for m in opp_moves:
                one_in = False
                for player, play in self.plays:
                    if self.opponent[self.player] == player and m[0] == play[0] and m[1] == play[1]:
                        one_in = True
                # if a move is not in the "plays" dictionary, break out
                if not one_in:
                    all_in = False
                    break

            # skip move selection if there are only one possible moves
            if len(opp_moves) == 1:
                opponent_move = opp_moves[0]
                board_copy.make_move(opponent_move, self.opponent[self.player])

            # if the "plays" dictionary has all the moves in the list of possible moves
            # calculate the UCT of each move and choose the best move
            elif all_in:

                # create another dictionary to store each possible move's UCT value
                UCT_moves = {}
                for m in opp_moves:
                    UCT_moves[m] = 0

                # first calculate the parents node's total number of simulations
                sp = 0
                for m in opp_moves:
                    for player, play in self.plays:
                        if self.opponent[self.player] == player and m[0] == play[0] and m[1] == play[1]:
                            sp += self.plays[(player, play)]

                for m in opp_moves:
                    # then calculate the number of wins and simulations for each child node
                    wi = 0
                    si = 0
                    di = 0
                    for player, play in self.plays:
                        if self.opponent[self.player] == player and m[0] == play[0] and m[1] == play[1]:
                            wi = self.wins[(player, play)]
                            si = self.plays[(player, play)]
                            di = distance_from_center(m)

                    # after we got all the variables, calculate the UCT for each child node
                    for p in UCT_moves:
                        if m[0] == p[0] and m[1] == p[1]:
                            dist_heuristic = -di/si
                            uct = (wi/si) + (self.c * sqrt(log(sp)/si)) + dist_heuristic
                            UCT_moves[p] = uct

                # after we calculated the UCT value for each move, pick the move with the max UCT
                opponent_move = max(UCT_moves, key=UCT_moves.get)
                board_copy.make_move(opponent_move, self.opponent[self.player])


            # else choose a random move
            elif opp_moves:
                opponent_move = choice(opp_moves)
                board_copy.make_move(opponent_move, self.opponent[self.player])
            #board_copy.show_board()

            # check whether the move is already in the "plays" dictionary
            is_in = False
            for player, play in self.plays:
                if self.opponent[self.player] == player and opponent_move[0] == play[0] and opponent_move[1] == play[1]:
                    is_in = True

            # if we have not expanded in this simulation and the move is not in the dictionary
            # add the move in the dictionary (create a node)
            if expand and not is_in:
                expand = False
                self.plays[(self.opponent[self.player], opponent_move)] = 1
                self.wins[(self.opponent[self.player], opponent_move)] = 1
            visited_boards.add((self.opponent[self.player], opponent_move))

            # check if winner after making a move
            winner = board_copy.is_win(self.opponent[self.player])
            if winner != 0:
                break

        # for all the moves in visited_boards
        # if the move is in the dictionary, increase the number of simulations
        # if the move led to a win, increase the number of wins
        for player1, play1 in visited_boards:
            for player2, play2 in self.plays:
                if player1 == player2 and play1[0] == play2[0] and play1[1] == play2[1]:
                    self.plays[player2, play2] += 1
                    if player1 == winner:
                        self.wins[player2, play2] += 1
                    elif player1 == self.player and winner == -1:
                        self.wins[player2, play2] += 1
