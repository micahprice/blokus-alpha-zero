import numpy as np
import logging
from game.travelblokus.config import BlokusConfig

blokusConfig = BlokusConfig()

class Game:
    def __init__(self):

        self.all_shapes = blokusConfig.all_shapes
        self.gameState = GameState(board=np.zeros(14*14, dtype=np.int), playerTurn=1, availableShapes=(self.all_shapes, self.all_shapes))
        self.currentPlayer = self.gameState.playerTurn # feel like this should only be kept track of in GameState, not Game
        # self.actionSpace = np.zeros(blokusConfig.n_possible_moves, dtype=np.int)  # AFAIK this is never used.
        self.pieces = {'1': 'X', '0': '-', '-1': 'O'} # pieces is not the best name for blokus specifically
        self.grid_shape = (14, 14)
        self.input_shape = (2, 14, 14) # playerpieces board and opponentpieces board I think. I believe you could include past states as more channels if desired
        self.name = 'travelblokus'
        self.state_size = len(self.gameState.binary)     # 2*14*14
        self.action_size = blokusConfig.n_possible_moves # 13798!
        self.all_shape_configurations = BlokusConfig.all_shape_configurations

    def reset(self):
        """Resets gameState and currentPlayer

        Returns:
            self.GameState
        """
        self.gameState = GameState(board=np.zeros(14*14, dtype=np.int), playerTurn=1, player1Shapes=self.all_shapes, player2Shapes=self.all_shapes)
        self.currentPlayer = self.gameState.playerTurn
        return self.gameState

    def step(self, action):
        """Take one game step and update gameState.

        One game step involves a player performing an action and updating currentPlayer.
        The returned state will reference the player that hasn't made a move yet.

        Args:
            action: idx to action in self.actionSpace?
        Returns:
            next_state: new gameState
            win_value: None unlessis_done is true (aka game is finished).
                If game is finished, -1 if currentPlayer lost, 0 if tied, 1 if currentPlayer won.
            is_done (bool): True if action caused game to end
            info: Hard Coded to None. Presumably does something somewhere.
        """
        next_state, win_value, done = self.gameState.takeAction(action)
        self.gameState = next_state
        self.currentPlayer = self.gameState.playerTurn
        info = None # Does this show up in a log somewhere?
        return ((next_state, win_value, done, info))

    def identities(self, state, actionValues):
        """Collects and returns state and AVs for all symmetrically identical boardstates.

        I believe this is only used for committing moves to memory (Memory.commit_to_stmemory), which converts results to a dict and stores them.
        Similar to horizontal or vertical flipping of images as data augmentation in image recognition problems.
        This is called by PlayMatches, which at the end of a match adds 'value' of -1, 0, or 1 to each stored dict
        based on if it represents a move by the winning, tying, or losing player of that match.

        For blokus, there are 8 (4 90 degree rotational symmetries * vertical/horizontal symmetry) identities returned.

        Args:
            state:
            actionValues
        Returns:
            identities: list of tuples of state, actionvalues  ( [(state1, AVs1), (state2, Avs2),...] )
                        that represent all symmetrically identical boardstates.
        """
        identities = [(state,actionValues)]
        cB = state.board # currentBoard
        cAV = actionValues # curentActionValues

        for n in range(3): # rotate board 90 degrees, three times
            cB = np.array([
                    cB[182], cB[168], cB[154], cB[140], cB[126], cB[112], cB[98],  cB[84], cB[70], cB[56], cB[42], cB[28], cB[14], cB[0],
                    cB[183], cB[169], cB[155], cB[141], cB[127], cB[113], cB[99],  cB[85], cB[71], cB[57], cB[43], cB[29], cB[15], cB[1],
                    cB[184], cB[170], cB[156], cB[142], cB[128], cB[114], cB[100], cB[86], cB[72], cB[58], cB[44], cB[30], cB[16], cB[2],
                    cB[185], cB[171], cB[157], cB[143], cB[129], cB[115], cB[101], cB[87], cB[73], cB[59], cB[45], cB[31], cB[17], cB[3],
                    cB[186], cB[172], cB[158], cB[144], cB[130], cB[116], cB[102], cB[88], cB[74], cB[60], cB[46], cB[32], cB[18], cB[4],
                    cB[187], cB[173], cB[159], cB[145], cB[131], cB[117], cB[103], cB[89], cB[75], cB[61], cB[47], cB[33], cB[19], cB[5],
                    cB[188], cB[174], cB[160], cB[146], cB[132], cB[118], cB[104], cB[90], cB[76], cB[62], cB[48], cB[34], cB[20], cB[6],
                    cB[189], cB[175], cB[161], cB[147], cB[133], cB[119], cB[105], cB[91], cB[77], cB[63], cB[49], cB[35], cB[21], cB[7],
                    cB[190], cB[176], cB[162], cB[148], cB[134], cB[120], cB[106], cB[92], cB[78], cB[64], cB[50], cB[36], cB[22], cB[8],
                    cB[191], cB[177], cB[163], cB[149], cB[135], cB[121], cB[107], cB[93], cB[79], cB[65], cB[51], cB[37], cB[23], cB[9],
                    cB[192], cB[178], cB[164], cB[150], cB[136], cB[122], cB[108], cB[94], cB[80], cB[66], cB[52], cB[38], cB[24], cB[10],
                    cB[193], cB[179], cB[165], cB[151], cB[137], cB[123], cB[109], cB[95], cB[81], cB[67], cB[53], cB[39], cB[25], cB[11],
                    cB[194], cB[180], cB[166], cB[152], cB[138], cB[124], cB[110], cB[96], cB[82], cB[68], cB[54], cB[40], cB[26], cB[12],
                    cB[195], cB[181], cB[167], cB[153], cB[139], cB[125], cB[111], cB[97], cB[83], cB[69], cB[55], cB[41], cB[27], cB[13]
                    ])

            cAV = np.array([
                    cAV[182], cAV[168], cAV[154], cAV[140], cAV[126], cAV[112], cAV[98],  cAV[84], cAV[70], cAV[56], cAV[42], cAV[28], cAV[14], cAV[0],
                    cAV[183], cAV[169], cAV[155], cAV[141], cAV[127], cAV[113], cAV[99],  cAV[85], cAV[71], cAV[57], cAV[43], cAV[29], cAV[15], cAV[1],
                    cAV[184], cAV[170], cAV[156], cAV[142], cAV[128], cAV[114], cAV[100], cAV[86], cAV[72], cAV[58], cAV[44], cAV[30], cAV[16], cAV[2],
                    cAV[185], cAV[171], cAV[157], cAV[143], cAV[129], cAV[115], cAV[101], cAV[87], cAV[73], cAV[59], cAV[45], cAV[31], cAV[17], cAV[3],
                    cAV[186], cAV[172], cAV[158], cAV[144], cAV[130], cAV[116], cAV[102], cAV[88], cAV[74], cAV[60], cAV[46], cAV[32], cAV[18], cAV[4],
                    cAV[187], cAV[173], cAV[159], cAV[145], cAV[131], cAV[117], cAV[103], cAV[89], cAV[75], cAV[61], cAV[47], cAV[33], cAV[19], cAV[5],
                    cAV[188], cAV[174], cAV[160], cAV[146], cAV[132], cAV[118], cAV[104], cAV[90], cAV[76], cAV[62], cAV[48], cAV[34], cAV[20], cAV[6],
                    cAV[189], cAV[175], cAV[161], cAV[147], cAV[133], cAV[119], cAV[105], cAV[91], cAV[77], cAV[63], cAV[49], cAV[35], cAV[21], cAV[7],
                    cAV[190], cAV[176], cAV[162], cAV[148], cAV[134], cAV[120], cAV[106], cAV[92], cAV[78], cAV[64], cAV[50], cAV[36], cAV[22], cAV[8],
                    cAV[191], cAV[177], cAV[163], cAV[149], cAV[135], cAV[121], cAV[107], cAV[93], cAV[79], cAV[65], cAV[51], cAV[37], cAV[23], cAV[9],
                    cAV[192], cAV[178], cAV[164], cAV[150], cAV[136], cAV[122], cAV[108], cAV[94], cAV[80], cAV[66], cAV[52], cAV[38], cAV[24], cAV[10],
                    cAV[193], cAV[179], cAV[165], cAV[151], cAV[137], cAV[123], cAV[109], cAV[95], cAV[81], cAV[67], cAV[53], cAV[39], cAV[25], cAV[11],
                    cAV[194], cAV[180], cAV[166], cAV[152], cAV[138], cAV[124], cAV[110], cAV[96], cAV[82], cAV[68], cAV[54], cAV[40], cAV[26], cAV[12],
                    cAV[195], cAV[181], cAV[167], cAV[153], cAV[139], cAV[125], cAV[111], cAV[97], cAV[83], cAV[69], cAV[55], cAV[41], cAV[27], cAV[13]
                    ])

            identities.append((GameState(cB, state.playerTurn), cAV))
        # now reflect on horizontal axis (could do vertical, but this concat is 4x faster)
        cB = np.concatenate((
                cB[182:], cB[168:182], cB[154:168], cB[140:154], cB[126:140],
                cB[112:126], cB[98:112],  cB[84:98], cB[70:84], cB[56:70],
                cB[42:56],  cB[28:42], cB[14:28], cB[:14]
                ))
        cAV = np.concatenate((
                cAV[182:], cAV[168:182], cAV[154:168], cAV[140:154], cAV[126:140],
                cAV[112:126], cAV[98:112],  cAV[84:98], cAV[70:84], cAV[56:70],
                cAV[42:56],  cAV[28:42], cAV[14:28], cAV[:14]
                ))

        identities.append((GameState(cB, state.playerTurn), cAV))

        for n in range(3): # rotate (reflected) board 90 degrees again, three times
            cB = np.array([
                    cB[182], cB[168], cB[154], cB[140], cB[126], cB[112], cB[98],  cB[84], cB[70], cB[56], cB[42], cB[28], cB[14], cB[0],
                    cB[183], cB[169], cB[155], cB[141], cB[127], cB[113], cB[99],  cB[85], cB[71], cB[57], cB[43], cB[29], cB[15], cB[1],
                    cB[184], cB[170], cB[156], cB[142], cB[128], cB[114], cB[100], cB[86], cB[72], cB[58], cB[44], cB[30], cB[16], cB[2],
                    cB[185], cB[171], cB[157], cB[143], cB[129], cB[115], cB[101], cB[87], cB[73], cB[59], cB[45], cB[31], cB[17], cB[3],
                    cB[186], cB[172], cB[158], cB[144], cB[130], cB[116], cB[102], cB[88], cB[74], cB[60], cB[46], cB[32], cB[18], cB[4],
                    cB[187], cB[173], cB[159], cB[145], cB[131], cB[117], cB[103], cB[89], cB[75], cB[61], cB[47], cB[33], cB[19], cB[5],
                    cB[188], cB[174], cB[160], cB[146], cB[132], cB[118], cB[104], cB[90], cB[76], cB[62], cB[48], cB[34], cB[20], cB[6],
                    cB[189], cB[175], cB[161], cB[147], cB[133], cB[119], cB[105], cB[91], cB[77], cB[63], cB[49], cB[35], cB[21], cB[7],
                    cB[190], cB[176], cB[162], cB[148], cB[134], cB[120], cB[106], cB[92], cB[78], cB[64], cB[50], cB[36], cB[22], cB[8],
                    cB[191], cB[177], cB[163], cB[149], cB[135], cB[121], cB[107], cB[93], cB[79], cB[65], cB[51], cB[37], cB[23], cB[9],
                    cB[192], cB[178], cB[164], cB[150], cB[136], cB[122], cB[108], cB[94], cB[80], cB[66], cB[52], cB[38], cB[24], cB[10],
                    cB[193], cB[179], cB[165], cB[151], cB[137], cB[123], cB[109], cB[95], cB[81], cB[67], cB[53], cB[39], cB[25], cB[11],
                    cB[194], cB[180], cB[166], cB[152], cB[138], cB[124], cB[110], cB[96], cB[82], cB[68], cB[54], cB[40], cB[26], cB[12],
                    cB[195], cB[181], cB[167], cB[153], cB[139], cB[125], cB[111], cB[97], cB[83], cB[69], cB[55], cB[41], cB[27], cB[13]
                    ])

            cAV = np.array([
                    cAV[182], cAV[168], cAV[154], cAV[140], cAV[126], cAV[112], cAV[98],  cAV[84], cAV[70], cAV[56], cAV[42], cAV[28], cAV[14], cAV[0],
                    cAV[183], cAV[169], cAV[155], cAV[141], cAV[127], cAV[113], cAV[99],  cAV[85], cAV[71], cAV[57], cAV[43], cAV[29], cAV[15], cAV[1],
                    cAV[184], cAV[170], cAV[156], cAV[142], cAV[128], cAV[114], cAV[100], cAV[86], cAV[72], cAV[58], cAV[44], cAV[30], cAV[16], cAV[2],
                    cAV[185], cAV[171], cAV[157], cAV[143], cAV[129], cAV[115], cAV[101], cAV[87], cAV[73], cAV[59], cAV[45], cAV[31], cAV[17], cAV[3],
                    cAV[186], cAV[172], cAV[158], cAV[144], cAV[130], cAV[116], cAV[102], cAV[88], cAV[74], cAV[60], cAV[46], cAV[32], cAV[18], cAV[4],
                    cAV[187], cAV[173], cAV[159], cAV[145], cAV[131], cAV[117], cAV[103], cAV[89], cAV[75], cAV[61], cAV[47], cAV[33], cAV[19], cAV[5],
                    cAV[188], cAV[174], cAV[160], cAV[146], cAV[132], cAV[118], cAV[104], cAV[90], cAV[76], cAV[62], cAV[48], cAV[34], cAV[20], cAV[6],
                    cAV[189], cAV[175], cAV[161], cAV[147], cAV[133], cAV[119], cAV[105], cAV[91], cAV[77], cAV[63], cAV[49], cAV[35], cAV[21], cAV[7],
                    cAV[190], cAV[176], cAV[162], cAV[148], cAV[134], cAV[120], cAV[106], cAV[92], cAV[78], cAV[64], cAV[50], cAV[36], cAV[22], cAV[8],
                    cAV[191], cAV[177], cAV[163], cAV[149], cAV[135], cAV[121], cAV[107], cAV[93], cAV[79], cAV[65], cAV[51], cAV[37], cAV[23], cAV[9],
                    cAV[192], cAV[178], cAV[164], cAV[150], cAV[136], cAV[122], cAV[108], cAV[94], cAV[80], cAV[66], cAV[52], cAV[38], cAV[24], cAV[10],
                    cAV[193], cAV[179], cAV[165], cAV[151], cAV[137], cAV[123], cAV[109], cAV[95], cAV[81], cAV[67], cAV[53], cAV[39], cAV[25], cAV[11],
                    cAV[194], cAV[180], cAV[166], cAV[152], cAV[138], cAV[124], cAV[110], cAV[96], cAV[82], cAV[68], cAV[54], cAV[40], cAV[26], cAV[12],
                    cAV[195], cAV[181], cAV[167], cAV[153], cAV[139], cAV[125], cAV[111], cAV[97], cAV[83], cAV[69], cAV[55], cAV[41], cAV[27], cAV[13]
                    ])

            identities.append((GameState(cB, state.playerTurn), cAV))
        # should be list of 8 tuples
        return identities


class GameState():
    """
    Blokus specific GameState

    Attributes:
        board: current board
        pieces: needed for render(). could move render to Game and remove pieces from GameState but eh.
        playerTurn:
        availableShapes: (currentPlayerShapesList, otherPlayerShapesList) must be flipped every turn
        binary:  1d binary array of currentplayer_position, other_position
        id:  returns string of characters 1d array of player1_position, player2_position
        allowedActions: list of indices within actionSpace
        isEndGame: bool signifying game ended after move on previous turn
        winValue: 1,0,-1 depending on if end of game (on previous turn) caused currentPlayer to win, draw, or lose
        score: (currentPlayerPoints, opponentPlayerPoints)

        winValue and score are only evaluated at end of game. To test this, they will be set to None unless isEndGame is true
    """
    def __init__(self, board, playerTurn, availableShapes):
        self.board = board
        self.pieces = {'1': 'X', '0': '-', '-1': 'O'} # feel like this should only be kept track off in Game, not GameState
        self.playerTurn = playerTurn
        self.availableShapes = availableShapes # must be flipped every turn
        self.binary = self._binary()
        self.id = self._convertStateToId()
        self.allowedActions = self._allowedActions()
        self.isEndGame = self._checkForEndGame() # isEndGame, score, winValue need to be in this order
        self.score = self._getScore()
        self.winValue = self._getWinValue()


    def _allowedActions(self):
        # TODO
        pass

    def _binary(self):
        """returns 1d binary array of currentplayer_position, other_position

              current    other
        eg. [0,1,0....,....0,0,1]
        """
        currentplayer_position = np.zeros(len(self.board), dtype=np.int)
        currentplayer_position[self.board == self.playerTurn] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -self.playerTurn] = 1

        position = np.append(currentplayer_position, other_position)

        return (position)

    def _convertStateToId(self):
        """returns string of characters 1d array of player1_position, player2_position

             P1      P2
        eg "010......001"
        """
        player1_position = np.zeros(len(self.board), dtype=np.int)
        player1_position[self.board == 1] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -1] = 1

        position = np.append(player1_position, other_position)

        id = ''.join(map(str, position))

        return id

    def _checkForEndGame(self):
        if len(self.allowedActions) == 0: # out of pieces or can't play
            future_opponent_turn = GameState(self.board, -self.playerTurn, self.availableShapes[::-1]) # will duplicate effort with takeAction.
            if future_opponent_turn.allowedActions == 0:
                return True
        return False # in py2 bool is slower than int, but not in py3

    def _getScore(self):
        """ Counts number of squares filled by player pieces"""
        # This is actually backwards. real scoring differs but is based on number of tiles
        # left, and you can get a -5 bonus if the A shape (1 block) is your last piece.
        # if only playing one game, the -5 is pointless, it just ensures a win. Could code that in but eh.
        if not self.isEndGame:
            return None
        else:
            currentPlayerScore = self.binary[:144].sum() # currentPlayer board coverage as 1
            otherPlayerScore   = self.binary[144:].sum() # otherPlayer board coverage as 1
        return currentPlayerScore, otherPlayerScore

    def _getWinValue(self):
        """ Highest number of squares filled by pieces wins"""
        if not self.isEndGame:
            return None
        else:
            if self.score[0] > self.score[1]:
                return 1 # currentPlayer has more squares on board
            elif self.score[0] < self.score[1]:
                return -1 # currentPlayer has less squares on board
            else:
                return 0  # tie

    def takeAction(self, action):
        # For Connect4: fills in a -1 or 1 based on the move and player turn.
        # For Blokus, this will be more complicated
        # Which may mean the chooseAction method in Agent needs to be edited
        # but I think this function alone returns the new board state based on the action
        # Also. What happens when there are no allowedActions?


        if len(self.allowedActions) == 0:
            # Game is NOT over!
            # because checking isEndGame goes a turn in the future, and it
            # didn't end last turn, opponent is guaranteed to have a move next turn.
            # make no moves, pass on to opponent.
            newState = GameState(self.board, -self.playerTurn, self.availableShapes[::-1])
        else:
            #TODO
            pass
            #bla bla

            #newBoard = np.array(self.board)
            #newBoard[action] = self.playerTurn

            #newState = GameState(newBoard, -self.playerTurn)

        done = newState.isEndGame
        winValue = newState.winValue # None if game isn't over. testing if this will break anything

        return newState, winValue, done

    def render(self, logger):
        for row in range(14):
            logger.info([self.pieces[str(x)] for x in self.board[(14 * row): (14 * row + 14)]])
        logger.info('--------------')
