import numpy as np
import sys
np.set_printoptions(threshold = 100000)

def create_start(): # tested and works
    empty = np.zeros((8, 8, 12)) # use 7- to make sure rows and columns line up
    empty[0, 7-4, 0] = 1 #king
    empty[0, 7-3, 1] = 1 #queen
    empty[0, 7-0, 2] = 1 #rook 
    empty[0, 7-7, 2] = 1 #rook2
    empty[0, 7-2, 3] = 1 #bishops
    empty[0, 7-5, 3] = 1
    empty[0, 7-1, 4] = 1 #horseys
    empty[0, 7-6, 4] = 1
    empty[1, :, 5] = 1 #pawns
    empty[7, 7-4, 6] = 1 #king
    empty[7, 7-3, 7] = 1 #queen
    empty[7, 7-0, 8] = 1 #rook 
    empty[7, 7-7, 8] = 1 #rook2
    empty[7, 7-2, 9] = 1 #bishops
    empty[7, 7-5, 9] = 1
    empty[7, 7-1, 10] = 1 #horseys
    empty[7, 7-6, 10] = 1
    empty[6, :, 11] = 1 #pawns
    return empty

def removepiece(current, change, channel):
    try: 
        current[int(change[1]) - 1, :, channel] = 0
    except ValueError:
        current[:, letters[change[1]], channel] = 0

samples = 50000
letters = {'a': 7, 'b': 6, 'c': 5, 'd': 4, 'e':3, "f":2, "g":1, "h": 0}
white = {'K': 0, 'Q': 1, 'R': 2, 'B': 3, 'N': 4, "pawn": 5, "player": "white"}
black = {'K': 6, 'Q': 7, 'R': 8, 'B': 9, 'N': 10, "pawn": 11, "player": "black"} 
sections = ['val', "test", 'train']
num_moves = 150
count = 0
while True: # keeps going until your data or storage runs out
    count += 1
    for ttv in sections: # iterates over train, test, and validation sets: you can remove the loop if you only have one section
        X1 = np.load(f'X_{ttv}.npy', allow_pickle = True)
        X = X1[samples * (count-1):samples * count, 1]
        X1 = 1
        out = np.zeros((X.shape[0], 8, 8, 12, num_moves)) #rows before columns
        destroy = []
        for i in range(len(X)): # iterates over games in the dataset 
            discard = False
            positions = np.zeros((1, 8, 8, 12, num_moves))
            game = X[i]
            moves = game.split()
            moves = moves[:num_moves]
            current = create_start()
            for j in range(len(moves)):

                en_passant = False
                skip_placement = False
                if j % 2 == 0:
                    color = white
                else:
                    color = black
                change = moves[j] # gets the moves we're on
                piece = change[0] # finds which piece is moved
                if change[0] == '1' or change[0] == '0': # detects end of game
                    break
                if piece == 'O': # Castling
                    king = color["K"]
                    rook = color["R"]
                    if len(change) <= 4: # kingside
                        if color['player'] == 'white':
                            last_rank = 0 
                        else:
                            last_rank = 7
                        current[last_rank, 7-6, king] = 1
                        current[last_rank, 7-4, king] = 0
                        current[last_rank, 7-5, rook] = 1
                        current[last_rank, 7-7, rook] = 0
                    if len(change) >= 5: # Queenside
                        if color['player'] == 'white':
                            last_rank = 0 
                        else:
                            last_rank = 7
                        current[last_rank, 7-2, king] = 1
                        current[last_rank, 7-4, king] = 0
                        current[last_rank, 7-3, rook] = 1
                        current[last_rank, 7-0, rook] = 0


                else:

                    try: # finds which channel to change
                        channel = color[piece] 
                    except KeyError:
                        piece = "pawn"
                        channel = color['pawn']
                    for place in range(1, 5): # finds which row the move is on
                        try: 
                            row = int(change[-place]) - 1
                        except ValueError:
                            pass
                        else:
                            break
                    for place in range(2, 5):
                        try:
                            column = letters[change[-place]]
                        except KeyError:
                            pass
                        else:
                            break
                    capture = change.find("x")
                    if capture >= 0: # if there's a capture set all pieces there to 0
                        if np.sum(current[row, column, :]) != 0:
                            current[row, column, :] = 0
                        else: # en passant
                            if piece != 'pawn':
                                discard = True 
                                break
                            if color['player'] == 'white':
                                current[row - 1, column, :] = 0
                            if color['player'] == 'black':
                                current[row + 1, column, :] = 0
                            current[row, column, channel] = 1
                            en_passant = True
                    if en_passant is False:
                         # position destroyed before
                    # in case there are two pieces that can move to one place
                        if capture == 2: # remove the old position of the piece. Eg. Raxb1
                            removepiece(current, change, channel)
                        elif capture == -1 and change[-1] != '#' and change[-1] != '+' and len(change) == 4 and change.find('=') == -1:
                            removepiece(current, change, channel)
                        elif capture == -1 and len(change) == 5 and change.find('=') == -1:
                            removepiece(current, change, channel)
                        else:
                            if piece == 'K':
                                for k in range(-1, 2):
                                    for m in range(-1, 2):
                                        try:
                                            current[row + k, column + m, channel] = 0
                                        except IndexError:
                                            pass
                            elif piece == 'Q':
                                up = True 
                                upleft = True 
                                upright = True 
                                down = True 
                                downleft = True 
                                downright = True 
                                left = True 
                                right = True 
                                for k in range(1, 8):
                                    try:
                                        if up is True:
                                            if np.sum(current[row + k, column, :]) != 0: # ends the search if it bumps a piece
                                                up = False
                                            current[row + k, column, channel] = 0 # removes rows
                                            
                                    except IndexError:
                                        pass
                                    try:
                                        if down is True:
                                            if np.sum(current[row - k, column, :]) != 0:
                                                down = False
                                            current[row - k, column, channel] = 0 # removes rows
                                            if row - k <= 0:
                                                down = False
                                    except IndexError:
                                        pass 
                                    try:
                                        if left is True:
                                            if np.sum(current[row, column + k, :]) != 0:
                                                left = False
                                            current[row, column + k, channel] = 0 # removes rows
                                            
                                    except IndexError:
                                        pass
                                    try:
                                        if right is True:
                                            if np.sum(current[row, column - k, :]) != 0:
                                                right = False
                                            current[row, column - k, channel] = 0 # removes rows
                                            if column - k <= 0:
                                                right = False
                                    except IndexError:
                                        pass
                                    try:
                                        if upleft is True:
                                            if np.sum(current[row + k, column + k, :]) != 0:
                                                upleft = False
                                            current[row + k, column + k, channel] = 0 # removes rows
                                            
                                    except IndexError:
                                        pass
                                    try:
                                        if upright is True:
                                            if np.sum(current[row + k, column - k , :]) != 0:
                                                upright = False
                                            current[row + k, column - k, channel] = 0 # removes rows
                                            if column - k <= 0:
                                                upright = False
                                    except IndexError:
                                        pass 
                                    try:
                                        if downleft is True:
                                            if np.sum(current[row - k, column + k, :]) != 0:
                                                downleft = False
                                            current[row - k, column + k, channel] = 0 # removes rows
                                            if row - k <= 0:
                                                downleft = False
                                    except IndexError:
                                        pass
                                    try:
                                        if downright is True:
                                            if np.sum(current[row - k, column - k, :]) != 0:
                                                downright = False
                                            current[row - k, column - k, channel] = 0 # removes rows
                                            if row - k <= 0 or column - k <= 0:
                                                downright = False
                                            
                                    except IndexError:
                                        pass
                            elif piece == "R":
                                up = True 
                                down = True 
                                left = True 
                                right = True 
                                for k in range(1, 8):
                                    try: # good
                                        if up is True:
                                            if np.sum(current[row + k, column, :]) != 0: # doesn't trigger condition because I checked the cleared square
                                                up = False 
                                            current[row + k, column, channel] = 0 # removes rows
                                    except IndexError:
                                        pass
                                    try: # going from behind to delete the rook
                                        if down is True:
                                            if np.sum(current[row - k, column, :]) != 0:
                                                down = False
                                            current[row - k, column, channel] = 0 
                                            if row - k <= 0:
                                                down = False
                                    except IndexError:
                                        pass
                                    try:
                                        if left is True:
                                            if np.sum(current[row, column + k, :]) != 0:
                                                left = False
                                            current[row, column + k, channel] = 0 # removes rows
                                    except IndexError:
                                        pass
                                    try:
                                        if right is True:
                                            if np.sum(current[row, column - k, :]) != 0:
                                                right = False 
                                            current[row, column - k, channel] = 0 # removes rows
                                            if column - k <= 0:
                                                right = False
                                    except IndexError:
                                        pass
                            elif piece == 'B':
                                up = True 
                                down = True 
                                left = True 
                                right = True 
                                for k in range(1, 8):
                                    try:
                                        if up is True:
                                            if np.sum(current[row + k, column + k, :]) != 0: 
                                                up = False 
                                            current[row + k, column + k, channel] = 0 # removes rows
                                    except IndexError:
                                        pass
                                    try:
                                        if down is True:
                                            if np.sum(current[row + k, column - k, :]) != 0:
                                                down = False
                                            current[row + k, column - k, channel] = 0 # removes rows
                                            if column - k <= 0:
                                                down = False
                                            
                                    except IndexError:
                                        pass
                                    try:
                                        if left is True:
                                            if np.sum(current[row - k, column + k, :]) != 0:
                                                left = False
                                            current[row- k, column + k, channel] = 0 # removes rows
                                            if row - k <= 0:
                                                left = False
                                    except IndexError:
                                        pass
                                    try:
                                        if right is True:
                                            if np.sum(current[row - k, column - k, :]) != 0:
                                                right = False
                                            current[row - k, column - k, channel] = 0 
                                            if row - k <= 0 or column - k <= 0:
                                                right = False
                                    except IndexError:
                                        pass
                            elif piece == 'N':
                                for k in range(1, 3): # between 1 and 2
                                    m = 3 - k
                                    try:
                                        current[row + k, column + m, channel] = 0
                                    except IndexError:
                                        pass
                                    try:
                                        current[row + k, column - m, channel] = 0
                                    except IndexError:
                                        pass
                                    try:
                                        current[row - k, column + m, channel] = 0
                                    except IndexError:
                                        pass
                                    try:
                                        current[row - k, column - m, channel] = 0
                                    except IndexError:
                                        pass
                            else: # if it's a pawn
                                promotion = change.find('=') # not finding promotions
                                original_column = letters[change[0]] # gets the column of the pawn
                                if color['player'] == 'white': # check if the pawns are white or black
                                    if current[row - 1, original_column, channel] == 1: # doesn't matter if there was a capture
                                        current[row - 1, original_column, channel] = 0
                                    else:
                                        current[row - 2, original_column, channel] = 0
                                else:
                                    if current[row + 1, original_column, channel] == 1:
                                        current[row + 1, original_column, channel] = 0
                                    else:
                                        current[row + 2, original_column, channel] = 0
                                if promotion >= 0:
                                    skip_placement = True
                                    new_channel = color[change[promotion + 1]] 
                                    if color['player'] == 'white':
                                        current[row - 1, original_column, channel] = 0
                                    else:
                                        current[row + 1, original_column, channel] = 0
                                    current[row, column, new_channel] = 1




                        if skip_placement is False:
                            current[row, column, channel] = 1
                        if j != 0:
                            try:
                                if capture < 0:
                                    assert np.sum(current) == np.sum(positions[0, :, :, :, j-1]), f"piece not removed properly"
                                else:
                                    assert np.sum(current) == np.sum(positions[0, :, :, :, j-1]) - 1, f"piece not removed properly"
                            except AssertionError:
                                discard = True
                                break

                positions[0, :, :, :, j] = current
            out[i, :, :, :, :] = positions
            if discard is True: # discards games that are strange (np.delete causes the shell to kill the program)
                destroy.append(i)
            if i % 10000 == 0:
                print(i/X.shape[0])
            if i == samples:
                break
        for board in reversed(destroy):
            out[board, :, :, :, :] = 0
        np.save(f"X_{ttv}_board{count}.npy", out)



#out: [train_examples, 8, 8, 12, num_moves]
