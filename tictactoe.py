""" Tic-Tac-Toe in Python

SAMPLE RUN OF TIC-TAC-TOE
Here’s what the user sees when they run the Tic-Tac-Toe program. The text the player enters is in bold.

Welcome to Tic-Tac-Toe!
Do you want to be X or O?
X
The computer will go first.
O| |
-+-+-
 | |
 -+-+-
  | |
  What is your next move? (1-9)
  3
  O| |
  -+-+-
   | |
   -+-+-
   O| |X

   What is your next move? (1-9)
   4
   O| |O
   -+-+-
   X| |
   -+-+-
   O| |X
   What is your next move? (1-9)
   5
   O|O|O
   -+-+-
   X|X|
   -+-+-
   O| |X
   The computer has beaten you! You lose.
   Do you want to play again? (yes or no)
   no

"""

import random
from typing import List


def map_of_the_board():
    print('7' + '|' + '8' + '|' + '9')
    print('-+-+-')
    print('4' + '|' + '5' + '|' + '6')
    print('-+-+-')
    print('1' + '|' + '2' + '|' + '3')


def drawBoard(board: List):
    # This function prints out the board that it was passed.

    # "board" is a list of 10 strings representing the board (ignore index 0).

    print(board[7] + '|' + board[8] + '|' + board[9])
    print('-+-+-')
    print(board[4] + '|' + board[5] + '|' + board[6])
    print('-+-+-')
    print(board[1] + '|' + board[2] + '|' + board[3])


def inputPlayerLetter() -> List:
    # Lets the player type which letter they want to be.
    # Return a list with player's leyyer as the first item and computer's letter as the second.

    letter = ''
    while not (letter == 'X' or letter == 'O'):
        print('Do you want to be X or O?')
        letter = input().upper()

    # The first element in the list is the player's letter; the second is the computer's letter.

    if letter == 'X':
        return ['X', 'O']
    else:
        return ['O', 'X']


def whoGoesFirst() -> str:
    # Randomly chose which player goes first.
    if random.randint(0,1) == 0:
        return 'computer'
    else:
        return 'player'


def makeMove(board: List, letter: str, move: int):
    # Make a move for either the computer or the user
    board[move] = letter


def isWinner(bo, le):
    # Given a board and a player's letter, this function return True that player has won.

    # We use "bo" instead of "board" and "le" instead of "letter" so that we don't have to type as much.

    return (
            (bo[4] == le and bo[5] == le and bo[6] == le) or # Across the middle
            (bo[1] == le and bo[2] == le and bo[3] == le) or # Across the bottom
            (bo[7] == le and bo[4] == le and bo[1] == le) or # Down the left side
            (bo[8] == le and bo[5] == le and bo[2] == le) or # Down the middle
            (bo[9] == le and bo[6] == le and bo[3] == le) or # Down the right side
            (bo[7] == le and bo[5] == le and bo[3] == le) or # Diagonal
            (bo[9] == le and bo[5] == le and bo[1] == le) # Diagnoal
            )


def getBoardCopy(board) -> List:
    # Make a copy of the board list and return it
    boardCopy = []
    for i in board:
        boardCopy.append(i)
    return boardCopy

def isSpaceFree(board, move):
    # Return True if passed move is free on the passed board.
    return board[move] == ' '

def getPlayerMove(board) -> int:
    # Let the player enter their move
    move = ' '
    while move not in '1 2 3 4 5 6 7 8 9'.split() or not isSpaceFree(board, int(move)):
        print('What is your next move? (1-9)')
        move = input()
    return int(move)

def chooseRandomMoveFromList(board, moveList):
    # Returns valid move from the passed list on the passed board.
    # Returns none if there is no valid move.

    possibleMoves = []
    for i in moveList:
        if isSpaceFree(board, i):
            possibleMoves.append(i)

    if len(possibleMoves) != 0:
        return random.choice(possibleMoves)
    else:
        return None

def getComputerMove(board: List, computerLetter: str):
    # Given a board the computer's letter, determine where to move and return that move.

    if computerLetter == 'X':
        playerLetter = 'O'
    else:
        playerLetter = 'X'

    # Here is the algorithm for out Tic-Tac-Toe "AI" (or at least an intelligent player):
    # First, check if we can win in the next move.

    for i in range(1,10):
        boardCopy = getBoardCopy(board)
        if isSpaceFree(boardCopy, i):
            makeMove(boardCopy, computerLetter, i)
            if isWinner(boardCopy, computerLetter):
                return i

    # Chck if the player can win on their next move and block them.
    for i in range(1,10):
        boardCopy = getBoardCopy(board)
        if isSpaceFree(boardCopy, i):
            makeMove(boardCopy, playerLetter, i)
            if isWinner(boardCopy, playerLetter):
                return i


    # Try to take one of the corners, if they are free.
    move = chooseRandomMoveFromList(board, [1,3,7,9])
    if move != None:
        return move

    # Try to take the center, if it is free.
    if isSpaceFree(board, 5):
        return 5

    # Move on one of the sides.
    return chooseRandomMoveFromList(board, [2,4,6,8])


def isBoardFull(board:List):
    # Return true if every space on the board has been taken.  Otherwise, return False.

    for i in range(1,10):
        if isSpaceFree(board, i):
            return False
    return True



print('Welcome to Tic-Tac-Toe!')


map_of_the_board()


while True:
    # Reset the board.
    theBoard = [' '] * 10

    playerLetter, computerLetter = inputPlayerLetter()

    turn = whoGoesFirst()
    print('The '+turn+' will go first.')
    gameIsPlaying = True

    while gameIsPlaying:
        if turn == "player":
            #Player's Turn
            drawBoard(theBoard)
            move = getPlayerMove(theBoard)
            makeMove(theBoard,playerLetter, move)

            if isWinner(theBoard, playerLetter):
                drawBoard(theBoard)
                print('Yey- Good for you - You\'ve won the game!')
                gameIsPlaying = False

            elif isBoardFull(theBoard):
                drawBoard(theBoard)
                print('The game is a tie!')
                break
            else:
                turn = 'computer'
        else:
            #Computer's Turn
            move = getComputerMove(theBoard, computerLetter)
            makeMove(theBoard, computerLetter, move)

            if isWinner(theBoard, computerLetter):
                drawBoard(theBoard)
                print('The computer has beaten you. You lose :(')
                gameIsPlaying = False
            elif isBoardFull(theBoard):
                drawBoard(theBoard)
                print('The game is a tie!')
                break
            else:
                turn = 'player'







