def winning_move(char[:,:] board, char col):
    cdef char height, width, row, r, player, c1, c2, d1, d2 
    height = board.shape[0]
    width = board.shape[1]

    row = height - 1
    while row >= 0 and board[row, col] != -1:
        row -= 1
    row += 1
    player = board[row, col]
    if height - row >= 4:
        r = row - 1
        while r > height and board[r, col] == player:
            r -= 1
        if r - row >= 4:
            return True

    c1 = col - 1
    while c1 >= 0 and board[row, c1] == player:
        c1 -= 1
    c2 = col + 1
    while c2 < width and board[row, c2] == player:
        c2 += 1

    if c2 - c1 > 4:
        return True

    d1 = 1
    while row - d1 >= 0 and col - d1 >= 0 and board[row - d1, col - d1] == player:
        d1 += 1

    d2 = 1
    while (
        row + d2 < height and col + d2 < width and board[row + d2, col + d2] == player
    ):
        d2 += 1

    if d2 + d1 > 4:
        return True

    d1 = 1
    while row - d1 >= 0 and col + d1 < width and board[row - d1, col + d1] == player:
        d1 += 1

    d2 = 1
    while row + d2 < height and col - d2 >= 0 and board[row + d2, col - d2] == player:
        d2 += 1

    if d2 + d1 > 4:
        return True

    return False

