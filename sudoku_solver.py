import numpy as np


def solve_sudoku(board):
    """
    Solves a Sudoku puzzle and modifies the input board in-place.
    :param board: 2D list of integers representing the Sudoku board (0 indicates an empty cell)
    :return: Boolean indicating whether the Sudoku was solved successfully.
    """

    def is_valid(num, pos):
        # Check row
        for i in range(len(board[0])):
            if board[pos[0]][i] == num and pos[1] != i:
                return False

        # Check column
        for i in range(len(board)):
            if board[i][pos[1]] == num and pos[0] != i:
                return False

        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3

        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if board[i][j] == num and (i, j) != pos:
                    return False
        return True

    def find_empty():
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    return (i, j)  # row, col
        return None

    def solve():
        find = find_empty()
        if not find:
            return True
        else:
            row, col = find

        for i in range(1, 10):
            if is_valid(i, (row, col)):
                board[row][col] = i

                if solve():
                    return True

                board[row][col] = 0

        return False

    solve()
    return board


def optimize_recognized_digits(digits):
    print("Original digits:")
    print(digits)

    optimized_digits = []
    for row in digits:
        optimized_row = []
        for digit in row:
            # If digit is a double number, crop '1' (assuming '1' is an artifact)
            if digit > 9:
                digit_str = str(digit)
                digit_str = digit_str.replace('1', '')  # Remove '1' from the digit
                # After removing '1', if the remaining string is empty or not a valid digit, consider it as an empty cell
                digit = int(digit_str) if digit_str.isdigit() and 0 < int(digit_str) <= 9 else 0
            # Ensure digit is within valid range, otherwise consider it as an empty cell
            digit = digit if digit in range(1, 10) else 0
            optimized_row.append(digit)
        optimized_digits.append(optimized_row)

    # Convert optimized_digits to a NumPy array
    optimized_digits = np.array(optimized_digits)

    # For debugging: print the optimized digits
    print("Optimized digits:")
    print(optimized_digits)

    return optimized_digits
