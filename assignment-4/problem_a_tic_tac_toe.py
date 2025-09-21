import math
import time

class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.nodes_evaluated = 0
        
    def print_board(self):
        print("  0   1   2")
        for i in range(3):
            print(f"{i} {self.board[i][0]} | {self.board[i][1]} | {self.board[i][2]}")
            if i < 2:
                print("  ---------")
    
    def is_winner(self, player):
        for i in range(3):
            if all(self.board[i][j] == player for j in range(3)):
                return True
            if all(self.board[j][i] == player for j in range(3)):
                return True
        
        if all(self.board[i][i] == player for i in range(3)):
            return True
        if all(self.board[i][2-i] == player for i in range(3)):
            return True
        
        return False
    
    def is_board_full(self):
        return all(self.board[i][j] != ' ' for i in range(3) for j in range(3))
    
    def is_game_over(self):
        return self.is_winner('X') or self.is_winner('O') or self.is_board_full()
    
    def make_move(self, row, col, player):
        if self.board[row][col] == ' ':
            self.board[row][col] = player
            return True
        return False
    
    def undo_move(self, row, col):
        self.board[row][col] = ' '
    
    def get_empty_cells(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']
    
    def evaluate_board(self):
        if self.is_winner('X'):
            return 1
        elif self.is_winner('O'):
            return -1
        else:
            return 0
    
    def minimax(self, depth, is_maximizing, alpha=-math.inf, beta=math.inf, use_alphabeta=False):
        self.nodes_evaluated += 1
        
        if self.is_game_over():
            return self.evaluate_board()
        
        if is_maximizing:
            max_eval = -math.inf
            for row, col in self.get_empty_cells():
                self.make_move(row, col, 'X')
                eval_score = self.minimax(depth + 1, False, alpha, beta, use_alphabeta)
                self.undo_move(row, col)
                max_eval = max(max_eval, eval_score)
                
                if use_alphabeta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
            
            return max_eval
        else:
            min_eval = math.inf
            for row, col in self.get_empty_cells():
                self.make_move(row, col, 'O')
                eval_score = self.minimax(depth + 1, True, alpha, beta, use_alphabeta)
                self.undo_move(row, col)
                min_eval = min(min_eval, eval_score)
                
                if use_alphabeta:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
            
            return min_eval
    
    def get_best_move(self, use_alphabeta=False):
        best_score = -math.inf
        best_move = None
        self.nodes_evaluated = 0
        
        for row, col in self.get_empty_cells():
            self.make_move(row, col, 'X')
            score = self.minimax(0, False, use_alphabeta=use_alphabeta)
            self.undo_move(row, col)
            
            if score > best_score:
                best_score = score
                best_move = (row, col)
        
        return best_move
    
    def reset_game(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.nodes_evaluated = 0

def compare_algorithms():
    print("Minimax Algorithm Comparison")
    print("=" * 50)
    
    game = TicTacToe()
    
    test_positions = [
        [['X', 'O', ' '], [' ', 'X', 'O'], [' ', ' ', ' ']],
        [['X', ' ', 'O'], [' ', ' ', ' '], ['O', ' ', 'X']],
        [[' ', 'X', ' '], ['O', ' ', 'X'], [' ', ' ', 'O']],
        [['X', 'O', 'X'], [' ', ' ', ' '], ['O', ' ', ' ']],
        [[' ', ' ', ' '], ['X', 'O', ' '], [' ', 'X', 'O']]
    ]
    
    for i, test_board in enumerate(test_positions, 1):
        print(f"\nTest Position {i}:")
        game.board = [row[:] for row in test_board]
        game.print_board()
        
        print("\nMinimax without Alpha-Beta:")
        start_time = time.time()
        game.nodes_evaluated = 0
        best_move_no_ab = game.get_best_move(use_alphabeta=False)
        time_no_ab = time.time() - start_time
        nodes_no_ab = game.nodes_evaluated
        
        print(f"Best move: {best_move_no_ab}")
        print(f"Nodes evaluated: {nodes_no_ab}")
        print(f"Time taken: {time_no_ab:.6f} seconds")
        
        print("\nMinimax with Alpha-Beta:")
        start_time = time.time()
        game.nodes_evaluated = 0
        best_move_ab = game.get_best_move(use_alphabeta=True)
        time_ab = time.time() - start_time
        nodes_ab = game.nodes_evaluated
        
        print(f"Best move: {best_move_ab}")
        print(f"Nodes evaluated: {nodes_ab}")
        print(f"Time taken: {time_ab:.6f} seconds")
        
        print(f"\nPerformance Improvement:")
        print(f"Nodes reduction: {((nodes_no_ab - nodes_ab) / nodes_no_ab * 100):.1f}%")
        print(f"Time improvement: {((time_no_ab - time_ab) / time_no_ab * 100):.1f}%")

def play_game():
    print("Tic-Tac-Toe Game - AI vs Human")
    print("=" * 40)
    print("You are 'O', AI is 'X'")
    print("Enter row and column (0-2) to make your move")
    print("=" * 40)
    
    game = TicTacToe()
    
    while not game.is_game_over():
        game.print_board()
        
        if game.current_player == 'O':
            try:
                row = int(input(f"\nPlayer O, enter row (0-2): "))
                col = int(input("Player O, enter column (0-2): "))
                
                if 0 <= row <= 2 and 0 <= col <= 2:
                    if game.make_move(row, col, 'O'):
                        game.current_player = 'X'
                    else:
                        print("Cell already occupied!")
                else:
                    print("Invalid input! Enter values between 0-2")
            except ValueError:
                print("Invalid input! Enter numbers only")
        else:
            print("\nAI (X) is thinking...")
            best_move = game.get_best_move(use_alphabeta=True)
            if best_move:
                row, col = best_move
                game.make_move(row, col, 'X')
                print(f"AI plays at ({row}, {col})")
                game.current_player = 'O'
    
    game.print_board()
    
    if game.is_winner('X'):
        print("\nAI (X) wins!")
    elif game.is_winner('O'):
        print("\nPlayer (O) wins!")
    else:
        print("\nIt's a tie!")

def demonstrate_optimal_play():
    print("Demonstrating Optimal AI Play")
    print("=" * 40)
    
    game = TicTacToe()
    moves = 0
    
    while not game.is_game_over() and moves < 9:
        game.print_board()
        
        if game.current_player == 'X':
            print("\nAI (X) is thinking...")
            best_move = game.get_best_move(use_alphabeta=True)
            if best_move:
                row, col = best_move
                game.make_move(row, col, 'X')
                print(f"AI plays at ({row}, {col})")
                game.current_player = 'O'
        else:
            print("\nAI (O) is thinking...")
            game.current_player = 'X'
            best_move = game.get_best_move(use_alphabeta=True)
            if best_move:
                row, col = best_move
                game.make_move(row, col, 'O')
                print(f"AI plays at ({row}, {col})")
                game.current_player = 'X'
        
        moves += 1
    
    game.print_board()
    
    if game.is_winner('X'):
        print("\nAI (X) wins!")
    elif game.is_winner('O'):
        print("\nAI (O) wins!")
    else:
        print("\nIt's a tie!")

def main():
    print("Tic-Tac-Toe AI - Min-Max Algorithm with Alpha-Beta Pruning")
    print("=" * 70)
    
    while True:
        print("\nChoose an option:")
        print("1. Compare Minimax vs Alpha-Beta Pruning Performance")
        print("2. Play against AI")
        print("3. Watch AI vs AI (Optimal Play)")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            compare_algorithms()
        elif choice == '2':
            play_game()
        elif choice == '3':
            demonstrate_optimal_play()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please enter 1-4")

if __name__ == "__main__":
    main()
