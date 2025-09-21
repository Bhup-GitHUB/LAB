import heapq
import copy

class PuzzleState:
    def __init__(self, board, parent=None, move=None, cost=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.cost = cost
        self.blank_pos = self.find_blank()
    
    def find_blank(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    return (i, j)
        return None
    
    def get_heuristic(self, goal):
        h = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != 0:
                    goal_i, goal_j = self.find_position(goal, self.board[i][j])
                    h += abs(i - goal_i) + abs(j - goal_j)
        return h
    
    def find_position(self, board, value):
        for i in range(3):
            for j in range(3):
                if board[i][j] == value:
                    return (i, j)
        return None
    
    def get_priority(self, goal):
        return self.cost + self.get_heuristic(goal)
    
    def get_successors(self):
        successors = []
        i, j = self.blank_pos
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        direction_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        for (di, dj), direction in zip(directions, direction_names):
            new_i, new_j = i + di, j + dj
            
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_board = copy.deepcopy(self.board)
                new_board[i][j] = new_board[new_i][new_j]
                new_board[new_i][new_j] = 0
                
                new_state = PuzzleState(new_board, self, direction, self.cost + 1)
                successors.append(new_state)
        
        return successors
    
    def is_goal(self, goal_board):
        return self.board == goal_board
    
    def __lt__(self, other):
        return self.cost < other.cost
    
    def __eq__(self, other):
        return self.board == other.board
    
    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.board))

def print_board(board):
    print("+---+---+---+")
    for row in board:
        print("|", end="")
        for cell in row:
            if cell == 0:
                print("   |", end="")
            else:
                print(f" {cell} |", end="")
        print()
        print("+---+---+---+")

def print_solution(state):
    path = []
    while state:
        path.append(state)
        state = state.parent
    
    path.reverse()
    
    print("Solution Path:")
    print("=" * 30)
    
    for i, state in enumerate(path):
        if i == 0:
            print("Initial State:")
        elif i == len(path) - 1:
            print("Goal State:")
        else:
            print(f"Step {i} (Move: {state.move}):")
        
        print_board(state.board)
        print()

def solve_8_puzzle(initial, goal):
    initial_state = PuzzleState(initial)
    goal_state = PuzzleState(goal)
    
    if initial_state.board == goal_state.board:
        return initial_state
    
    frontier = []
    heapq.heappush(frontier, (initial_state.get_priority(goal), initial_state))
    explored = set()
    
    step = 1
    
    while frontier:
        current_priority, current_state = heapq.heappop(frontier)
        
        if current_state in explored:
            continue
        
        explored.add(current_state)
        
        print(f"Step {step}: Exploring state with priority {current_priority}")
        print(f"Cost: {current_state.cost}, Heuristic: {current_state.get_heuristic(goal)}")
        print_board(current_state.board)
        print()
        
        if current_state.is_goal(goal):
            return current_state
        
        successors = current_state.get_successors()
        
        for successor in successors:
            if successor not in explored:
                priority = successor.get_priority(goal)
                heapq.heappush(frontier, (priority, successor))
                print(f"  Added successor with priority {priority} (Move: {successor.move})")
        
        print(f"Frontier size: {len(frontier)}")
        print(f"Explored states: {len(explored)}")
        print("-" * 40)
        
        step += 1
        
        if step > 50:
            print("Maximum steps reached")
            break
    
    return None

def solve_bfs(initial, goal):
    initial_state = PuzzleState(initial)
    goal_state = PuzzleState(goal)
    
    if initial_state.board == goal_state.board:
        return initial_state
    
    from collections import deque
    frontier = deque([initial_state])
    explored = set()
    
    while frontier:
        current_state = frontier.popleft()
        
        if current_state in explored:
            continue
        
        explored.add(current_state)
        
        if current_state.is_goal(goal):
            return current_state
        
        successors = current_state.get_successors()
        
        for successor in successors:
            if successor not in explored:
                frontier.append(successor)
    
    return None

def main():
    print("8-Puzzle Problem Solver")
    print("=" * 40)
    
    initial_state = [
        [1, 2, 3],
        [8, 0, 4],
        [7, 6, 5]
    ]
    
    goal_state = [
        [2, 8, 1],
        [0, 4, 3],
        [7, 6, 5]
    ]
    
    print("Initial State:")
    print_board(initial_state)
    print()
    
    print("Goal State:")
    print_board(goal_state)
    print()
    
    print("Solving using A* Search (A-star with Manhattan Distance Heuristic):")
    print("=" * 60)
    
    solution = solve_8_puzzle(initial_state, goal_state)
    
    if solution:
        print("\nSolution Found!")
        print(f"Total moves: {solution.cost}")
        print_solution(solution)
        
        print("Algorithm Analysis:")
        print("- Search Algorithm: A* with Manhattan Distance Heuristic")
        print("- Heuristic Function: Sum of Manhattan distances of misplaced tiles")
        print("- Guarantees optimal solution")
        print(f"- Total cost: {solution.cost} moves")
        
    else:
        print("No solution found!")
    
    print("\n" + "=" * 40)
    print("Alternative: BFS Solution")
    print("=" * 40)
    
    bfs_solution = solve_bfs(initial_state, goal_state)
    
    if bfs_solution:
        print("\nBFS Solution Found!")
        print(f"Total moves: {bfs_solution.cost}")
        print_solution(bfs_solution)
    else:
        print("BFS: No solution found!")

if __name__ == "__main__":
    main()
