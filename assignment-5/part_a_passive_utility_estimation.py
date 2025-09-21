import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

class GridWorld:
    def __init__(self, rows=3, cols=4):
        self.rows = rows
        self.cols = cols
        self.terminal_states = {(0, 3): 1, (1, 3): -1}
        self.wall = (1, 1)
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_effects = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        self.gamma = 0.9
        self.stochasticity = 0.1
        
    def is_valid_state(self, state):
        row, col = state
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if state == self.wall:
            return False
        return True
    
    def get_next_state(self, state, action):
        if state in self.terminal_states:
            return state
        
        intended_effect = self.action_effects[action]
        
        if random.random() < (1 - 2 * self.stochasticity):
            next_state = (state[0] + intended_effect[0], state[1] + intended_effect[1])
        else:
            perpendicular_actions = []
            for a in self.actions:
                if a != action and self.action_effects[a] != intended_effect:
                    perpendicular_actions.append(a)
            
            if perpendicular_actions:
                chosen_action = random.choice(perpendicular_actions)
                perpendicular_effect = self.action_effects[chosen_action]
                next_state = (state[0] + perpendicular_effect[0], state[1] + perpendicular_effect[1])
            else:
                next_state = state
        
        if not self.is_valid_state(next_state):
            return state
        
        return next_state
    
    def get_reward(self, state):
        if state in self.terminal_states:
            return self.terminal_states[state]
        return -0.04
    
    def get_fixed_policy_action(self, state):
        if state in self.terminal_states:
            return None
        
        row, col = state
        
        if col < self.cols - 1:
            next_state = (row, col + 1)
            if self.is_valid_state(next_state):
                return 'RIGHT'
        
        if row < self.rows - 1:
            next_state = (row + 1, col)
            if self.is_valid_state(next_state):
                return 'DOWN'
        
        return 'RIGHT'
    
    def get_modified_policy_action(self, state):
        if state in self.terminal_states:
            return None
        
        row, col = state
        
        if row < self.rows - 1:
            next_state = (row + 1, col)
            if self.is_valid_state(next_state):
                return 'DOWN'
        
        if col < self.cols - 1:
            next_state = (row, col + 1)
            if self.is_valid_state(next_state):
                return 'RIGHT'
        
        return 'DOWN'

class PassiveUtilityEstimator:
    def __init__(self, grid_world):
        self.grid_world = grid_world
        self.state_returns = defaultdict(list)
        self.state_utilities = {}
    
    def generate_episode(self, policy_func, max_steps=100):
        episode = []
        state = (0, 0)
        total_return = 0
        discount = 1.0
        
        for step in range(max_steps):
            if state in self.grid_world.terminal_states:
                break
            
            action = policy_func(state)
            if action is None:
                break
            
            next_state = self.grid_world.get_next_state(state, action)
            reward = self.grid_world.get_reward(next_state)
            
            episode.append((state, action, reward, next_state))
            total_return += discount * reward
            discount *= self.grid_world.gamma
            
            state = next_state
        
        return episode, total_return
    
    def estimate_utilities(self, num_episodes=1000, policy_func=None):
        if policy_func is None:
            policy_func = self.grid_world.get_fixed_policy_action
        
        self.state_returns = defaultdict(list)
        
        for episode_num in range(num_episodes):
            episode, total_return = self.generate_episode(policy_func)
            
            for state, action, reward, next_state in episode:
                if state not in self.terminal_states:
                    self.state_returns[state].append(total_return)
        
        for state in self.state_returns:
            self.state_utilities[state] = np.mean(self.state_returns[state])
    
    def print_utilities(self):
        print("Estimated State Utilities:")
        print("=" * 30)
        
        utility_grid = np.full((self.grid_world.rows, self.grid_world.cols), np.nan)
        
        for state, utility in self.state_utilities.items():
            row, col = state
            utility_grid[row, col] = utility
        
        for row in range(self.grid_world.rows):
            for col in range(self.grid_world.cols):
                state = (row, col)
                if state == self.grid_world.wall:
                    print(" WALL ", end="")
                elif state in self.grid_world.terminal_states:
                    print(f" {self.grid_world.terminal_states[state]:>4.1f} ", end="")
                elif state in self.state_utilities:
                    print(f" {self.state_utilities[state]:>4.2f}", end="")
                else:
                    print("  N/A ", end="")
            print()
        print()
    
    def plot_heatmap(self, title="Utility Estimates"):
        utility_grid = np.full((self.grid_world.rows, self.grid_world.cols), np.nan)
        
        for state, utility in self.state_utilities.items():
            row, col = state
            utility_grid[row, col] = utility
        
        plt.figure(figsize=(8, 6))
        plt.imshow(utility_grid, cmap='RdYlBu', aspect='auto')
        plt.colorbar(label='Utility Value')
        
        for row in range(self.grid_world.rows):
            for col in range(self.grid_world.cols):
                state = (row, col)
                if state == self.grid_world.wall:
                    plt.text(col, row, 'WALL', ha='center', va='center', fontweight='bold')
                elif state in self.grid_world.terminal_states:
                    plt.text(col, row, f'{self.grid_world.terminal_states[state]}', 
                            ha='center', va='center', fontweight='bold', fontsize=12)
                elif state in self.state_utilities:
                    plt.text(col, row, f'{self.state_utilities[state]:.3f}', 
                            ha='center', va='center')
        
        plt.title(title)
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.xticks(range(self.grid_world.cols))
        plt.yticks(range(self.grid_world.rows))
        plt.tight_layout()
        plt.show()

def compare_policies():
    print("Comparing Fixed vs Modified Policy")
    print("=" * 40)
    
    grid_world = GridWorld()
    estimator = PassiveUtilityEstimator(grid_world)
    
    print("Original Fixed Policy: 'Always go right if possible, else down'")
    estimator.estimate_utilities(num_episodes=1000, policy_func=grid_world.get_fixed_policy_action)
    estimator.print_utilities()
    estimator.plot_heatmap("Fixed Policy: Right Priority")
    
    print("\nModified Policy: 'Always go down if possible, else right'")
    estimator.estimate_utilities(num_episodes=1000, policy_func=grid_world.get_modified_policy_action)
    estimator.print_utilities()
    estimator.plot_heatmap("Modified Policy: Down Priority")

def add_obstacles():
    print("Adding More Obstacles and Terminal States")
    print("=" * 45)
    
    class ExtendedGridWorld(GridWorld):
        def __init__(self):
            super().__init__()
            self.terminal_states = {(0, 3): 1, (1, 3): -1, (2, 3): 0.5}
            self.walls = [(1, 1), (0, 1), (2, 1)]
        
        def is_valid_state(self, state):
            row, col = state
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                return False
            if state in self.walls:
                return False
            return True
    
    extended_grid = ExtendedGridWorld()
    estimator = PassiveUtilityEstimator(extended_grid)
    
    print("Extended Grid with additional terminal state and obstacles:")
    print("Terminal states: (0,3)=+1, (1,3)=-1, (2,3)=+0.5")
    print("Walls: (1,1), (0,1), (2,1)")
    
    estimator.estimate_utilities(num_episodes=1000)
    estimator.print_utilities()
    estimator.plot_heatmap("Extended Grid with More Obstacles")

def test_stochasticity():
    print("Testing Different Stochasticity Levels")
    print("=" * 40)
    
    stochasticity_levels = [0.05, 0.1, 0.2, 0.3]
    
    for stoch in stochasticity_levels:
        print(f"\nStochasticity Level: {stoch}")
        print("-" * 25)
        
        grid_world = GridWorld()
        grid_world.stochasticity = stoch
        estimator = PassiveUtilityEstimator(grid_world)
        
        estimator.estimate_utilities(num_episodes=1000)
        estimator.print_utilities()

def main():
    print("Passive Direct Utility Estimation in Grid World")
    print("=" * 50)
    
    print("Grid World Setup:")
    print("- 3x4 grid")
    print("- Terminal state (0,3) with reward +1")
    print("- Terminal state (1,3) with reward -1")
    print("- Wall at (1,1)")
    print("- Fixed policy: 'Always go right if possible, else down'")
    print("- 80% chance of intended movement, 10% each for perpendicular")
    print("- Discount factor γ = 0.9")
    print("- Step cost = -0.04")
    print()
    
    grid_world = GridWorld()
    estimator = PassiveUtilityEstimator(grid_world)
    
    print("Estimating utilities with 1000 episodes...")
    estimator.estimate_utilities(num_episodes=1000)
    
    print("\nResults:")
    estimator.print_utilities()
    estimator.plot_heatmap("Original Fixed Policy")
    
    print("\n" + "=" * 50)
    compare_policies()
    
    print("\n" + "=" * 50)
    add_obstacles()
    
    print("\n" + "=" * 50)
    test_stochasticity()
    
    print("\nAnalysis:")
    print("- Passive utility estimation learns state values from observed returns")
    print("- Higher stochasticity makes learning more challenging")
    print("- Policy modifications affect the utility estimates")
    print("- Additional obstacles and terminal states change the utility landscape")

if __name__ == "__main__":
    main()
