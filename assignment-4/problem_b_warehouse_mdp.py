import numpy as np
import matplotlib.pyplot as plt

class WarehouseMDP:
    def __init__(self):
        self.grid_size = 4
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.gamma = 0.9
        
        self.rewards = np.array([
            [0, 0, -1, 10],
            [0, -10, 0, 0],
            [0, 0, 0, -10],
            [0, -1, 0, 0]
        ])
        
        self.start_state = (0, 0)
        self.goal_state = (0, 3)
        
        self.transition_probabilities = {
            'UP': {'UP': 0.8, 'LEFT': 0.1, 'RIGHT': 0.1},
            'DOWN': {'DOWN': 0.8, 'LEFT': 0.1, 'RIGHT': 0.1},
            'LEFT': {'LEFT': 0.8, 'UP': 0.1, 'DOWN': 0.1},
            'RIGHT': {'RIGHT': 0.8, 'UP': 0.1, 'DOWN': 0.1}
        }
    
    def is_valid_state(self, state):
        row, col = state
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size
    
    def get_next_states(self, state, action):
        row, col = state
        next_states = {}
        
        intended_direction = action
        prob_dist = self.transition_probabilities[intended_direction]
        
        for actual_action, prob in prob_dist.items():
            if actual_action == 'UP':
                next_state = (row - 1, col)
            elif actual_action == 'DOWN':
                next_state = (row + 1, col)
            elif actual_action == 'LEFT':
                next_state = (row, col - 1)
            elif actual_action == 'RIGHT':
                next_state = (row, col + 1)
            
            if not self.is_valid_state(next_state):
                next_state = state
            
            next_states[next_state] = prob
        
        return next_states
    
    def get_reward(self, state):
        row, col = state
        return self.rewards[row, col]
    
    def value_iteration(self, threshold=1e-6, max_iterations=1000):
        print("Value Iteration Algorithm")
        print("=" * 40)
        
        V = np.zeros((self.grid_size, self.grid_size))
        policy = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        for iteration in range(max_iterations):
            V_old = V.copy()
            
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    state = (row, col)
                    
                    if state == self.goal_state:
                        V[row, col] = self.get_reward(state)
                        continue
                    
                    best_value = -np.inf
                    best_action = 0
                    
                    for action_idx, action in enumerate(self.actions):
                        action_value = 0
                        next_states = self.get_next_states(state, action)
                        
                        for next_state, prob in next_states.items():
                            next_row, next_col = next_state
                            reward = self.get_reward(next_state)
                            action_value += prob * (reward + self.gamma * V_old[next_row, next_col])
                        
                        if action_value > best_value:
                            best_value = action_value
                            best_action = action_idx
                    
                    V[row, col] = best_value
                    policy[row, col] = best_action
            
            if np.max(np.abs(V - V_old)) < threshold:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return V, policy
    
    def policy_iteration(self, threshold=1e-6, max_iterations=1000):
        print("Policy Iteration Algorithm")
        print("=" * 40)
        
        policy = np.zeros((self.grid_size, self.grid_size), dtype=int)
        V = np.zeros((self.grid_size, self.grid_size))
        
        for iteration in range(max_iterations):
            V_old = V.copy()
            
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    state = (row, col)
                    
                    if state == self.goal_state:
                        V[row, col] = self.get_reward(state)
                        continue
                    
                    action = self.actions[policy[row, col]]
                    action_value = 0
                    next_states = self.get_next_states(state, action)
                    
                    for next_state, prob in next_states.items():
                        next_row, next_col = next_state
                        reward = self.get_reward(next_state)
                        action_value += prob * (reward + self.gamma * V_old[next_row, next_col])
                    
                    V[row, col] = action_value
            
            if np.max(np.abs(V - V_old)) < threshold:
                break
            
            policy_stable = True
            
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    state = (row, col)
                    
                    if state == self.goal_state:
                        continue
                    
                    old_action = policy[row, col]
                    best_action = 0
                    best_value = -np.inf
                    
                    for action_idx, action in enumerate(self.actions):
                        action_value = 0
                        next_states = self.get_next_states(state, action)
                        
                        for next_state, prob in next_states.items():
                            next_row, next_col = next_state
                            reward = self.get_reward(next_state)
                            action_value += prob * (reward + self.gamma * V[next_row, next_col])
                        
                        if action_value > best_value:
                            best_value = action_value
                            best_action = action_idx
                    
                    if best_action != old_action:
                        policy_stable = False
                    
                    policy[row, col] = best_action
            
            if policy_stable:
                print(f"Policy converged after {iteration + 1} iterations")
                break
        
        return V, policy
    
    def print_grid(self, title, grid, format_func=None):
        print(f"\n{title}:")
        print("  ", end="")
        for col in range(self.grid_size):
            print(f"{col:>8}", end="")
        print()
        
        for row in range(self.grid_size):
            print(f"{row} ", end="")
            for col in range(self.grid_size):
                if format_func:
                    print(f"{format_func(grid[row, col]):>8}", end="")
                else:
                    print(f"{grid[row, col]:>8.2f}", end="")
            print()
        print()
    
    def print_policy(self, policy):
        print("\nOptimal Policy (Arrows):")
        arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        
        print("  ", end="")
        for col in range(self.grid_size):
            print(f"{col:>4}", end="")
        print()
        
        for row in range(self.grid_size):
            print(f"{row} ", end="")
            for col in range(self.grid_size):
                if (row, col) == self.goal_state:
                    print("  G ", end="")
                elif (row, col) == self.start_state:
                    print("  S ", end="")
                else:
                    print(f"  {arrow_map[policy[row, col]]} ", end="")
            print()
        print()
    
    def simulate_optimal_policy(self, policy, num_episodes=10):
        print("Simulating Optimal Policy")
        print("=" * 30)
        
        total_rewards = []
        
        for episode in range(num_episodes):
            state = self.start_state
            total_reward = 0
            steps = 0
            path = [state]
            
            while state != self.goal_state and steps < 50:
                action = self.actions[policy[state[0], state[1]]]
                next_states = self.get_next_states(state, action)
                
                next_state = np.random.choice(
                    list(next_states.keys()),
                    p=list(next_states.values())
                )
                
                reward = self.get_reward(next_state)
                total_reward += reward * (self.gamma ** steps)
                
                state = next_state
                path.append(state)
                steps += 1
            
            total_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
            print(f"Path: {path[:5]}..." if len(path) > 5 else f"Path: {path}")
        
        print(f"\nAverage reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
        return total_rewards

def main():
    print("Warehouse Robot Navigation - Markov Decision Process")
    print("=" * 60)
    
    mdp = WarehouseMDP()
    
    print("Environment Setup:")
    print(f"Grid size: {mdp.grid_size}x{mdp.grid_size}")
    print(f"Start state: {mdp.start_state}")
    print(f"Goal state: {mdp.goal_state}")
    print(f"Discount factor (γ): {mdp.gamma}")
    print()
    
    print("Reward Structure:")
    mdp.print_grid("Reward Grid", mdp.rewards, lambda x: f"{x:>4}")
    
    print("MDP Components:")
    print("- States: Each cell in the 4x4 grid")
    print("- Actions: UP, DOWN, LEFT, RIGHT")
    print("- Transition Probabilities:")
    print("  - Intended direction: 0.8")
    print("  - Perpendicular directions: 0.1 each")
    print("- Rewards: +10 (goal), -10 (heavy obstacle), -1 (light obstacle), 0 (normal)")
    print()
    
    print("Solving MDP using Value Iteration:")
    V_vi, policy_vi = mdp.value_iteration()
    
    mdp.print_grid("Optimal Value Function (Value Iteration)", V_vi)
    mdp.print_policy(policy_vi)
    
    print("Solving MDP using Policy Iteration:")
    V_pi, policy_pi = mdp.policy_iteration()
    
    mdp.print_grid("Optimal Value Function (Policy Iteration)", V_pi)
    mdp.print_policy(policy_pi)
    
    print("Comparing Results:")
    print(f"Value functions identical: {np.allclose(V_vi, V_pi)}")
    print(f"Policies identical: {np.array_equal(policy_vi, policy_pi)}")
    
    print("\nSimulating Optimal Policy:")
    rewards = mdp.simulate_optimal_policy(policy_vi)
    
    print("\nAlgorithm Analysis:")
    print("- Both Value Iteration and Policy Iteration found optimal solutions")
    print("- The optimal policy guides the robot to maximize cumulative reward")
    print("- The robot learns to avoid heavy obstacles (-10) and reach the goal (+10)")

if __name__ == "__main__":
    main()
