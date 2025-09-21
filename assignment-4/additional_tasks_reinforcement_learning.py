import numpy as np
from collections import defaultdict

class ModelBasedRL:
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.rewards = defaultdict(lambda: defaultdict(float))
        self.states = set()
        self.actions = set()
        
    def learn_model_from_episodes(self, training_episodes):
        print("Learning Environment Model from Episodes")
        print("=" * 50)
        
        for episode in training_episodes:
            for transition in episode:
                state, action, reward, next_state = transition
                self.states.add(state)
                self.states.add(next_state)
                self.actions.add(action)
                
                self.rewards[state][action] = reward
                self.transitions[state][action][next_state] += 1
        
        for state in self.transitions:
            for action in self.transitions[state]:
                total_count = sum(self.transitions[state][action].values())
                for next_state in self.transitions[state][action]:
                    self.transitions[state][action][next_state] /= total_count
        
        print(f"Learned model for {len(self.states)} states and {len(self.actions)} actions")
        return self.transitions, self.rewards
    
    def value_iteration(self, threshold=1e-6, max_iterations=1000):
        print("\nApplying Value Iteration")
        print("=" * 30)
        
        V = {state: 0.0 for state in self.states}
        
        for iteration in range(max_iterations):
            V_old = V.copy()
            delta = 0
            
            for state in self.states:
                best_value = -np.inf
                
                for action in self.actions:
                    if action in self.transitions[state]:
                        action_value = 0
                        for next_state, prob in self.transitions[state][action].items():
                            reward = self.rewards[state][action]
                            action_value += prob * (reward + self.gamma * V_old[next_state])
                        
                        best_value = max(best_value, action_value)
                
                V[state] = best_value
                delta = max(delta, abs(V[state] - V_old[state]))
            
            if delta < threshold:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return V
    
    def extract_policy(self, V):
        print("\nExtracting Optimal Policy")
        print("=" * 30)
        
        policy = {}
        
        for state in self.states:
            best_action = None
            best_value = -np.inf
            
            for action in self.actions:
                if action in self.transitions[state]:
                    action_value = 0
                    for next_state, prob in self.transitions[state][action].items():
                        reward = self.rewards[state][action]
                        action_value += prob * (reward + self.gamma * V[next_state])
                    
                    if action_value > best_value:
                        best_value = action_value
                        best_action = action
            
            policy[state] = best_action
        
        return policy
    
    def print_value_grid(self, V, grid_size=5):
        print("\nOptimal Value Function Grid:")
        print("=" * 30)
        
        print("   ", end="")
        for col in range(grid_size):
            print(f"{col:>8}", end="")
        print()
        
        for row in range(grid_size):
            print(f"{row}  ", end="")
            for col in range(grid_size):
                state = (row, col)
                if state in V:
                    print(f"{V[state]:>8.2f}", end="")
                else:
                    print("      N/A", end="")
            print()
        print()
    
    def print_policy_grid(self, policy, grid_size=5):
        print("Optimal Policy Grid:")
        print("=" * 25)
        
        arrow_map = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}
        
        print("   ", end="")
        for col in range(grid_size):
            print(f"{col:>4}", end="")
        print()
        
        for row in range(grid_size):
            print(f"{row}  ", end="")
            for col in range(grid_size):
                state = (row, col)
                if state in policy and policy[state]:
                    print(f"  {arrow_map[policy[state]]} ", end="")
                else:
                    print("  ? ", end="")
            print()
        print()

def create_training_episodes():
    training_episodes = [
        [((0, 0), 'U', 0, (0, 0)), ((0, 0), 'D', 0, (1, 0)), ((0, 0), 'L', 0, (0, 0)), ((0, 0), 'R', -1, (0, 1))],
        [((0, 1), 'U', 0, (0, 1)), ((0, 1), 'D', 0, (1, 1)), ((0, 1), 'L', 0, (0, 0)), ((0, 1), 'R', 0, (0, 2))],
        [((0, 2), 'U', 0, (0, 2)), ((0, 2), 'D', 0, (1, 2)), ((0, 2), 'L', -1, (0, 1)), ((0, 2), 'R', 0, (0, 3))],
        [((0, 3), 'U', 0, (0, 3)), ((0, 3), 'D', 0, (1, 3)), ((0, 3), 'L', 0, (0, 2)), ((0, 3), 'R', 0, (0, 4))],
        [((0, 4), 'U', 0, (0, 4)), ((0, 4), 'D', 0, (1, 4)), ((0, 4), 'L', 0, (0, 3)), ((0, 4), 'R', 0, (0, 4))],
        [((1, 0), 'U', 0, (0, 0)), ((1, 0), 'D', 0, (2, 0)), ((1, 0), 'L', 0, (1, 0)), ((1, 0), 'R', 0, (1, 1))],
        [((1, 1), 'U', -1, (0, 1)), ((1, 1), 'D', -10, (2, 1)), ((1, 1), 'L', 0, (1, 0)), ((1, 1), 'R', 0, (1, 2))],
        [((1, 2), 'U', 0, (0, 2)), ((1, 2), 'D', 0, (2, 2)), ((1, 2), 'L', 0, (1, 1)), ((1, 2), 'R', 0, (1, 3))],
        [((1, 3), 'U', 0, (0, 3)), ((1, 3), 'D', 0, (2, 3)), ((1, 3), 'L', 0, (1, 2)), ((1, 3), 'R', 0, (1, 4))],
        [((1, 4), 'U', 0, (0, 4)), ((1, 4), 'D', 0, (2, 4)), ((1, 4), 'L', 0, (1, 3)), ((1, 4), 'R', 0, (1, 4))],
        [((2, 0), 'U', 0, (1, 0)), ((2, 0), 'D', 0, (3, 0)), ((2, 0), 'L', 0, (2, 0)), ((2, 0), 'R', -10, (2, 1))],
        [((2, 1), 'U', 0, (1, 1)), ((2, 1), 'D', 0, (3, 1)), ((2, 1), 'L', 0, (2, 0)), ((2, 1), 'R', 0, (2, 2))],
        [((2, 2), 'U', 0, (1, 2)), ((2, 2), 'D', 0, (3, 2)), ((2, 2), 'L', -10, (2, 1)), ((2, 2), 'R', 0, (2, 3))],
        [((2, 3), 'U', 0, (1, 3)), ((2, 3), 'D', 0, (3, 3)), ((2, 3), 'L', 0, (2, 2)), ((2, 3), 'R', 0, (2, 4))],
        [((2, 4), 'U', 0, (1, 4)), ((2, 4), 'D', 10, (3, 4)), ((2, 4), 'L', 0, (2, 3)), ((2, 4), 'R', 0, (2, 4))],
        [((3, 0), 'U', 0, (2, 0)), ((3, 0), 'D', 0, (4, 0)), ((3, 0), 'L', 0, (3, 0)), ((3, 0), 'R', 0, (3, 1))],
        [((3, 1), 'U', -10, (2, 1)), ((3, 1), 'D', 0, (4, 1)), ((3, 1), 'L', 0, (3, 0)), ((3, 1), 'R', 0, (3, 2))],
        [((3, 2), 'U', 0, (2, 2)), ((3, 2), 'D', 0, (4, 2)), ((3, 2), 'L', 0, (3, 1)), ((3, 2), 'R', 0, (3, 3))],
        [((3, 3), 'U', 0, (2, 3)), ((3, 3), 'D', 0, (4, 3)), ((3, 3), 'L', 0, (3, 2)), ((3, 3), 'R', 10, (3, 4))],
        [((3, 4), 'U', 0, (2, 4)), ((3, 4), 'D', 0, (4, 4)), ((3, 4), 'L', 0, (3, 3)), ((3, 4), 'R', 0, (3, 4))],
        [((4, 0), 'U', 0, (3, 0)), ((4, 0), 'D', 0, (4, 0)), ((4, 0), 'L', 0, (4, 0)), ((4, 0), 'R', 0, (4, 1))],
        [((4, 1), 'U', 0, (3, 1)), ((4, 1), 'D', 0, (4, 1)), ((4, 1), 'L', 0, (4, 0)), ((4, 1), 'R', 0, (4, 2))],
        [((4, 2), 'U', 0, (3, 2)), ((4, 2), 'D', 0, (4, 2)), ((4, 2), 'L', 0, (4, 1)), ((4, 2), 'R', 0, (4, 3))],
        [((4, 3), 'U', 0, (3, 3)), ((4, 3), 'D', 0, (4, 3)), ((4, 3), 'L', 0, (4, 2)), ((4, 3), 'R', 0, (4, 4))],
        [((4, 4), 'U', 10, (3, 4)), ((4, 4), 'D', 0, (4, 4)), ((4, 4), 'L', 0, (4, 3)), ((4, 4), 'R', 0, (4, 4))]
    ]
    return training_episodes

def main():
    print("Model-Based Reinforcement Learning")
    print("=" * 50)
    print("Learning environment model from episodes and applying value iteration")
    print()
    
    rl_agent = ModelBasedRL(gamma=0.9)
    training_episodes = create_training_episodes()
    
    print("Training Episodes Sample:")
    print("Format: (state, action, reward, next_state)")
    for i, episode in enumerate(training_episodes[:3]):
        print(f"\nEpisode {i+1}:")
        for transition in episode[:2]:
            print(f"  {transition}")
        if len(episode) > 2:
            print("  ...")
    
    transitions, rewards = rl_agent.learn_model_from_episodes(training_episodes)
    
    print("\nLearned Transition Probabilities (sample):")
    for state in list(transitions.keys())[:3]:
        print(f"\nState {state}:")
        for action in transitions[state]:
            print(f"  Action {action}:")
            for next_state, prob in transitions[state][action].items():
                print(f"    {next_state}: {prob:.3f}")
    
    V = rl_agent.value_iteration()
    
    rl_agent.print_value_grid(V)
    
    policy = rl_agent.extract_policy(V)
    
    rl_agent.print_policy_grid(policy)
    
    print("Analysis:")
    print("- Model-based RL learned the environment dynamics from episodes")
    print("- Value iteration found the optimal value function")
    print("- The optimal policy maximizes cumulative discounted rewards")
    print("- Goal states (with +10 reward) have high values")
    print("- Trap states (with -10 penalty) have low values")
    
    print("\nKey Findings:")
    print("- The agent learns to navigate towards high-reward states")
    print("- The policy avoids states with negative rewards")
    print("- The discount factor γ=0.9 balances immediate vs future rewards")

if __name__ == "__main__":
    main()
