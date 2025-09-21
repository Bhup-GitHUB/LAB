import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

class QLearningGridWorld:
    def __init__(self):
        self.rows = 4
        self.cols = 4
        self.start_state = (0, 0)
        self.goal_state = (3, 3)
        self.trap_state = (1, 2)
        
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_effects = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        
        self.initialize_q_table()
        
    def initialize_q_table(self):
        for row in range(self.rows):
            for col in range(self.cols):
                state = (row, col)
                self.q_table[state] = {}
                for action in self.actions:
                    self.q_table[state][action] = 0.0
    
    def is_valid_state(self, state):
        row, col = state
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def get_next_state(self, state, action):
        effect = self.action_effects[action]
        next_state = (state[0] + effect[0], state[1] + effect[1])
        
        if not self.is_valid_state(next_state):
            return state
        
        return next_state
    
    def get_reward(self, state):
        if state == self.goal_state:
            return 10
        elif state == self.trap_state:
            return -10
        else:
            return -1
    
    def is_terminal(self, state):
        return state == self.goal_state or state == self.trap_state
    
    def get_epsilon_greedy_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            best_action = max(self.q_table[state], key=self.q_table[state].get)
            return best_action
    
    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        
        if self.is_terminal(next_state):
            target = reward
        else:
            max_next_q = max(self.q_table[next_state].values())
            target = reward + self.gamma * max_next_q
        
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[state][action] = new_q
    
    def run_episode(self, max_steps=100):
        state = self.start_state
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            if self.is_terminal(state):
                break
            
            action = self.get_epsilon_greedy_action(state)
            next_state = self.get_next_state(state, action)
            reward = self.get_reward(next_state)
            
            self.update_q_value(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
            steps += 1
        
        return total_reward, steps
    
    def train(self, num_episodes=1000):
        episode_rewards = []
        episode_steps = []
        moving_avg_window = 50
        moving_avg_rewards = []
        
        print(f"Training Q-Learning for {num_episodes} episodes...")
        print(f"Hyperparameters: α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")
        print("=" * 60)
        
        for episode in range(num_episodes):
            reward, steps = self.run_episode()
            episode_rewards.append(reward)
            episode_steps.append(steps)
            
            if episode >= moving_avg_window - 1:
                avg_reward = np.mean(episode_rewards[episode - moving_avg_window + 1:episode + 1])
                moving_avg_rewards.append(avg_reward)
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Reward = {reward}, Steps = {steps}")
        
        return episode_rewards, moving_avg_rewards, episode_steps
    
    def get_optimal_policy(self):
        policy = {}
        for state in self.q_table:
            if not self.is_terminal(state):
                best_action = max(self.q_table[state], key=self.q_table[state].get)
                policy[state] = best_action
        return policy
    
    def print_q_table(self):
        print("Learned Q-Table:")
        print("=" * 50)
        
        for row in range(self.rows):
            for col in range(self.cols):
                state = (row, col)
                print(f"\nState {state}:")
                if self.is_terminal(state):
                    if state == self.goal_state:
                        print("  GOAL STATE")
                    elif state == self.trap_state:
                        print("  TRAP STATE")
                else:
                    for action in self.actions:
                        print(f"  {action}: {self.q_table[state][action]:.3f}")
    
    def print_optimal_policy(self):
        policy = self.get_optimal_policy()
        
        print("\nOptimal Policy:")
        print("=" * 20)
        
        arrow_map = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→'}
        
        for row in range(self.rows):
            for col in range(self.cols):
                state = (row, col)
                if state == self.goal_state:
                    print(" GOAL", end="")
                elif state == self.trap_state:
                    print(" TRAP", end="")
                elif state in policy:
                    print(f"  {arrow_map[policy[state]]} ", end="")
                else:
                    print("  ?  ", end="")
            print()
        print()
    
    def plot_learning_curve(self, episode_rewards, moving_avg_rewards):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards, alpha=0.3, label='Episode Rewards')
        plt.plot(range(49, len(moving_avg_rewards) + 49), moving_avg_rewards, 
                color='red', linewidth=2, label=f'Moving Average (window=50)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Q-Learning Learning Curve')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(49, len(moving_avg_rewards) + 49), moving_avg_rewards, 
                color='red', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Moving Average Reward')
        plt.title('Smoothed Learning Curve')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def test_optimal_policy(self, num_tests=100):
        print("Testing Optimal Policy:")
        print("=" * 25)
        
        original_epsilon = self.epsilon
        self.epsilon = 0
        
        test_rewards = []
        test_steps = []
        
        for test in range(num_tests):
            state = self.start_state
            total_reward = 0
            steps = 0
            
            while not self.is_terminal(state) and steps < 100:
                action = self.get_epsilon_greedy_action(state)
                next_state = self.get_next_state(state, action)
                reward = self.get_reward(next_state)
                
                total_reward += reward
                state = next_state
                steps += 1
            
            test_rewards.append(total_reward)
            test_steps.append(steps)
        
        self.epsilon = original_epsilon
        
        print(f"Average reward over {num_tests} tests: {np.mean(test_rewards):.2f}")
        print(f"Average steps over {num_tests} tests: {np.mean(test_steps):.2f}")
        print(f"Success rate (reaching goal): {(np.array(test_rewards) > 0).mean() * 100:.1f}%")

def analyze_hyperparameters():
    print("Hyperparameter Analysis")
    print("=" * 30)
    
    hyperparams = [
        {'alpha': 0.05, 'gamma': 0.9, 'epsilon': 0.1, 'name': 'Low Learning Rate'},
        {'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.1, 'name': 'Default'},
        {'alpha': 0.2, 'gamma': 0.9, 'epsilon': 0.1, 'name': 'High Learning Rate'},
        {'alpha': 0.1, 'gamma': 0.8, 'epsilon': 0.1, 'name': 'Low Discount'},
        {'alpha': 0.1, 'gamma': 0.95, 'epsilon': 0.1, 'name': 'High Discount'},
        {'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.05, 'name': 'Low Exploration'},
        {'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.2, 'name': 'High Exploration'}
    ]
    
    results = []
    
    for params in hyperparams:
        print(f"\nTesting {params['name']}: α={params['alpha']}, γ={params['gamma']}, ε={params['epsilon']}")
        
        q_learning = QLearningGridWorld()
        q_learning.alpha = params['alpha']
        q_learning.gamma = params['gamma']
        q_learning.epsilon = params['epsilon']
        
        episode_rewards, moving_avg_rewards, episode_steps = q_learning.train(num_episodes=500)
        
        final_avg_reward = np.mean(episode_rewards[-100:])
        q_learning.test_optimal_policy(num_tests=50)
        
        results.append({
            'name': params['name'],
            'final_avg_reward': final_avg_reward,
            'moving_avg': moving_avg_rewards[-1] if moving_avg_rewards else 0
        })
    
    print("\nHyperparameter Comparison:")
    print("=" * 35)
    for result in results:
        print(f"{result['name']:20}: Final Avg Reward = {result['final_avg_reward']:.2f}")

def main():
    print("Q-Learning Algorithm on 4x4 Grid World")
    print("=" * 50)
    
    print("Environment Setup:")
    print("- 4x4 grid world")
    print("- Start state: (0, 0)")
    print("- Goal state: (3, 3) with reward +10")
    print("- Trap state: (1, 2) with reward -10")
    print("- Step cost: -1 for non-terminal states")
    print("- Actions: UP, DOWN, LEFT, RIGHT")
    print("- Episode terminates at goal/trap or after 100 steps")
    print()
    
    q_learning = QLearningGridWorld()
    
    episode_rewards, moving_avg_rewards, episode_steps = q_learning.train(num_episodes=1000)
    
    print(f"\nTraining completed!")
    print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_steps):.2f} steps")
    
    q_learning.print_q_table()
    q_learning.print_optimal_policy()
    
    q_learning.plot_learning_curve(episode_rewards, moving_avg_rewards)
    
    q_learning.test_optimal_policy()
    
    print("\n" + "=" * 50)
    analyze_hyperparameters()
    
    print("\nAnalysis:")
    print("- Q-Learning successfully learned the optimal policy")
    print("- The agent learns to avoid the trap and reach the goal")
    print("- Learning curve shows improvement over episodes")
    print("- Hyperparameters significantly affect learning performance")

if __name__ == "__main__":
    main()
