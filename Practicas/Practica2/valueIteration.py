# Declaración de constantes
T_MAX = 25
NUM_EPISODES = 10
GAMMA = 0.95
REWARD_THRESHOLD = 10

import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ValueIterationAgent:
    def __init__(self, env, gamma):
        self.env = env
        self.V = np.zeros(self.env.observation_space.n)
        self.gamma = gamma
        
    def calc_action_value(self, state, action):
        action_value = sum([prob * (reward + self.gamma * self.V[next_state])
                            for prob, next_state, reward, _ 
                            in self.env.unwrapped.P[state][action]]) 
        return action_value

    def select_action(self, state):
        best_action = best_value = None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if not best_value or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def value_iteration(self):
        max_diff = 0
        for state in range(self.env.observation_space.n):
            state_values = []
            for action in range(self.env.action_space.n):  
                state_values.append(self.calc_action_value(state, action))
            new_V = max(state_values)
            diff = abs(new_V - self.V[state])
            if diff > max_diff:
                max_diff = diff
            self.V[state] = new_V
        return self.V, max_diff
    
    def policy(self):   
        policy = np.zeros(env.observation_space.n) 
        for s in range(env.observation_space.n):
            Q_values = [self.calc_action_value(s,a) for a in range(self.env.action_space.n)] 
            policy[s] = np.argmax(np.array(Q_values))        
        return policy
    


def check_improvements():
    reward_test = 0.0
    for i in range(NUM_EPISODES):
        total_reward = 0.0
        state, _ = env.reset()
        for i in range(T_MAX):
            action = agent.select_action(state)
            new_state, new_reward, is_done, truncated, _ = env.step(action)
            total_reward += new_reward
            if is_done: 
                break
            state = new_state
        reward_test += total_reward
    reward_avg = reward_test / NUM_EPISODES
    return reward_avg

def train(agent): 
    rewards = []
    max_diffs = []
    t = 0
    best_reward = 0.0
     
    while best_reward < REWARD_THRESHOLD:
        _, max_diff = agent.value_iteration()
        max_diffs.append(max_diff)
        print("After value iteration, max_diff = " + str(max_diff))
        t += 1
        reward_test = check_improvements()
        rewards.append(reward_test)
               
        if reward_test > best_reward:
            print(f"Best reward updated {reward_test:.2f} at iteration {t}") 
            best_reward = reward_test
    
    return rewards, max_diffs

def draw_rewards(rewards):
    data = pd.DataFrame({'Episode': range(1, len(rewards) + 1), 'Reward': rewards})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y='Reward', data=data)

    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()

    plt.show()






env = gym.make("Taxi-v3", render_mode="rgb_array")

def run_expertiment():
    rewards, max_diffs = train(agent)

    is_done = False
    rewards = []
    for n_ep in range(NUM_EPISODES):
        state, _ = env.reset()
        print('Episode: ', n_ep)
        total_reward = 0
        for i in range(T_MAX):
            action = agent.select_action(state)
            state, reward, is_done, truncated, _ = env.step(action)
            total_reward = total_reward + reward
            env.render()
            if is_done:
                break
        rewards.append(total_reward)
    draw_rewards(rewards)

#Defineix els paràmetres del teu experiment aquí:
T_MAX = 50
NUM_EPISODES = 10
GAMMA = 0.95
REWARD_THRESHOLD = 10

agent = ValueIterationAgent(env, gamma=GAMMA)
run_expertiment()