import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from gymnasium import Wrapper

# Declaración de constantes
T_MAX = 500
NUM_EPISODES = 700
GAMMA = 0.99
REWARD_THRESHOLD = 0.9

EPSILON = 1
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.1

LEARNING_RATE = 1
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_MIN = 0.05

DECAY_ITERATIONS = 5

#Base experimento num_episodes
'''
T_MAX = 100
NUM_EPISODES = 1000
GAMMA = 0.6 #0.95
REWARD_THRESHOLD = 0.9
LEARNING_RATE = 0.1
EPSILON = 0.1
'''

env = gym.make("Taxi-v3")

def test_episode(agent, env):
    env.reset()
    is_done = False
    t = 0

    while not is_done:
        action = agent.select_action()
        state, reward, is_done, truncated, info = env.step(action)
        t += 1
    return state, reward, is_done, truncated, info

def draw_rewards(rewards):
    data = pd.DataFrame({'Episode': range(1, len(rewards) + 1), 'Reward': rewards})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y='Reward', data=data)

    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(0, 20)

    plt.show()

def draw_steps(steps):
    data = pd.DataFrame({'Episode': range(1, len(steps) + 1), 'Steps': steps})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y='Steps', data=data)

    plt.title('Steps Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    plt.tight_layout()

    plt.show()

def print_policy(policy):
    visual_help = {0:'<', 1:'v', 2:'>', 3:'^'}
    policy_arrows = [visual_help[x] for x in policy]
    print(np.array(policy_arrows).reshape([-1, 4]))

class QLearningAgent:
    def __init__(self, env, gamma, learning_rate, epsilon, t_max):
        self.env = env
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.t_max = t_max

    def select_action(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state,])

    def update_Q(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state,])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error

    def learn_from_episode(self):
        state, _ = env.reset()
        total_reward = 0
        total_steps = 0
        for i in range(self.t_max):
            action = self.select_action(state)
            new_state, new_reward, is_done, truncated, _ = self.env.step(action)
            total_reward += new_reward
            total_steps += 1
            self.update_Q(state, action, new_reward, new_state)
            if is_done:
                break
            state = new_state
        return total_reward, total_steps

    def policy(self):
        policy = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n):
            policy[s] = np.argmax(np.array(self.Q[s]))
        return policy
    

agent = QLearningAgent(env, gamma=GAMMA, learning_rate=LEARNING_RATE, epsilon=EPSILON, t_max=T_MAX)
rewards = []
total_steps = []
for i in range(NUM_EPISODES):
    reward, steps = agent.learn_from_episode()
    #print("New reward: " + str(reward))
    rewards.append(reward)
    total_steps.append(steps)
    if i % DECAY_ITERATIONS == 0:
      if agent.learning_rate <= LEARNING_RATE_MIN:
        agent.learning_rate = LEARNING_RATE_MIN
      else:
        agent.learning_rate *= LEARNING_RATE_DECAY
      if agent.epsilon <= EPSILON_MIN:
        agent.epsilon = EPSILON_MIN
      else:
        agent.epsilon *= EPSILON_DECAY
      
draw_steps(total_steps)
draw_rewards(rewards)

### EVALUACION ###

is_done = False
rewards = []
n_done = 0
for n_ep in range(NUM_EPISODES):
    state, _ = env.reset()
    #print('Episode: ', n_ep)
    total_reward = 0
    for i in range(T_MAX):
        action = agent.select_action(state, training=False)
        state, reward, is_done, truncated, _ = env.step(action)
        total_reward = total_reward + reward
        #env.render()
        if is_done:
          n_done += 1
          break
    rewards.append(total_reward)
draw_rewards(rewards)
print('Episodios completados: ' + str(n_done/NUM_EPISODES))