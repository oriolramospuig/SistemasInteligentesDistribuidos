# Imports necesarios
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Declaración de constantes
TRAINING_EPISODES = 1000
NUM_EPISODES = 100
GAMMA = 0.95
LEARNING_RATE = 0.2
LEARNING_RATE_DECAY = 0.99  # Decay rate ajustado
EPSILON = 1.0  # Valor inicial de epsilon
EPSILON_DECAY = 0.995  # Decay rate para epsilon
MIN_EPSILON = 0.1  # Valor mínimo para epsilon

# Cargar el entorno Taxi-v3
env = gym.make("Taxi-v3")
env.reset(seed=0)

def draw_history(history, title):
    window_size = 50
    data = pd.DataFrame({'Episode': range(1, len(history) + 1), title: history})
    data['rolling_avg'] = data[title].rolling(window_size).mean()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y=title, data=data)
    sns.lineplot(x='Episode', y='rolling_avg', data=data)

    plt.title(title + ' Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

class ReinforceAgent:
    def __init__(self, env, gamma, learning_rate, lr_decay=1, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1, seed=0):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.state_space = env.observation_space.n
        self.action_space = env.action_space.n
        self.policy = np.ones([self.state_space, self.action_space]) / self.action_space
        np.random.seed(seed)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)  # Exploración
        else:
            return np.random.choice(np.arange(self.action_space), p=self.policy[state])  # Explotación

    def update_policy(self, trajectory):
        for t, (state, action, reward) in enumerate(trajectory):
            G = sum([self.gamma ** i * traj[2] for i, traj in enumerate(trajectory[t:])])
            self.policy[state, action] += self.learning_rate * (G - np.sum(self.policy[state] * G))
            # Normalizar la política
            self.policy[state] = np.clip(self.policy[state], 1e-10, None)  # Evitar valores negativos
            self.policy[state] /= np.sum(self.policy[state])  # Asegurar que las probabilidades sumen 1

    def train(self, episodes):
        rewards = []
        for episode in range(episodes):
            if episode % 100 == 0:
              print('episode: ' + str(episode))
            state, _ = self.env.reset()
            trajectory = []
            total_reward = 0
            done = False

            count = 0
            while not done and count < 500:
                action = self.choose_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                trajectory.append((state, action, reward))
                total_reward += reward
                state = next_state
                count += 1

            self.update_policy(trajectory)
            rewards.append(total_reward)
            self.learning_rate *= self.lr_decay

            # Decaimiento de epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

        return rewards

# Entrenamiento del agente
agent = ReinforceAgent(env, GAMMA, LEARNING_RATE, LEARNING_RATE_DECAY, EPSILON, EPSILON_DECAY, MIN_EPSILON)
training_rewards = agent.train(TRAINING_EPISODES)

# Visualización de resultados
draw_history(training_rewards, "Total Reward")

# Evaluación del agente entrenado
test_rewards = []
for i in range(NUM_EPISODES):
    print('episode end: ' + str(i))
    state, _ = env.reset()
    total_reward = 0
    done = False
    count = 0
    while not done and count < 500:
        action = agent.choose_action(state)
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        count += 1
    test_rewards.append(total_reward)

print(f"Average Test Reward: {np.mean(test_rewards)}")
draw_history(test_rewards, "Test Reward")
