import random

import numpy as np

from solution_concepts import SolutionConcept
from game_model import GameModel
import abc


class MARLAlgorithm(abc.ABC):
    @abc.abstractmethod
    def learn(self, joint_action, rewards, next_state: int, observations):
        pass

    @abc.abstractmethod
    def explain(self):
        pass

    @abc.abstractmethod
    def select_action(self, state):
        pass


class JALGT(MARLAlgorithm):
    def __init__(self, agent_id, game: GameModel, solution_concept: SolutionConcept,
                 gamma=0.95, alpha=0.5, epsilon=0.2, seed=42):
        self.agent_id = agent_id
        self.game = game
        self.solution_concept = solution_concept
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        # Q: N x S x AS
        self.q_table = np.zeros((self.game.num_agents, self.game.num_states,
                                 len(self.game.action_space)))
        # Política conjunta por defecto: distribución uniforme respecto
        # de las acciones conjuntas, para cada acción (pi(a | s))
        self.joint_policy = np.ones((self.game.num_agents, self.game.num_states,
                                     self.game.num_actions)) / self.game.num_actions
        self.metrics = {"td_error": []}

    def value(self, agent_id, state):
        value = 0
        for idx, joint_action in enumerate(self.game.action_space):
            payoff = self.q_table[agent_id][state][idx]
            joint_probability = np.prod([self.joint_policy[i][state][joint_action[i]]
                                         for i in range(self.game.num_agents)])
            value += payoff * joint_probability
        return value

    def update_policy(self, agent_id, state):
        self.joint_policy[agent_id][state] = self.solution_concept.solution_policy(agent_id, state, self.game,
                                                                                   self.q_table)

    def learn(self, joint_action, rewards, state, next_state):
        joint_action_index = self.game.action_space_index[joint_action]
        for agent_id in range(self.game.num_agents):
            agent_reward = rewards[agent_id]
            agent_game_value_next_state = self.value(agent_id, next_state)
            agent_q_value = self.q_table[agent_id][state][joint_action_index]
            td_target = agent_reward + self.gamma * agent_game_value_next_state - agent_q_value
            self.q_table[agent_id][state][joint_action_index] += self.alpha * td_target
            self.update_policy(agent_id, state)
            # Guardamos el error de diferencia temporal para estadísticas posteriores
            self.metrics['td_error'].append(td_target)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def solve(self, agent_id, state):
        return self.joint_policy[agent_id][state]

    def select_action(self, state, train=True):
        if train:
            if self.rng.random() < self.epsilon:
                return self.rng.choice(range(self.game.num_actions))
            else:
                probs = self.solve(self.agent_id, state)
                np.random.seed(self.rng.randint(0, 10000))
                return np.random.choice(range(self.game.num_actions), p=probs)
        else:
            return np.argmax(self.solve(self.agent_id, state))

    def explain(self, state=0):
        return self.solution_concept.debug(self.agent_id, state, self.game, self.q_table)
    

class QLearningAgent(MARLAlgorithm):
    def __init__(self, env, gamma, learning_rate, epsilon, t_max, agent_id, exp_config):
        self.env = env
        self.Q = np.zeros((exp_config["num_states"], env.action_space.n)) 
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.t_max = t_max
        self.agent_id = agent_id
        self.metrics = {"td_error": []}

    #def select_action(self, state, training=True):
    #    if training and random.random() <= self.epsilon:
    #        return np.random.choice(self.env.action_space.n)
    #    else:
    #        return np.argmax(self.Q[state, ])   
    def select_action(self, state, train=None):  # Accept the optional train argument
        state = int(state)
        if random.random() <= self.epsilon:  # We only explore during training
            return np.random.choice(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state, :])

    def update_Q(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state, ])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error

    def learn_from_episode(self):
        state, _ = self.env.reset()
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
        policy = np.zeros(self.env.observation_space.n)
        for s in range(self.observation_space.n):
            policy[s] = np.argmax(np.array(self.Q[s]))
        return policy
    
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
    
    def learn(self, actions, rewards, state, next_state):
        best_next_action = np.argmax(self.Q[next_state, ])
        reward = rewards[self.agent_id]  # Recompensa del agente actual
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, actions[self.agent_id]]
        self.Q[state, actions[self.agent_id]] += self.learning_rate * td_error
        self.metrics['td_error'].append(td_error)

    def explain(self):
        pass

