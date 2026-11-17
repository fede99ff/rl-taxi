import gymnasium as gym
import random
import numpy as np

from tqdm import tqdm

##implementatio of QL agent

class Q_Learning_Agent:

    #Defining the Agent based on regular tabular Q-learning. It is based on a table
    #where for every possible state in the enviroment we have an entry with the
    #assosiated Q-value. The Q-values are learn during the traing using a greedy
    #policy following the updating rule:
    # Q_n(s, a) ← Q_n−1(s, a) + α*[r + max_a'(γ*Q_n−1(s′, a′)) − Q_n−1(s, a)] = (1-α)* Q_n−1(s, a) + α(r + γ*max_a'(Q_n−1(s′, a′)))
    #where max_a'(Q_n−1(s′, a′)) selects the action that maximize the Q value in s'.

    def __init__(self,
                 env,
                 is_training,
                 learning_rate = 0.1,
                 discount_factor = 0.99,
                 epsilon_decay = 0.995,
                 initial_epsilon = 1,
                 final_epsilon = 0.05
                 ):

        self.env = env
        self.is_training = is_training

        if not is_training:
            self.q_table= np.load("q_table.npy")
        else:
            self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

        self.alpha = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def get_action(self, state):
        #use a greedy policy to select next action if you are testing or if
        #a number bigger then epsilon is sampled during training
        if not self.is_training or random.uniform(0, 1) > self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            return self.env.action_space.sample()

    def update_q_values(self, state, action, reward, terminated, next_state):
        if terminated:
            #as there will not be other steps the bellman equation is truncated
            #without the q value for next state
            self.q_table[state, action] = (1 - self.alpha)*self.q_table[state,action] + self.alpha*(reward)

        else:
            self.q_table[state, action]  = (1 - self.alpha)*self.q_table[state, action] + self.alpha*(reward + self.discount_factor * np.max(self.q_table[next_state, : ]))

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self):
        np.save("q_table.npy", self.q_table)

##implementatio of SARSA agent

class SARSA_Agent:

    #Defining the agent using SARSA. Here the agent on policy meaning that even
    #if it still uses the table, it updates it by selecting the next action using
    #the current policy and not the next action as the one the maximixe the next
    #reward. The formula is the following:
    # Q_n(s, a) ← Q_n−1(s, a) + α*[r + γ*Q_n−1(s′, a′) − Q_n−1(s, a)] = (1-α)* Q_n−1(s, a) + α(r + γ*Q_n−1(s′, a′))
    #Notice that the next action is selected according to the current policy.

    def __init__(self,
                 env,
                 is_training,
                 learning_rate = 0.1,
                 discount_factor = 0.99,
                 epsilon_decay = 0.995,
                 initial_epsilon = 1,
                 final_epsilon = 0.05
                 ):
        self.env = env
        self.is_training = is_training

        if not is_training:
            self.q_table= np.load("SARSA_q_table.npy")
        else:
            self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

        self.alpha = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def get_action(self, state):

        #use a greedy policy to select next action if you are testing or if
        #a number bigger then epsilon is sampled during training

        if not self.is_training or random.uniform(0,1) > self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            return self.env.action_space.sample()

    def update_q_values(self, state, action, reward, terminated, next_state, next_action):
        if terminated:
            self.q_table[state][action] = (1 - self.alpha)*self.q_table[state][action] + self.alpha*(reward)
        else:
            self.q_table[state][action] = (1 - self.alpha)*self.q_table[state][action] + self.alpha*(reward + self.discount_factor*self.q_table[next_state][next_action])

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self):
        np.save("SARSA_q_table.npy", self.q_table)

##Loop for training and testing the tabular agents

def run_QL(num_episodes, agent_, lr = 0.1, df = 0.9, eps = 0.995, is_training=False, render = False):

    #Loop used to run a series of different episods in the enviroment.
    #The agent is istantiated here along with the enviroment. The loop
    #work for both training and testing using a variable to destinguisc
    #between the two phases. Notice that it works for both tabular agents

    env =  gym.make('Taxi-v3',  render_mode='human' if render else None)
    if agent_ == "QL":
        agent = Q_Learning_Agent(env, is_training, lr, df, eps)
    elif agent_ == "SARSA":
        agent = SARSA_Agent(env, is_training, lr, df, eps)

    success = []
    rewards = []
    steps = []

    for i in tqdm(range(num_episodes)):
        observation, info = env.reset()
        done = False
        r = 0
        s = 0
        action = agent.get_action(observation)

        while not done:
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_action = agent.get_action(next_observation)

            r += reward
            s += 1

            if is_training and agent_ == "QL":
                agent.update_q_values(observation, action, reward, terminated, next_observation)
            elif is_training and agent_ == "SARSA":
                agent.update_q_values(observation, action, reward, terminated, next_observation, next_action)

            action = next_action
            observation = next_observation
            done = terminated or truncated

        rewards.append(r)
        steps.append(s)

        if terminated: success.append(1)
        else: success.append(0)

        if i%600 == 0 and is_training:
            print("  current value of epsilon =",agent.epsilon)

        agent.decay_epsilon()

    if is_training:
        agent.save_q_table()

    return [rewards, steps, success]