import gymnasium as gym
import random
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

import replay_buffer as buff
import DQN_network as net

#DQN implementatio

class single_DQN_Agent:
    def __init__(self,
                env,
                replay_buffer,
                is_training,
                device = "cpu",
                lr = 0.01,
                discount_factor = 0.9,
                espsilon_decay = 0.005,
                epsilon = 0.999,
                final_epsilon = 0.1
                ):

    #Defining the class for the DQN agent, this will use only one network to
    #approximate the best policy. Policy is learn using the mean squared error
    #between the predicted q-value by the network

        self.device = device
        self.env = env

        self.replay_buffer = replay_buffer
        self.is_training = is_training

        self.lr = lr
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = espsilon_decay
        self.initial_epsilon = epsilon
        self.final_epsilon = final_epsilon

        self.input_dim = env.observation_space.n
        self.output_dim = env.action_space.n

        self.q_network = net.DQN(self.input_dim, self.output_dim, device= device)

        if not is_training:
            self.q_network.load_state_dict(torch.load("single_dqn_model.pth", map_location=self.device)) # Load to the correct device

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

    def get_action(self, state, ep):

        #Action are selected with a greedy policy at training time, while for the
        #tests we simply follow the policy learned by the network.
        #Notice that to perform epsilon decay i decided not to use another function
        #as it significatly slowed down the training process.

        self.epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon)*np.exp(-self.epsilon_decay * ep)

        #during tests or when the epsilon greedy policy tells to exploit at training time use the network outpus
        if not self.is_training or random.uniform(0, 1) > self.epsilon:

            # Ensure state is on the correct device
            return torch.argmax(self.q_network(state)).item()

        else:
            return self.env.action_space.sample()

    def network_update(self, ep):
        state, action, reward, next_state, done = self.replay_buffer.get_batch(128)

        #network prediction
        q_val = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze()


        with torch.no_grad():
            #objective of the network = max_a(Q(s',a)*discount_factor) + r
            max_next_q_values = self.q_network(next_state).max(1)[0]
            t_val = reward + self.discount_factor * max_next_q_values * (1 - done)

        loss = nn.MSELoss()(q_val, t_val)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model_param(self):
        torch.save(self.q_network.state_dict(), "single_dqn_model.pth")

##DQN with target network: agent implementation

class TargetDQN_Agent:

    #Defining the class for the DQN agent with a target network. It uses two twin
    #networks, the q-network and the target-network. The target-network is a frozen
    #copy of the q-network used to compute the targets: it is synced every 7 episodes
    #by simply copying the q-network parameters into it.

    def __init__(self,
                 env,
                 replay_buffer,
                 is_training,
                 device = "cpu",
                 lr = 0.01,
                 discount_factor = 0.9,
                 espsilon_decay = 0.005,
                 epsilon = 0.999,
                 final_epsilon = 0.1
                ):

        self.device = device
        self.env = env

        self.replay_buffer = replay_buffer
        self.is_training = is_training

        self.lr = lr
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = espsilon_decay
        self.initial_epsilon = epsilon
        self.final_epsilon = final_epsilon

        self.input_dim = env.observation_space.n
        self.output_dim = env.action_space.n

        self.q_network = net.DQN(self.input_dim, self.output_dim, device=device)
        self.t_network = net.DQN(self.input_dim, self.output_dim, device=device)

        if self.is_training:
            self.t_network.load_state_dict(self.q_network.state_dict())
        else:
            self.q_network.load_state_dict(torch.load("target_dqn_model.pth", map_location=self.device))

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

    def get_action(self, state, ep):

        #Action are selected with a greedy policy at training time, while for the
        #tests we simply follow the policy learned by the network.
        #Notice that to perform epsilon decay i decided not to use another function
        #as it significatly slowed down the training process.

        self.epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon)*np.exp(-self.epsilon_decay * ep)

        if not self.is_training or random.uniform(0,1) > self.epsilon:
            return torch.argmax(self.q_network(state)).item()
        else:
            return self.env.action_space.sample()


    def network_update(self, ep):
        state, action, reward, next_state, done = self.replay_buffer.get_batch(128)

        q_val = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze()

        
        #DQN with target network: the target is computed with the frozen copy of
        #the q-network instead of the q-network itself, which stabilises training

        with torch.no_grad():
            #the target network computes max_a Q_target(s', a), while the single DQN
            #agent uses the q-network itself for this
            max_next_q_values = self.t_network(next_state).max(1)[0]
            t_val = reward + self.discount_factor * max_next_q_values * (1 - done)
        

        loss = nn.MSELoss()(q_val, t_val)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #7 as target-network update frequency was found empirically
        if ep % 7 == 0:
            self.t_network.load_state_dict(self.q_network.state_dict())

    def save_model_param(self):
        torch.save(self.q_network.state_dict(), "target_dqn_model.pth")

class DDQN_Agent:

    #Defining the class for the DDQN agent. It uses two twin networks, similarly to 
    #the q-network and the target-network. The difference is in the way the target is 
    #score is selected.

    def __init__(self,
                 env,
                 replay_buffer,
                 is_training,
                 device = "cpu",
                 lr = 0.01,
                 discount_factor = 0.9,
                 espsilon_decay = 0.005,
                 epsilon = 0.999,
                 final_epsilon = 0.1
                ):

        self.device = device
        self.env = env

        self.replay_buffer = replay_buffer
        self.is_training = is_training

        self.lr = lr
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = espsilon_decay
        self.initial_epsilon = epsilon
        self.final_epsilon = final_epsilon

        self.input_dim = env.observation_space.n
        self.output_dim = env.action_space.n

        self.q_network = net.DQN(self.input_dim, self.output_dim, device=device)
        self.t_network = net.DQN(self.input_dim, self.output_dim, device=device)

        if self.is_training:
            self.t_network.load_state_dict(self.q_network.state_dict())
        else:
            self.q_network.load_state_dict(torch.load("ddqn_model.pth", map_location=self.device))

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

    def get_action(self, state, ep):

        #Action are selected with a greedy policy at training time, while for the
        #tests we simply follow the policy learned by the network.
        #Notice that to perform epsilon decay i decided not to use another function
        #as it significatly slowed down the training process.

        self.epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon)*np.exp(-self.epsilon_decay * ep)

        if not self.is_training or random.uniform(0,1) > self.epsilon:
            return torch.argmax(self.q_network(state)).item()
        else:
            return self.env.action_space.sample()


    def network_update(self, ep):
        state, action, reward, next_state, done = self.replay_buffer.get_batch(128)

        q_val = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze()
        
        #DDQN: the action is selected by the online network but its q value is compared 
        #against the one from same same action but computed with the target network.

        with torch.no_grad():
            best_next_actions = self.q_network(next_state).argmax(dim=1, keepdim=True)

            max_next_q_values = self.t_network(next_state).gather(1, best_next_actions).squeeze(1)

            t_val = reward + self.discount_factor * max_next_q_values * (1 - done)
        

        loss = nn.MSELoss()(q_val, t_val)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #7 as target-network update frequency was found empirically
        if ep % 7 == 0:
            self.t_network.load_state_dict(self.q_network.state_dict())

    def save_model_param(self):
        torch.save(self.q_network.state_dict(), "ddqn_model.pth")

##Loop for training and testing the DQN and target-network DQN agents

def run_single_DQL(num_episodes,
                   agent,
                   device = "cpu",
                   lr = 0.1,
                   df = 0.9,
                   eps = 0.999,
                   is_training = False,
                   render = False
                   ):

    #Loop used to run a series of different episods in the enviroment.
    #The agent is istantiated here along with the enviroment. The loop
    #work for both training and testing using a variable to destinguisc
    #between the two phases.

    env =  gym.make('Taxi-v3',  render_mode='human' if render else None)
    replay_buf = buff.Replay_Buffer(device=device)

    if agent == "DQN":
        agent = single_DQN_Agent(env,replay_buf, is_training, device, lr, df, eps)
    elif agent == "TargetDQN":
        agent = TargetDQN_Agent(env, replay_buf, is_training, device, lr, df, eps)
    elif agent == "DDQN":
        agent = DDQN_Agent(env, replay_buf, is_training, device, lr, df, eps)
    #variable and lists used to collect data to return for metrics

    steps_done = 0
    success = []
    rewards = []
    steps = []

    for i in tqdm(range(num_episodes)):
        observation, info = env.reset()
        done = False

        r = 0
        s = 0
        while not done:

            action = agent.get_action(observation, i)
            next_observation, reward, terminated, truncated, info = env.step(action)

            replay_buf.add_elem((observation, action, reward, next_observation, terminated))

            done = terminated or truncated
            observation = next_observation

            steps_done += 1
            r += reward
            s += 1
            
            
        rewards.append(r)
        steps.append(s)

        if terminated: success.append(1)
        else: success.append(0)

        #want to make sure that there are enogh samples in the replay buffer
        #before traning the agent.

        if replay_buf.get_len() >= 1000 and is_training:
            agent.network_update(i)

        #used to keep track of the epsilon-decay during training

        if i % 3000 == 0 and is_training: print("  Current value of epsilon: ", agent.epsilon)

    if is_training:
        agent.save_model_param()

    return [rewards, steps, success]
