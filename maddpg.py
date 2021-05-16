# main code for Multi ddpg agent that contains the neural network setup
# see ddpg.py for other details in the network
# policy + critic update
import torch
from ddpg import DDPG
from utilities import ReplayBuffer


class MADDPG():
    def __init__(self, num_agents, state_size, action_size, random_seed, device, buffer_size=int(1e5), batch_size=128, gamma=0.99, tau=0.01):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random_seed
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.agents = [DDPG(state_size, action_size, random_seed, device) for _ in range(self.num_agents)]
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed, device)
    
    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = []
        for state, agent in zip(states, self.agents):
            action = agent.act(state, add_noise)
            actions.append(action)
        return actions
            
    def step(self, states, actions, rewards, next_states, dones):
        # push data to the buffer from each agent
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # check buffer size condition for start training
        if(len(self.memory) > self.batch_size):
            for _ in range(self.num_agents):
                experience = self.memory.sample()
                self.learn(experience, self.gamma, self.tau)
    
    def learn(self, experiences, gamma, tau):
        """Learn from an agents experiences. performs batch learning for multiple agents simultaneously"""
        for agent in self.agents:
            agent.learn(experiences, gamma, tau)

    def reset(self):
        """Reset the noise level of multiple agents to the mean value"""
        for agent in self.agents:
            agent.reset()