import os
import torch
import numpy as np
from collections import deque
from unityagents import UnityEnvironment
from maddpg import MADDPG
from utilities import torch_device, save_checkpoint
import matplotlib.pyplot as plt

# Unity environment
unity_exe = os.path.join(os.getcwd(), "data/Tennis_Linux_NoVis/Tennis.x86_64")
env = UnityEnvironment(file_name=unity_exe)

# Get env details
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
states = env_info.vector_observations
state_size = states.shape[1]
action_size = brain.vector_action_space_size
device = torch_device()

# Hyper-parameters
seed = 42               # random number generator
buffer_size = int(1e5)  # replay buffer size
batch_size = 128        # minibatch size
gamma = 0.99            # discount factor
tau = 0.01              # weight update
n_episodes = 2000       # number of episode
n_print_interval = 100  # print and save checkpoint progress

# Create multiple DDPG agents
agent = MADDPG(num_agents, state_size, action_size, seed, device, buffer_size, batch_size, gamma, tau)

# Tranining function
def train(episodes=2000, print_interval=100):
    agent_scores = deque(maxlen=100)
    all_scores = []
    for episode in range(1, episodes + 1):
        # reset all agent scores
        scores = np.zeros(num_agents)
        # reset unity env
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        # reset agent (noise)
        agent.reset()

        while True:
            # interact with env
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            # learn
            agent.step(states, actions, rewards, next_states, dones)
            # record next state cycle
            scores += rewards
            states = next_states
            # termination
            if np.any(dones):
                break

        # compute max score
        score = np.max(scores)
        agent_scores.append(score)
        all_scores.append(score)

        # print training scores
        print('\rEpisode {}\tAverage Score: {:.5f}\tMax Score: {:.5f} in 100 episodes'.format(episode,
                                                                                              np.mean(agent_scores),
                                                                                              np.max(agent_scores)),
              end="")
        if episode % print_interval == 0:
            print('\rEpisode {}\tAverage Score: {:.5f}\tMax Score: {:.5f} in 100 episodes'.format(episode,
                                                                                                  np.mean(agent_scores),
                                                                                                  np.max(agent_scores)))
            save_checkpoint(agent.agents)
        # if solved
        if np.mean(agent_scores) >= 0.5:
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.5f}'.format(episode, np.mean(agent_scores)))
            save_checkpoint(agent.agents)
            break

    return all_scores

# training
all_scores = train(n_episodes, n_print_interval)

# plot scores
plt.figure()
plt.xlabel('Episode')
plt.ylabel('Scores')
plt.plot(all_scores)
plt.savefig('tennis_scores.png')
plt.show()

# close unity environment
env.close()
