#!/usr/bin/env python
# coding: utf-8

#
# Play MADDPG Tennis Project
#
from unityagents import UnityEnvironment
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from maddpg import MADDPG
from utilities import torch_device, restore_checkpoint

# Play function
def play(env, agent, episodes=50):
    mean_scores = []
    for episode in range(1, episodes + 1):
        scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=training_mode)[brain_name]
        states = env_info.vector_observations
        num_play = 0
        while True:
            # interact with env
            actions = agent.act(states, add_noise=False)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += rewards
            states = next_states
            # termination
            if np.any(dones):
                break
            else:
                num_play+=1

        # print score for this episode (should be > 0.5)
        # compute max score
        score = np.max(scores)
        mean_scores.append(score)
        print('\rEpisode {}\t Average Score: {:.2f}\t Number of play: {}'.format(episode, np.mean(mean_scores), num_play, end=""))

    return mean_scores


# Load unity env
unity_exe = os.path.join(os.getcwd(), "data/Tennis_Linux/Tennis.x86_64")
#unity_exe = os.path.join(os.getcwd(), "data/Tennis_Linux_NoVis/Tennis.x86_64")
env = UnityEnvironment(file_name=unity_exe)

#
# Inference
#

# Get env details
training_mode = False
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=training_mode)[brain_name]
num_agents = len(env_info.agents)
states = env_info.vector_observations
state_size = states.shape[1]
action_size = brain.vector_action_space_size
seed = 42
device = torch_device()

# Create multiple DDPG agents and restore actor checkpoint
agent = MADDPG(num_agents, state_size, action_size, seed, device)
restore_checkpoint(agent.agents)


# play the game
mean_scores = play(env, agent, episodes=5)
# plot scores
plt.figure()
plt.xlabel('Episode')
plt.ylabel('Scores')
plt.plot(mean_scores)
plt.show()
env.close()
