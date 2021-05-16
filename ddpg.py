import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Actor, Critic
from utilities import OUNoise


class DDPG():
    def __init__(self, state_size, action_size, random_seed, device, lr_critic=5e-4, lr_actor=5e-4, weight_decay=0.0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.device = device
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.weight_decay = weight_decay

        # Noise proccess
        self.noise = OUNoise(action_size, random_seed)

        # Actor network
        self.actor_local = Actor(state_size, action_size, random_seed).to(self.device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic network
        self.critic_local = Critic(state_size, action_size, random_seed).to(self.device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        # Perform hard copy
        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)

    def reset(self):
        """Resets the noise process to mean"""
        self.noise.reset()

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if(add_noise):
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def learn(self, experiences, gamma, tau):
        """Update policy and value parameters using given batch of experience tuples.
           Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
           where:
               actor_target(state) -> action
               critic_target(state, action) -> Q-value
           Params
           ======
               experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
               gamma (float): discount factor
        """
        # get batch sample data
        states, actions, rewards, next_states, dones = experiences

        # ----------------------- update CRITIC (local) networks ----------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ----------------------- update ACTOR (local) networks ----------------------- #
        # Actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------- update ACTOR/CRITIC (target) networks --------------------- #
        self.soft_update(self.critic_local, self.critic_target, tau)
        self.soft_update(self.actor_local, self.actor_target, tau)
    
    def soft_update(self, source, target, tau):
        for target_param, local_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, source, target):
        for target_param, local_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(local_param.data)
