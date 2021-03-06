import copy
import random
import numpy as np
from collections import deque, namedtuple
import torch

class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.seed = random.seed(seed)
        self.device = device
        self.action_size = action_size
        self.batch_size = batch_size
        self.replay_memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"]) #standard S,A,R,S',done
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.replay_memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.replay_memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):#override default __len__ operator
        """Return the current size of internal memory."""
        return len(self.replay_memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

def torch_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_checkpoint(agents):
    for id, agent in enumerate(agents):
        actor_checkpoint = "checkpoint_actor_" + str(id) + ".pth"
        critic_checkpoint = "checkpoint_critic_" + str(id) + ".pth"
        torch.save(agent.actor_local.state_dict(), actor_checkpoint)
        torch.save(agent.critic_local.state_dict(), critic_checkpoint)

def restore_checkpoint(agents):
    for id, agent in enumerate(agents):
        actor_checkpoint = "checkpoint_actor_" + str(id) + ".pth"
        agent.actor_local.load_state_dict(torch.load(actor_checkpoint))