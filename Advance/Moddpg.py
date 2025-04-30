import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dim=256):
        super(Critic, self).__init__()
        total_state_dim = sum(np.prod(dim) for dim in state_dims.values())
        
        self.net = nn.Sequential(
            nn.Linear(total_state_dim + action_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        # Convert action dict to numpy array
        action_array = np.concatenate([v.flatten() for v in action.values()])
        
        # Convert states to numpy arrays
        state = {k: v.numpy() if torch.is_tensor(v) else v for k,v in state.items()}
        next_state = {k: v.numpy() if torch.is_tensor(v) else v for k,v in next_state.items()}
        
        self.buffer.append((state, action_array, reward, next_state))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return self._process_batch(states, actions, rewards, next_states)
    
    def __len__(self):
        return len(self.buffer)
    
    def _process_batch(self, states, actions, rewards, next_states):
        def process_states(states):
            return {k: torch.FloatTensor(np.array([s[k] for s in states])) 
                    for k in states[0].keys()}
        
        return {
            'states': process_states(states),
            'actions': torch.FloatTensor(np.array(actions)),
            'rewards': torch.FloatTensor(np.array(rewards)),
            'next_states': process_states(next_states)
        }

class MO_DDPG:
    def __init__(self, env,state_dim,action_dim,device, weights=[0.7, 0.3], lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005,  ):
        # Environment specs
        self.env = env
        self.device = device
        # self.state_dims = {k: v.shape for k,v in env.observation_space.spaces.items()}
        # self.action_dim = sum(np.prod(space.shape) for space in env.action_space.spaces.values())
        self.action_dim = action_dim
        self.state_dims = state_dim
        # Networks
        self.actor = Actor(self.state_dims, self.action_dim)
        self.actor_target = Actor(self.state_dims, self.action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(self.state_dims, self.action_dim)
        self.critic_target = Critic(self.state_dims, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        # self.weights = torch.tensor(weights, dtype=torch.float32)
        self.replay_buffer = ReplayBuffer(100000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.state_dim = sum([
        #     2,  # uav_position
        #     self.env.num_users,  # transmit_power
        #     self.env.num_users,  # direct_channels
        #     self.env.num_users,  # reflected_channels
        #     self.env.ris_elements,  # ris_phases
        #     2 + self.env.ris_elements + self.env.num_users  # previous_action
        # ])
        # Move networks to device
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def _flatten_state(self, state_dict):
        components = [
            state_dict['uav_position'].flatten(),
            state_dict['transmit_power'].flatten(),
            state_dict['direct_channels'].flatten(),
            state_dict['reflected_channels'].flatten(),
            state_dict['ris_phases'].flatten(),
            state_dict['previous_action'].flatten()
        ]
        return torch.FloatTensor(np.concatenate(components)).to(self.device)

    def select_action(self, state_dict, noise_scale=0.1):
        with torch.no_grad():
            flat_state = self._flatten_state(state_dict).unsqueeze(0)
            action = self.actor(flat_state).cpu().numpy()[0]
        
        if noise_scale != 0:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = (action + noise).clip(-1, 1)
        
        # Convert to action dictionary using environment's action space structure
        action_dict = {}
        ptr = 0
        for name, space in self.env.action_space.spaces.items():  # Changed to .items()
            dim = np.prod(space.shape)
            action_dict[name] = action[ptr:ptr+dim].reshape(space.shape)
            ptr += dim
        return action_dict

    def update(self, batch_size=128):
        if len(self.replay_buffer) < batch_size:
            return

        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        states = self._flatten_state(batch['states']).to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = self._flatten_state(batch['next_states']).to(self.device)
        
        # Scalarize rewards
        scalar_rewards = torch.sum(rewards * self.weights.unsqueeze(0), dim=1, keepdim=True)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = scalar_rewards + self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # Actor update
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        # Soft update targets
        for target, source in zip(self.actor_target.parameters(), self.actor.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
        for target, source in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)