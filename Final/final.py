# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import scipy.stats as stats

# Simulation parameters from Table II
SIM_PARAMS = {
    'frequency': 2e9,  # 2 GHz
    'bandwidth': 15e6,  # 15 MHz
    'noise_power_spectral_density': -169,  # dBm/Hz
    'satellite_max_gain': 48,  # dB
    'uav_height': 1000,  # meters
    'uav_speed': 10,  # m/s
    'num_time_slots': 100,
    'num_episodes': 1000,  # Reduced for efficiency
    'steps_per_episode': 1000,
    'replay_buffer_size': 100000,
    'batch_size': 64,
    'discount_rate': 0.99,
    'learning_rate_actor': 0.0001,
    'learning_rate_critic': 0.001,
    'soft_update_tau': 0.005,
    'epsilon_start': 0.99,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'noise_variance': 1.0,
}

# System parameters
N_UAV = 1
K_VU = 6
M_RIS_DEFAULT = 16  # Default number of RIS elements
P_MAX = 100  # W
T_MAX = 100  # s
AREA_X = (-500, 500)  # m
AREA_Y = (-500, 500)  # m
L_PHASE = 8  # Discrete phase shift levels
W_PENALTY = 1000  # Penalty for constraint violation
REWARD_WEIGHTS = [0.6, 0.4]  # [w_RK, w_EK]

# Environment class
class ISUAVTNEnvironment:
    def __init__(self, use_ris=True, use_noma=True, m_ris=M_RIS_DEFAULT):
        self.use_ris = use_ris
        self.use_noma = use_noma
        self.m_ris = m_ris
        self.time_slot = 0
        self.uav_positions = np.zeros((N_UAV, 2), dtype=np.float64)  # Ensure float64
        self.vu_positions = np.random.uniform(
            low=[AREA_X[0], AREA_Y[0]], high=[AREA_X[1], AREA_Y[1]], size=(K_VU, 2)
        ).astype(np.float64)
        self.ris_phase_shifts = np.zeros((N_UAV, self.m_ris), dtype=np.float64)
        self.satellite_power = np.ones(K_VU, dtype=np.float64) * (P_MAX / K_VU)
        self.satellite_beamforming = np.ones((K_VU, 1), dtype=complex) / np.sqrt(K_VU)
        self.noise_power = (
            SIM_PARAMS['noise_power_spectral_density'] + 10 * np.log10(SIM_PARAMS['bandwidth'])
        )
        self.noise_power = 10 ** (self.noise_power / 10) / 1000  # W
        self.phase_levels = np.linspace(0, 2 * np.pi, L_PHASE, endpoint=False)
        self.reset()
        self.position_history = []

    def reset(self):
        self.uav_positions = np.array([[AREA_X[0], AREA_Y[0]] for _ in range(N_UAV)], dtype=np.float64)
        self.time_slot = 0
        if self.use_ris:
            self.ris_phase_shifts = np.random.choice(self.phase_levels, (N_UAV, self.m_ris)).astype(np.float64)
        else:
            self.ris_phase_shifts = np.zeros((N_UAV, self.m_ris), dtype=np.float64)
        state = self.get_state()
        return state

    def get_state(self):
        channel_gains = self.compute_channel_gains()
        prev_action = np.zeros(N_UAV * 2 + N_UAV * self.m_ris + K_VU * 2, dtype=np.float64)
        state = np.concatenate([
            self.satellite_power.flatten(),
            self.uav_positions.flatten(),
            prev_action,
            np.abs(channel_gains).flatten()
        ])
        return state

    def compute_channel_gains(self):
        wavelength = 3e8 / SIM_PARAMS['frequency']
        channel_gains = np.zeros((K_VU, N_UAV + 1), dtype=complex)

        for k in range(K_VU):
            # Direct link: Satellite to VU
            distance_s_vu = 36e6  # GEO distance
            free_space_loss = (wavelength / (4 * np.pi * distance_s_vu)) ** 2
            shadowing = np.random.lognormal(mean=0, sigma=0.1)
            satellite_gain = 10 ** (SIM_PARAMS['satellite_max_gain'] / 10)
            rician_k = 10
            h_los = np.sqrt(free_space_loss) * np.exp(1j * np.random.uniform(0, 2 * np.pi))
            h_nlos = np.sqrt(free_space_loss / (rician_k + 1)) * (
                np.random.randn() + 1j * np.random.randn()
            )
            h_s_k = np.sqrt(rician_k / (rician_k + 1)) * h_los + h_nlos
            channel_gains[k, 0] = h_s_k * np.sqrt(satellite_gain / shadowing)

            # Reflected link: Satellite -> RIS -> VU
            if self.use_ris:
                for n in range(N_UAV):
                    # Satellite to RIS
                    distance_s_ris = np.sqrt((36e6 - SIM_PARAMS['uav_height'])**2)
                    G_s_r = np.sqrt(free_space_loss) * np.exp(1j * np.random.uniform(0, 2 * np.pi))
                    # RIS to VU
                    distance_ris_vu = np.linalg.norm(self.uav_positions[n] - self.vu_positions[k])
                    free_space_loss_ris_vu = (wavelength / (4 * np.pi * distance_ris_vu)) ** 2
                    theta_r = np.arctan2(
                        self.vu_positions[k][1] - self.uav_positions[n][1],
                        self.vu_positions[k][0] - self.uav_positions[n][0]
                    )
                    d_x = wavelength / 2
                    a_s = np.exp(
                        1j * 2 * np.pi * d_x / wavelength * np.sin(theta_r) * np.arange(self.m_ris)
                    )
                    h_los = np.sqrt(free_space_loss_ris_vu) * a_s
                    h_nlos = np.sqrt(free_space_loss_ris_vu / (rician_k + 1)) * (
                        np.random.randn(self.m_ris) + 1j * np.random.randn(self.m_ris)
                    )
                    h_r_k = np.sqrt(rician_k / (rician_k + 1)) * h_los + h_nlos
                    if np.random.rand() < 0.1:  # Random phase shift for UAV/R
                        self.ris_phase_shifts[n] = np.random.choice(self.phase_levels, self.m_ris)
                    theta = np.diag(np.exp(1j * self.ris_phase_shifts[n]))
                    channel_gains[k, n + 1] = np.sum(h_r_k @ theta) * G_s_r  # Sum to get scalar

        return channel_gains

    def compute_data_rate(self):
        channel_gains = self.compute_channel_gains()
        rates = np.zeros(K_VU)
        synthetic_channels = np.zeros(K_VU, dtype=complex)

        for k in range(K_VU):
            synthetic_channels[k] = channel_gains[k, 0] + (np.sum(channel_gains[k, 1:], axis=0) if self.use_ris else 0)

        if self.use_noma:
            channel_norms = np.abs(synthetic_channels)
            sorted_indices = np.argsort(channel_norms)
            for i, idx in enumerate(sorted_indices):
                interference = sum(
                    self.satellite_power[j] * np.abs(synthetic_channels[idx]) ** 2
                    for j in sorted_indices[:i]
                )
                sinr = (
                    self.satellite_power[idx] * np.abs(synthetic_channels[idx]) ** 2 /
                    (interference + self.noise_power)
                )
                rates[idx] = SIM_PARAMS['bandwidth'] * np.log2(1 + sinr)
        else:
            for k in range(K_VU):
                sinr = (
                    self.satellite_power[k] * np.abs(synthetic_channels[k]) ** 2 /
                    self.noise_power
                )
                rates[k] = (SIM_PARAMS['bandwidth'] / K_VU) * np.log2(1 + sinr)

        return rates

    def compute_energy(self):
        blade_power = 100  # W
        movement_energy = blade_power * (T_MAX / SIM_PARAMS['num_time_slots'])
        return movement_energy * N_UAV

    def check_constraints(self, action):
        uav_velocities = action[:N_UAV * 2].reshape(N_UAV, 2)
        powers = action[-K_VU:]
        new_positions = self.uav_positions + uav_velocities * (T_MAX / SIM_PARAMS['num_time_slots'])
        if not all(AREA_X[0] <= pos[0] <= AREA_X[1] and AREA_Y[0] <= pos[1] <= AREA_Y[1] for pos in new_positions):
            return False
        if np.sum(powers) > P_MAX or any(p < 0 for p in powers):
            return False
        return True

    def step(self, action):
        if not self.check_constraints(action):
            reward = np.array([-W_PENALTY, -W_PENALTY])
            done = True
            next_state = self.get_state()
            return next_state, np.dot(REWARD_WEIGHTS, reward), done

        uav_velocities = action[:N_UAV * 2].reshape(N_UAV, 2)
        ris_phases = action[N_UAV * 2:N_UAV * (2 + self.m_ris)].reshape(N_UAV, self.m_ris)
        beamforming = action[N_UAV * (2 + self.m_ris):N_UAV * (2 + self.m_ris) + K_VU]
        powers = action[-K_VU:]

        self.uav_positions += uav_velocities * (T_MAX / SIM_PARAMS['num_time_slots'])
        self.uav_positions = np.clip(self.uav_positions, [AREA_X[0], AREA_Y[0]], [AREA_X[1], AREA_Y[1]])
        self.position_history.append(self.uav_positions.copy())  # Record new position
        if self.use_ris:
            self.ris_phase_shifts = np.clip(ris_phases, 0, 2 * np.pi)

        beamforming_complex = beamforming + 1j * beamforming
        beamforming_norm = beamforming_complex / np.linalg.norm(beamforming_complex)
        self.satellite_beamforming = beamforming_norm.reshape(K_VU, 1)

        powers = np.clip(powers, 0, P_MAX / K_VU)
        powers = powers * (P_MAX / np.sum(powers))
        self.satellite_power = powers

        rates = self.compute_data_rate()
        energy = self.compute_energy()
        sum_rate = np.sum(rates)
        energy_efficiency = sum_rate / energy if energy > 0 else 0
        reward = np.array([sum_rate / 1e6, energy_efficiency / 1e3])  # Scale for stability
        weighted_reward = np.dot(REWARD_WEIGHTS, reward)

        self.time_slot += 1
        done = self.time_slot >= SIM_PARAMS['num_time_slots']
        next_state = self.get_state()

        return next_state, weighted_reward, done

# Neural Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0, std=np.sqrt(2 / m.in_features))

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0, std=np.sqrt(2 / m.in_features))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# MO-DDPG Agent
class MODDPGAgent:
    def __init__(self, state_dim, action_dim, use_ris=True, m_ris=M_RIS_DEFAULT):
        self.use_ris = use_ris
        self.m_ris = m_ris
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=SIM_PARAMS['learning_rate_actor'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=SIM_PARAMS['learning_rate_critic'])

        self.replay_buffer = deque(maxlen=SIM_PARAMS['replay_buffer_size'])
        self.epsilon = SIM_PARAMS['epsilon_start']
        self.action_dim = action_dim

    def select_action(self, state):
        self.actor.eval()
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).squeeze(0).detach().numpy()
        self.actor.train()
        if np.random.rand() < self.epsilon:
            noise = np.random.normal(0, SIM_PARAMS['noise_variance'], self.action_dim)
            action += noise
        uav_velocities = action[:N_UAV * 2].reshape(N_UAV, 2)
        directions = np.sign(uav_velocities) * SIM_PARAMS['uav_speed']
        action[:N_UAV * 2] = directions.flatten()
        if self.use_ris:
            ris_phases = action[N_UAV * 2:N_UAV * (2 + self.m_ris)].reshape(N_UAV, self.m_ris)
            ris_phases = np.digitize(ris_phases, np.linspace(-1, 1, L_PHASE)) * (2 * np.pi / L_PHASE)
            action[N_UAV * 2:N_UAV * (2 + self.m_ris)] = ris_phases.flatten()
        return np.clip(action, -1, 1)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < SIM_PARAMS['batch_size']:
            return

        batch = np.random.choice(len(self.replay_buffer), SIM_PARAMS['batch_size'])
        states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[i] for i in batch])

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)  # [batch_size] -> [batch_size, 1]
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)  # [batch_size] -> [batch_size, 1]

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * SIM_PARAMS['discount_rate'] * target_Q

        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                SIM_PARAMS['soft_update_tau'] * param.data +
                (1 - SIM_PARAMS['soft_update_tau']) * target_param.data
            )
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                SIM_PARAMS['soft_update_tau'] * param.data +
                (1 - SIM_PARAMS['soft_update_tau']) * target_param.data
            )

        self.epsilon = max(SIM_PARAMS['epsilon_min'], self.epsilon * SIM_PARAMS['epsilon_decay'])


# %%
def train_agent(config):
    env = ISUAVTNEnvironment(
        use_ris=config['use_ris'],
        use_noma=config['use_noma'],
        m_ris=config.get('m_ris', M_RIS_DEFAULT)
    )
    state_dim = K_VU + N_UAV*2 + (N_UAV*2 + N_UAV*env.m_ris + K_VU*2) + K_VU*(N_UAV+1)
    action_dim = N_UAV*2 + N_UAV*env.m_ris + K_VU + K_VU
    agent = MODDPGAgent(state_dim, action_dim, use_ris=env.use_ris, m_ris=env.m_ris)

    data_rates = []
    energy_efficiencies = []
    throughputs = []
    propulsion_energies = []

    for episode in range(SIM_PARAMS['num_episodes']):
        state = env.reset()
        episode_rates = []
        episode_energies = []

        for step in range(SIM_PARAMS['steps_per_episode'] // SIM_PARAMS['num_time_slots']):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            rates = env.compute_data_rate()
            energy = env.compute_energy()
            episode_rates.append(np.sum(rates))
            episode_energies.append(energy)

            state = next_state
            if done:
                break

        data_rates.append(np.mean(episode_rates))
        energy_efficiency = np.mean(episode_rates) / np.mean(episode_energies)
        energy_efficiencies.append(energy_efficiency)
        throughputs.append(np.sum(episode_rates) * (T_MAX/SIM_PARAMS['num_time_slots']))
        propulsion_energies.append(np.mean(episode_energies))

    return {
        'data_rates': data_rates,
        'energy_efficiencies': energy_efficiencies,
        'throughputs': throughputs,
        'propulsion_energies': propulsion_energies,
        'config_name': config['name']
    }

# %%
configs = [
    {'use_ris': True, 'use_noma': True, 'name': 'RIS-NOMA'},
    {'use_ris': True, 'use_noma': False, 'name': 'RIS-OMA'},
    {'use_ris': False, 'use_noma': True, 'name': 'NoRIS-NOMA'},
    {'use_ris': False, 'use_noma': False, 'name': 'NoRIS-OMA'},
    {'use_ris': True, 'use_noma': True, 'name': 'UAV/R', 'random_phase': True}
]

training_results = []
for config in configs:
    print(f"Training {config['name']}...")
    results = train_agent(config)
    training_results.append(results)
    

# %%
# Plot CDF of Data Rates
plt.figure(figsize=(12, 8))
for result in training_results:
    data = result['data_rates']
    x = np.sort(data)
    y = np.arange(1, len(x)+1)/len(x)
    plt.plot(x/1e6, y, label=result['config_name'])
plt.title('CDF of Data Rate')
plt.xlabel('Data Rate (Mbps)')
plt.ylabel('CDF')
plt.legend()
plt.grid(True)
plt.show()
# Plot Energy Efficiency CDF
plt.figure(figsize=(12, 8))
for result in training_results:
    data = result['energy_efficiencies']
    x = np.sort(data)
    y = np.arange(1, len(x)+1)/len(x)
    plt.plot(x, y, label=result['config_name'])
plt.title('CDF of Energy Efficiency')
plt.xlabel('Energy Efficiency (bps/J)')
plt.ylabel('CDF')
plt.legend()
plt.grid(True)
plt.show()
# Plot Sum Rate vs RIS Elements
m_values = [8, 16, 32, 64]
plt.figure(figsize=(10, 6))
for config in configs:
    sum_rates = []
    for m in m_values:
        # Need to modify train_agent to handle different M values
        result = train_agent({**config, 'm_ris': m})
        sum_rates.append(np.mean(result['data_rates']))
    plt.plot(m_values, sum_rates, marker='o', label=config['name'])
plt.title('Sum Rate vs Number of RIS Elements')
plt.xlabel('Number of RIS Elements')
plt.ylabel('Average Sum Rate (bps)')
plt.legend()
plt.grid(True)
plt.show()


