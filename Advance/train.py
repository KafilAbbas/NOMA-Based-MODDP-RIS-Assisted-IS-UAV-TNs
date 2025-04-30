import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from collections import deque
import random
from env3 import RIS_UAV_Env
from Moddpg import MO_DDPG
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

# Create environment
env = RIS_UAV_Env(num_users=3, num_ris = 1, ris_elements=4, max_steps=1000)

# Create agent (MO-DDPG)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# agent = MO_DDPG(env, weights=[0.7, 0.3],device = device)
agent = MO_DDPG(
    env,
    state_dim=env.state_dim,  # Add this attribute to your environment
    device= device,
    action_dim=2 + env.ris_elements + env.num_users,
    weights=[0.7, 0.3]
    
)

episodes = 100  # Example
start_positions = []
end_positions = []

for episode in range(episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(env.max_steps):
        # Get action dictionary from agent
        action_dict = agent.select_action(state, noise_scale=0.1)
        
        # Environment step expects dictionary action
        next_state, reward, done, _ = env.step(action_dict)
        
        # Replay buffer automatically converts to array
        agent.replay_buffer.push(state, action_dict, reward, next_state)
        agent.update()
        
        episode_reward += reward
        state = next_state

        if done:
            break

    # Rest of the code remains the same...

    # Save end position
    end_uav_pos = state['uav_position']

    # start_positions.append(start_uav_pos)
    end_positions.append(end_uav_pos)

    print(f"Episode {episode+1}:")
    # print(f"Start UAV Position: {start_uav_pos}")
    print(f"End UAV Position: {end_uav_pos}")
    print(f"Total Reward: {episode_reward:.2f}")
    print("-" * 40)

# ---- After all episodes, plot start and end positions ----

start_positions = np.array(start_positions)
end_positions = np.array(end_positions)

plt.figure(figsize=(8, 8))
plt.scatter(start_positions[:, 0], start_positions[:, 1], c='blue', label='Start Positions', marker='o')
plt.scatter(end_positions[:, 0], end_positions[:, 1], c='red', label='End Positions', marker='x')
plt.plot(start_positions[:, 0], start_positions[:, 1], 'b--', alpha=0.5)
plt.plot(end_positions[:, 0], end_positions[:, 1], 'r--', alpha=0.5)
plt.xlim(0, env.area_size)
plt.ylim(0, env.area_size)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('UAV Start and End Positions Across Episodes')
plt.grid(True)
plt.legend()
plt.show()
