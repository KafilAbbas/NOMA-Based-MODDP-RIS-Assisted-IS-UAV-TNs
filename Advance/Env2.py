import numpy as np
import gym
from gym import spaces
from scipy.special import j1

class VehicleUser:
    def __init__(self, area_size=1000, speed_range=(10, 30), communication_range=200):
        # Initialize position and movement parameters
        self.position = np.random.uniform(0, area_size, size=2)
        self.speed = np.random.uniform(*speed_range)
        self.max_speed = 30
        # Initial direction (radians)
        self.direction = np.random.uniform(0, 2*np.pi)
        self.steps_since_turn = 0
        # Communication range
        self.communication_range = communication_range
        # Compute initial velocity vector
        self.velocity = np.array([
            self.speed * np.cos(self.direction),
            self.speed * np.sin(self.direction)
        ], dtype=np.float32)
        # Environment bounds
        self.area_size = area_size

    def move(self, dt=1.0):
        # Update velocity vector based on current speed and direction
        self.velocity = np.array([
            self.speed * np.cos(self.direction),
            self.speed * np.sin(self.direction)
        ], dtype=np.float32)
        # Move position
        self.position = np.clip(self.position + self.velocity * dt, 0, self.area_size)
        
        # Randomly update direction every 5-10 steps
        self.steps_since_turn += 1
        if self.steps_since_turn >= np.random.randint(5, 10):
            self.direction = np.random.uniform(0, 2*np.pi)
            self.steps_since_turn = 0



class RIS_UAV_Env(gym.Env):
    def __init__(self, num_users=3, num_ris=1, ris_elements=64, max_steps = 1000):
        super(RIS_UAV_Env, self).__init__()
        
        # Environment parameters
        self.max_steps = max_steps
        self.step_count = 0
        self.num_users = num_users
        self.num_ris = num_ris
        self.ris_elements = ris_elements
        self.max_power = 100  # Watts
        self.area_size = 1000  # meters
        self.max_velocity = 30  # m/s
        self.time_step = 1  # second
        self.carrier_freq = 2e9  # 2 GHz
        self.wavelength = 3e8 / self.carrier_freq
        self.M_R = 8  # Vertical elements
        self.N_R = 8  # Horizontal elements
        self.d_x = self.wavelength/2  # Horizontal spacing
        self.d_y = self.wavelength/2  # Vertical spacing
        # Initialize VUs
        self.vus = [VehicleUser(self.area_size) for _ in range(num_users)]
        
        # Define action and observation spaces
        
        self.action_space = spaces.Dict({
            'uav_movement': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'ris_phases': spaces.Box(low=0, high=2*np.pi, shape=(ris_elements,), dtype=np.float32),
            'beamforming': spaces.Box(low=0, high=1, shape=(num_users,), dtype=np.float32)
        })
        self.last_action = None
        
        # obs_dim = 2 + 2 * self.num_users + 1 + 1
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf,
        #     shape=(obs_dim,), dtype=np.float32
        # )

        self.observation_space = spaces.Dict({
            'uav_position': spaces.Box(low=0, high=self.area_size, shape=(2,), dtype=np.float32),
            'transmit_power': spaces.Box(low=0, high=self.max_power, shape=(num_users,), dtype=np.float32),
            'direct_channels': spaces.Box(low=-np.inf, high=np.inf, 
                                        shape=(num_users, ris_elements), dtype=np.float32),
            'reflected_channels': spaces.Box(low=-np.inf, high=np.inf, 
                                           shape=(num_users, ris_elements), dtype=np.float32),
            'ris_phases': spaces.Box(low=0, high=2*np.pi, 
                                   shape=(ris_elements,), dtype=np.float32),
            'previous_action': spaces.Box(low=-1, high=1, 
                                        shape=(2 + ris_elements + num_users,), dtype=np.float32)
        })

        self.state_dim = sum(np.prod(space.shape) for space in self.observation_space.spaces.values())
        self.action_dim = sum(np.prod(space.shape) for space in self.action_space.spaces.values())
        # Initialize state variables
        self.uav_position = None
        self.ris_phases = None
        self.beamforming = None
        self.energy = 1e5  # Initial energy (Joules)
        
        # Channel parameters
        self.K_RE = 10  # Rician factor
        self.G_s_t = 30  # dBi
        self.beam_width = 0.8  # degrees

    def _update_vus(self):
        """Update vehicle user positions and velocities"""
        for vu in self.vus:
            vu.move(self.time_step)

    def _satellite_to_ris_channel(self,d_sr, f_c, G_s_t_ris_dB,):
        """
        Satellite to RIS channel G_SR.
        d_sr: distance between satellite and RIS (meters)
        f_c: carrier frequency (Hz)
        G_s_t_ris_dB: satellite transmit gain towards RIS (in dB)
        """
        c = 3e8  # Speed of light
        lambda_wave = c / f_c

        # Free space path loss
        C_sr = (lambda_wave / (4 * np.pi * d_sr))**2
        
        # Random phase
        phase = np.exp(-1j * 2 * np.pi * d_sr / lambda_wave)
        
        # Rain attenuation (Normal distribution in dB)
        xi_dB = np.random.normal(loc=0.1, scale=0.3)  # Example parameters
        xi = 10 ** (xi_dB / 10)
        
        # Transmit gain (convert from dB to linear)
        G_s_t_ris = 10 ** (G_s_t_ris_dB / 10)
        
        # Final channel
        G_SR = np.sqrt(G_s_t_ris) * C_sr * xi**-0.5 * phase
        
        return G_SR

    def _ris_to_vu_channel(self,K_RE, theta_r, phi_r, G_e, d_ris_vu):
        """
        Parameters:
        K_RE : Rician factor
        theta_r : elevation angle (radians)
        phi_r : azimuth angle (radians)
        G_e : VU receive gain
        d_ris_vu : distance between RIS and VU (meters)
        """
        # Free space loss
        C_e = (self.wavelength/(4*np.pi*d_ris_vu))**2
        
        # Steering vectors (equations 6-7)
        a_x = np.exp(1j*2*np.pi*self.d_x/self.wavelength * np.sin(theta_r)*np.cos(phi_r) * np.arange(self.M_R))
        a_y = np.exp(1j*2*np.pi*self.d_y/self.wavelength * np.sin(theta_r)*np.sin(phi_r) * np.arange(self.N_R))
        
        A = np.outer(a_x, np.conj(a_y))  # Outer product
        h_LOS = np.sqrt(G_e * C_e) * A.flatten()
        
        # NLOS component
        h_NLOS = (np.random.randn(self.ris_elements) + 
                1j*np.random.randn(self.ris_elements)) / np.sqrt(2)
        
        # Combine components
        h_RE = (np.sqrt(K_RE/(K_RE + 1)) * h_LOS + 
                np.sqrt(1/(K_RE + 1)) * h_NLOS)
        return h_RE

    def _satellite_beam_gain_matrix(self, num_beams, num_users, beam_width=0.8, max_gain=52.1):
        """
        Implements the beam gain matrix based on reference [47]
        
        Parameters:
        num_beams : number of beams (K in paper)
        num_users : number of users (N in paper)
        beam_width : 3dB beamwidth in degrees (default 0.8° for GEO satellites)
        max_gain : maximum beam gain in dBi (default 52.1dBi for typical GEO)
        
        Returns:
        b : beam gain matrix of shape (num_beams, num_users)
        """
        # Convert from dBi to linear scale
        G_max = 10**(max_gain / 10)
        
        # Generate random user angles within beam coverage
        # (In practice, these would be the actual user angles)
        theta_users = np.random.uniform(-beam_width*3, beam_width*3, num_users)
        
        # Initialize beam gain matrix
        b = np.zeros((num_beams, num_users))
        
        # Satellite beam angles (equally spaced for simplicity)
        theta_beams = np.linspace(-beam_width*3, beam_width*3, num_beams)
        
        # Calculate beam pattern for each beam-user pair
        for k in range(num_beams):
            for n in range(num_users):
                # Angle difference between beam center and user
                theta = theta_users[n] - theta_beams[k]
                
                # Normalized angle (Eq.3 in [47])
                u = 2.07123 * np.sin(np.deg2rad(theta)) / np.sin(np.deg2rad(beam_width/2))
                
                # Beam pattern (using J1 Bessel function)
                if u == 0:
                    b[k,n] = G_max
                else:
                    b[k,n] = G_max * (j1(u)/u)**2
        
        return b


    def _satellite_to_vu_channel(theta_s, d_s, G_s_t ,G_s_r_Max, antenna_diam, mu_rain, sigma_rain,self):
        """
        Parameters:
        theta_s : off-axis angle (degrees)
        d_s : distance between satellite and VU (meters)
        G_s_t : satellite launch gain
        G_s_r_Max : maximum satellite receive gain
        antenna_diam : receiver antenna diameter (meters)
        mu_rain, sigma_rain : rain attenuation parameters
        """
        # Calculate G_s_r (equation 2)
        lambda_wave = self.wavelength
        theta_a = (20*lambda_wave/antenna_diam) * np.sqrt(G_s_r_Max - (2 + 15*np.log10(antenna_diam/lambda_wave)))
        theta_b = 15.85*(antenna_diam/lambda_wave)**-0.6
        
        if 0 < theta_s < theta_a:
            G_s_r = G_s_r_Max - 2.5e-3 * (antenna_diam * theta_s / lambda_wave)**2
        elif theta_a <= theta_s < theta_b:
            G_s_r = 2 + 15*np.log10(antenna_diam/lambda_wave)
        elif theta_b <= theta_s < 48:
            G_s_r = 32 - 25*np.log10(theta_s)
        else:
            G_s_r = -10
        
        # Free space loss
        C_s = (lambda_wave/(4*np.pi*d_s))**2
        
        # Rain attenuation (lognormal in dB)
        xi_dB = np.random.lognormal(mean=mu_rain, sigma=sigma_rain)
        xi = 10**(xi_dB/10)
        
        # Beam gain (simplified circular pattern)
        # (Actual implementation would use specific antenna pattern)
        b = self._satellite_beam_gain_matrix(num_beams=1, num_users=1)  # Placeholder
        beam_gain = b[0,0]  # Get gain for this beam-user pair
        # Phase component
        phase = np.exp(-1j*2*np.pi*d_s/lambda_wave)
        
        # Combined channel
        G = np.sqrt(G_s_t * G_s_r * C_s)  * (xi**-0.5) * beam_gain**0.5 * phase
        return G


    
    def _calculate_channels(self):
        """Calculate all channel components including VU mobility"""
        channels = []
        for i, vu in enumerate(self.vus):
            # Calculate distance between UAV and VU
            d_sat_vu = np.linalg.norm(self.uav_position - vu.position)
            
            # Satellite to VU channel
            G_sv = self._satellite_to_vu_channel(d_sat_vu)
            
            # RIS to VU channel
            h_ris_vu = self._ris_to_vu_channel(vu.position)
            
            # Combined channel
            combined_channel = G_sv + h_ris_vu.T @ np.diag(np.exp(1j*self.ris_phases)) @ self._satellite_to_ris_channel()
            channels.append(np.abs(combined_channel)**2)
            
        return np.array(channels)
    

    def _calculate_direct_channels(self):
        channels = []
        for i, vu in enumerate(self.vus):
            # Calculate distance between UAV and VU
            d_sat_vu = np.linalg.norm(self.uav_position - vu.position)
            
            # Satellite to VU channel
            G_sv = self._satellite_to_vu_channel(d_sat_vu)

            channels.append(np.abs(G_sv)**2)
            
        return np.array(channels)
    

    def reset(self):
        # Reset UAV position
        self.uav_position = np.random.uniform(0, self.area_size, 2)
        
        # Reset VUs
        for vu in self.vus:
            vu.position = np.random.uniform(0, self.area_size, 2)
            vu.velocity = np.random.uniform(-vu.max_speed, vu.max_speed, 2)
        
        # Reset RIS phases
        self.ris_phases = np.random.uniform(0, 2*np.pi, self.ris_elements)
        
        # Reset beamforming weights
        self.beamforming = np.ones(self.num_users) / np.sqrt(self.num_users)
        
        # Reset energy
        self.energy = 1e5
        
        return self._get_obs()



    def _get_obs(self):
        """Construct the state vector as defined in the paper."""
        return {
            'uav_position': self.uav_position,
            'transmit_power': self.beamforming * self.max_power,
            'direct_channels': self._calculate_direct_channels(),
            'cumulative_channels': self._calculate_channels(),
            'ris_phases': self.ris_phases,
            'previous_action': self.last_action
        }
    # def _get_obs(self):
    #     # 1) UAV position
    #     uv = self.uav_position.astype(np.float32)
    #     # 2) Effective channel real & imag
    #     h_eff = self._calculate_channels()  # shape (K,)
    #     h_real = np.real(h_eff).astype(np.float32)
    #     h_imag = np.imag(h_eff).astype(np.float32)
    #     # 3) Remaining energy fraction
    #     e_frac = np.array([self.energy / self.E_max], dtype=np.float32)
    #     # 4) Normalized time index
    #     t_norm = np.array([self.t / self.Tmax], dtype=np.float32)

    #     obs = np.concatenate([uv, h_real, h_imag, e_frac, t_norm])
    #     return obs
    

    #     return {
    #         'uav_position': self.uav_position,
    #         'vu_positions': np.array([vu.position for vu in self.vus]),
    #         'vu_velocities': np.array([vu.velocity for vu in self.vus]),
    #         'channels': self._calculate_channels(),
    #         'remaining_energy': np.array([self.energy]),
    #         'connectivity': self._calculate_connectivity()
    #     }

    def step(self, action):
        # Update VU positions first
        self._update_vus()
        
        self.last_action = action 
        # Original UAV and RIS updates
        movement = action['uav_movement'] * self.max_velocity * self.time_step
        self.uav_position = np.clip(self.uav_position + movement, 0, self.area_size)
        
        # Discrete phase shifts (8 levels)
        self.ris_phases = np.round(action['ris_phases']/(2*np.pi/8)) * (2*np.pi/8) % (2*np.pi)
        
        # Power constraint
        self.beamforming = action['beamforming']
        self.beamforming /= np.linalg.norm(self.beamforming)
        
        # Calculate reward components
        connectivity = self._calculate_connectivity()
        channels = self._calculate_channels()
        rates = np.log2(1 + channels * self.beamforming**2)
        
        # Energy consumption
        prop_energy = 100 * np.linalg.norm(movement)**2 * self.time_step
        comm_energy = 50 * np.sum(self.beamforming**2) * self.time_step
        self.energy -= (prop_energy + comm_energy)
        
        # Multi-objective reward
        reward = np.sum(rates * connectivity) - 0.01*(prop_energy + comm_energy)
        
        # Termination conditions
        done = (self.energy <= 0) or (np.sum(connectivity) == 0) or (self.step_count >= self.max_steps)

        self.step_count = self.step_count + 1
        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        print(f"\nUAV Position: {self.uav_position}")
        print(f"Remaining Energy: {self.energy:.1f} J")
        print("Current Rates:", self._calculate_channels())
        
# Example usage with visualization
if __name__ == "__main__":
    env = RIS_UAV_Env(num_users=3)
    obs = env.reset()
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()
        
        if done:
            print("\nEpisode finished!")
            break
            
    env.close()