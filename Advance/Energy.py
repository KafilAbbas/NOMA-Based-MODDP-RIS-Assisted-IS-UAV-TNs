import numpy as np

# Constants (adjust based on UAV specifications)
E1 = 150.0    # Propulsion power coefficient (J/(m/s))
E2 = 5.0      # Communication/hardware power (J/s)
H0 = 100.0    # Fixed altitude (m)
time_per_slot = 1.0  # Time per trajectory slot (seconds)

def calculate_uav_energy(trajectory):
    """
    Calculate total energy consumption for a fixed-wing UAV.
    
    Args:
        trajectory: List of (x, y) coordinates representing UAV positions over time.
    
    Returns:
        Total energy consumed (Joules).
    """
    total_energy = 0.0
    
    for t in range(1, len(trajectory)):
        # Previous and current positions (2D)
        q_prev = np.array(trajectory[t-1])
        q_curr = np.array(trajectory[t])
        
        # Calculate horizontal distance moved (meters)
        distance = np.linalg.norm(q_curr - q_prev)
        
        # Velocity (m/s). Fixed-wing UAVs cannot hover!
        velocity = distance / time_per_slot
        
        # Propulsion energy (dominant term)
        energy_propulsion = E1 * velocity * time_per_slot
        
        # Communication/hardware energy (minor term)
        energy_communication = E2 * time_per_slot
        
        total_energy += energy_propulsion + energy_communication
    
    return total_energy

# Example usage:
trajectory = [(0, 0), (100, 0), (100, 100), (0, 100)]  # UAV path (2D)
print(f"Total UAV Energy: {calculate_uav_energy(trajectory):.2f} J")